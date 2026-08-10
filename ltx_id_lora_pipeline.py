"""ID-LoRA two-stage pipeline on ltx-2-mlx.

Port of ID-LoRA-2.3 ``inference_two_stage.py``:

* Stage 1 — dev transformer + ID-LoRA, first-frame I2V, reference audio at
  **negative temporal positions** (prefix ``[ref ‖ target]``), video/audio CFG,
  STG, modality guidance, and **identity guidance**.
* Stage 2 — drop ID-LoRA, fuse distilled LoRA, 2× spatial upsample, freeze
  stage-1 audio latent, short distilled refine.

Reference audio is identity context only; generated speech comes from the
structured prompt (``[VISUAL] / [SPEECH] / [SOUNDS]``), not a remux of the WAV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import tqdm as tqdm_lib
from mlx_arsenal.diffusion import euler_step

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, apply_denoise_mask
from ltx_core_mlx.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core_mlx.loader import (
    LTXV_LORA_BLOCK_PREFIX,
    LTXV_LORA_COMFY_RENAMING_MAP,
    LoraStateDictWithStrength,
    SafetensorsStateDictLoader,
    StateDict,
    apply_loras,
)
from ltx_core_mlx.model.audio_vae import encode_audio
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.audio import load_audio
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import (
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_positions,
)
from ltx_core_mlx.utils.weights import apply_quantization
from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS, ltx2_schedule
from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.samplers import DenoiseOutput, _compute_per_token_timesteps, _is_uniform_mask

logger = logging.getLogger(__name__)

_mx_eval = getattr(mx, "eval")  # noqa: B009

DEFAULT_STAGE1_STEPS = 30
DEFAULT_VIDEO_CFG = 3.0
DEFAULT_AUDIO_CFG = 7.0
DEFAULT_STG_SCALE = 1.0
DEFAULT_IDENTITY_GUIDANCE_SCALE = 3.0
DEFAULT_MODALITY_SCALE = 3.0
DEFAULT_RESCALE_SCALE = 0.7
DEFAULT_DISTILLED_LORA_STRENGTH = 0.8
DEFAULT_NUM_FRAMES = 121
DEFAULT_FRAME_RATE = 24.0
MAX_STAGE1_LONG_SIDE = 512
MAX_STAGE1_PIXELS = 576 * 1024
RESOLUTION_DIVISOR = 32


def _resolve_dev_transformer(model_dir: Path) -> str:
    if (model_dir / "transformer-dev.safetensors").exists():
        return "transformer-dev.safetensors"
    raise RuntimeError(
        "ID-LoRA requires transformer-dev.safetensors in the model directory "
        "(use dgrauet/ltx-2.3-mlx or ltx-2.3-mlx-q8 — distilled-only checkpoints are not enough)."
    )


def snap_to_divisor(value: int, divisor: int = RESOLUTION_DIVISOR) -> int:
    return max(int(round(value / divisor)) * divisor, divisor)


def compute_id_lora_stage1_resolution(
    src_h: int,
    src_w: int,
    *,
    max_long: int = MAX_STAGE1_LONG_SIDE,
    max_pixels: int = MAX_STAGE1_PIXELS,
    divisor: int = RESOLUTION_DIVISOR,
) -> tuple[int, int]:
    """Match upstream ``compute_resolution_match_aspect`` for stage-1 pixels."""
    src_h = max(1, int(src_h))
    src_w = max(1, int(src_w))
    scale = max_long / max(src_h, src_w)
    pixel_scale = (max_pixels / (src_h * src_w)) ** 0.5
    scale = min(scale, pixel_scale)
    return (
        snap_to_divisor(int(round(src_h * scale)), divisor),
        snap_to_divisor(int(round(src_w * scale)), divisor),
    )


def patchify_id_lora_audio_reference_latent(
    vae_latents: mx.array,
    audio_patchifier,
    *,
    negative_positions: bool = True,
) -> tuple[mx.array, mx.array]:
    """Patchify reference audio and optionally shift RoPE into negative time.

    Same position math as LipDub's helper; callers must **prefix** tokens so
    identity-guidance slices (``[:, ref_len:]``) stay correct.
    """
    tokens, t_count = audio_patchifier.patchify(vae_latents)
    positions = compute_audio_positions(t_count)  # (1, T, 1)
    if negative_positions:
        aud_dur = float(mx.max(positions).item())
        positions = positions - (aud_dur + 0.04)
    return tokens, positions.astype(mx.float32)


def prefix_audio_reference_state(
    state: LatentState,
    *,
    patchified: mx.array,
    positions: mx.array,
    strength: float = 1.0,
) -> LatentState:
    """Prefix reference audio tokens: ``[ref ‖ target]`` (ID-LoRA layout)."""
    num_ref = int(patchified.shape[1])
    mask_value = 1.0 - float(strength)
    ref_mask = mx.full((state.denoise_mask.shape[0], num_ref, 1), mask_value)
    new_positions = None
    if state.positions is not None:
        new_positions = mx.concatenate([positions.astype(mx.float32), state.positions], axis=1)
    return LatentState(
        latent=mx.concatenate([patchified, state.latent], axis=1),
        clean_latent=mx.concatenate([patchified, state.clean_latent], axis=1),
        denoise_mask=mx.concatenate([ref_mask, state.denoise_mask], axis=1),
        positions=new_positions,
        attention_mask=None,
    )


def strip_prefix_audio_tokens(latent: mx.array, ref_len: int) -> mx.array:
    """Drop leading reference-audio tokens after stage 1."""
    if ref_len <= 0:
        return latent
    return latent[:, ref_len:, :]


def id_lora_identity_guidance_denoise_loop(
    model: X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    video_guider_factory: MultiModalGuiderFactory,
    audio_guider_factory: MultiModalGuiderFactory,
    sigmas: list[float],
    *,
    ref_audio_len: int = 0,
    identity_guidance_scale: float = DEFAULT_IDENTITY_GUIDANCE_SCALE,
    show_progress: bool = True,
) -> DenoiseOutput:
    """Guided Euler loop with optional identity guidance on prefixed audio refs."""
    video_positions = video_state.positions
    audio_positions = audio_state.positions
    video_attention_mask = video_state.attention_mask
    audio_attention_mask = audio_state.attention_mask

    video_x = video_state.latent
    audio_x = audio_state.latent
    video_uniform = _is_uniform_mask(video_state.denoise_mask)
    audio_uniform = _is_uniform_mask(audio_state.denoise_mask)

    steps = list(zip(sigmas[:-1], sigmas[1:]))
    # Look up ``tqdm.tqdm`` at call time so ``LocalVideoGenerator._track_model_progress``
    # can patch the class for Web UI / WebSocket step keepalives (a frozen
    # ``from tqdm import …`` binding would stall the UI on "GPU assigned — starting…").
    # mininterval=0 so every denoise step publishes even when steps are fast.
    iterator = tqdm_lib.tqdm(
        steps,
        desc="ID-LoRA stage 1",
        disable=not show_progress,
        mininterval=0,
    )

    for sigma, sigma_next in iterator:
        video_guider = video_guider_factory.build_from_sigma(sigma)
        audio_guider = audio_guider_factory.build_from_sigma(sigma)

        sigma_arr = mx.array([sigma], dtype=mx.bfloat16)
        batch = video_x.shape[0]
        base_kwargs: dict = dict(
            video_latent=video_x,
            audio_latent=audio_x,
            sigma=mx.broadcast_to(sigma_arr, (batch,)),
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_attention_mask=video_attention_mask,
            audio_attention_mask=audio_attention_mask,
        )
        if not video_uniform:
            base_kwargs["video_timesteps"] = _compute_per_token_timesteps(sigma, video_state.denoise_mask)
        if not audio_uniform:
            base_kwargs["audio_timesteps"] = _compute_per_token_timesteps(sigma, audio_state.denoise_mask)

        cond_kwargs = {
            **base_kwargs,
            "video_text_embeds": video_text_embeds,
            "audio_text_embeds": audio_text_embeds,
        }
        cond_v, cond_a = model(**cond_kwargs)

        neg_v: mx.array | float = 0.0
        neg_a: mx.array | float = 0.0
        if video_guider.do_unconditional_generation() or audio_guider.do_unconditional_generation():
            neg_v_embeds = (
                video_guider.negative_context if video_guider.negative_context is not None else video_text_embeds
            )
            neg_a_embeds = (
                audio_guider.negative_context if audio_guider.negative_context is not None else audio_text_embeds
            )
            neg_v, neg_a = model(
                **{
                    **base_kwargs,
                    "video_text_embeds": neg_v_embeds,
                    "audio_text_embeds": neg_a_embeds,
                }
            )

        ptb_v: mx.array | float = 0.0
        ptb_a: mx.array | float = 0.0
        if video_guider.do_perturbed_generation() or audio_guider.do_perturbed_generation():
            perturbation_list: list[Perturbation] = []
            if video_guider.do_perturbed_generation():
                perturbation_list.append(
                    Perturbation(
                        type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                        blocks=video_guider.params.stg_blocks,
                    )
                )
            if audio_guider.do_perturbed_generation():
                perturbation_list.append(
                    Perturbation(
                        type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                        blocks=audio_guider.params.stg_blocks,
                    )
                )
            ptb_v, ptb_a = model(
                **{
                    **base_kwargs,
                    "video_text_embeds": video_text_embeds,
                    "audio_text_embeds": audio_text_embeds,
                    "perturbations": BatchedPerturbationConfig(
                        perturbations=[PerturbationConfig(perturbations=perturbation_list)] * batch
                    ),
                }
            )

        mod_v: mx.array | float = 0.0
        mod_a: mx.array | float = 0.0
        if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
            mod_list = [
                Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
            ]
            mod_v, mod_a = model(
                **{
                    **base_kwargs,
                    "video_text_embeds": video_text_embeds,
                    "audio_text_embeds": audio_text_embeds,
                    "perturbations": BatchedPerturbationConfig(
                        perturbations=[PerturbationConfig(perturbations=mod_list)] * batch
                    ),
                }
            )

        video_x0 = video_guider.calculate(cond_v, neg_v, ptb_v, mod_v)
        audio_x0 = audio_guider.calculate(cond_a, neg_a, ptb_a, mod_a)

        if (
            identity_guidance_scale > 0
            and ref_audio_len > 0
            and audio_x.shape[1] > ref_audio_len
            and audio_positions is not None
        ):
            tgt_audio = audio_x[:, ref_audio_len:, :]
            tgt_positions = audio_positions[:, ref_audio_len:, :]
            tgt_mask = audio_state.denoise_mask[:, ref_audio_len:, :]
            noref_kwargs: dict = dict(
                video_latent=video_x,
                audio_latent=tgt_audio,
                sigma=mx.broadcast_to(sigma_arr, (batch,)),
                video_positions=video_positions,
                audio_positions=tgt_positions,
                video_attention_mask=video_attention_mask,
                audio_attention_mask=None,
                video_text_embeds=video_text_embeds,
                audio_text_embeds=audio_text_embeds,
            )
            if not video_uniform:
                noref_kwargs["video_timesteps"] = base_kwargs["video_timesteps"]
            if not _is_uniform_mask(tgt_mask):
                noref_kwargs["audio_timesteps"] = _compute_per_token_timesteps(sigma, tgt_mask)
            _, noref_a = model(**noref_kwargs)
            id_delta = identity_guidance_scale * (cond_a[:, ref_audio_len:, :] - noref_a)
            audio_x0 = mx.concatenate(
                [audio_x0[:, :ref_audio_len, :], audio_x0[:, ref_audio_len:, :] + id_delta],
                axis=1,
            )

        video_x0 = apply_denoise_mask(video_x0, video_state.clean_latent, video_state.denoise_mask)
        audio_x0 = apply_denoise_mask(audio_x0, audio_state.clean_latent, audio_state.denoise_mask)

        video_x = euler_step(video_x, video_x0, sigma, sigma_next)
        audio_x = euler_step(audio_x, audio_x0, sigma, sigma_next)
        mx.async_eval(video_x, audio_x)

    aggressive_cleanup()
    return DenoiseOutput(video_latent=video_x, audio_latent=audio_x)


class IDLoraTwoStagesPipeline(TI2VidTwoStagesPipeline):
    """Two-stage ID-LoRA: identity audio IC + first-frame I2V → distilled refine."""

    def __init__(
        self,
        model_dir: str,
        lora_paths: list[tuple[str, float]] | None = None,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        distilled_lora_strength: float = DEFAULT_DISTILLED_LORA_STRENGTH,
    ):
        if not lora_paths or len(lora_paths) != 1:
            raise ValueError("ID-LoRA requires exactly one LoRA (the ID-LoRA adapter).")
        model_path = Path(model_dir)
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
            dev_transformer=_resolve_dev_transformer(model_path),
            distilled_lora_strength=distilled_lora_strength,
        )
        self._id_lora = [(str(p), float(s)) for p, s in lora_paths]
        # Own fusion so ``_apply_pending_loras`` never reloads a clean DiT.
        self._lora_paths = list(self._id_lora)
        self._loras_fused = False
        self.pipeline_type = "id_lora"

    def _fuse_id_lora(self) -> None:
        if self._loras_fused or not self._id_lora:
            return
        assert self.dit is not None

        if self.low_ram_streaming:
            from ltx_core_mlx.loader.block_streaming import BlockLoraSource

            sources: list = list(object.__getattribute__(self.dit, "_lora_sources"))
            for lora_path, strength in self._id_lora:
                sources.append(
                    BlockLoraSource(
                        lora_path,
                        block_prefix=LTXV_LORA_BLOCK_PREFIX,
                        strength=strength,
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                    )
                )
                logger.info("ID-LoRA: attached stream %s (strength=%.3f)", Path(lora_path).name, strength)
            object.__setattr__(self.dit, "_lora_sources", sources)
            self._loras_fused = True
            return

        import mlx.utils

        model_weights = dict(mlx.utils.tree_flatten(self.dit.parameters()))
        model_sd = StateDict(sd=model_weights, size=0, dtype=set())
        loader = SafetensorsStateDictLoader()
        lora_sds = []
        for lora_path, strength in self._id_lora:
            lora_sd = loader.load(lora_path, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            logger.info("ID-LoRA: loaded %s (strength=%.3f)", Path(lora_path).name, strength)
            lora_sds.append(LoraStateDictWithStrength(state_dict=lora_sd, strength=strength))
        fused_sd = apply_loras(model_sd=model_sd, lora_sd_and_strengths=lora_sds)
        apply_quantization(self.dit, fused_sd.sd)
        self.dit.load_weights(list(fused_sd.sd.items()))
        aggressive_cleanup()
        self._loras_fused = True
        logger.info("ID-LoRA: fused adapter into dev transformer for stage 1")

    def _encode_reference_audio(self, audio_path: str) -> tuple[mx.array, mx.array]:
        audio_data = load_audio(audio_path, target_sample_rate=16000, mono=False)
        if audio_data is None:
            raise ValueError(f"No audio stream found in {audio_path}")

        def _encode(encoder, processor) -> mx.array:
            latent = encode_audio(audio_data.waveform, audio_data.sample_rate, encoder, processor)
            _mx_eval(latent)
            return latent

        vae_latent = self.audio_conditioner(_encode, free_after=False)
        return patchify_id_lora_audio_reference_latent(
            vae_latent,
            self.audio_patchifier,
            negative_positions=True,
        )

    def _reload_clean_dev_with_distilled(self) -> None:
        """Drop ID-LoRA and load clean dev + distilled LoRA for stage 2."""
        self.dit = None
        self._loras_fused = False
        aggressive_cleanup()
        self.dit = self._load_dev_transformer()
        assert self.dit is not None
        self._fuse_distilled_lora(self.dit)
        logger.info(
            "ID-LoRA stage 2: clean dev + distilled LoRA (strength=%.2f)",
            self._distilled_lora_strength,
        )

    def load(self) -> None:
        if self._loaded:
            return
        if self.dit is None:
            self.dit = self._load_dev_transformer()
        self._fuse_id_lora()
        self._load_vae_encoder()
        if self.upsampler is None:
            self._load_upsampler()
        self._loaded = True

    def generate_id_lora(
        self,
        prompt: str,
        image: str,
        audio_path: str,
        height: int,
        width: int,
        num_frames: int = DEFAULT_NUM_FRAMES,
        *,
        frame_rate: float = DEFAULT_FRAME_RATE,
        seed: int = 42,
        stage1_steps: int = DEFAULT_STAGE1_STEPS,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_VIDEO_CFG,
        audio_cfg_scale: float = DEFAULT_AUDIO_CFG,
        stg_scale: float = DEFAULT_STG_SCALE,
        identity_guidance_scale: float = DEFAULT_IDENTITY_GUIDANCE_SCALE,
    ) -> tuple[mx.array, mx.array]:
        """Run two-stage ID-LoRA. ``height``/``width`` are stage-1 pixel sizes."""
        if not image:
            raise ValueError("ID-LoRA requires a first-frame reference image")
        if not audio_path:
            raise ValueError("ID-LoRA requires reference audio")

        stage1_h = snap_to_divisor(int(height))
        stage1_w = snap_to_divisor(int(width))
        if stage1_h != height or stage1_w != width:
            logger.info("ID-LoRA: snapped stage-1 size %sx%s → %sx%s", height, width, stage1_h, stage1_w)

        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)

        ref_audio_tokens, ref_audio_positions = self._encode_reference_audio(audio_path)
        ref_audio_len = int(ref_audio_tokens.shape[1])
        if self.low_memory:
            self.audio_conditioner.free()
            aggressive_cleanup()

        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None
        assert self.upsampler is not None

        f_lat, h_lat, w_lat = compute_video_latent_shape(num_frames, stage1_h, stage1_w)
        video_shape = (1, f_lat * h_lat * w_lat, 128)
        audio_t = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_t, 128)

        video_positions = compute_video_positions(f_lat, h_lat, w_lat, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_t)

        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        conditionings_1 = combined_image_conditionings(
            [ImageConditioningInput(path=image, frame_idx=0, strength=1.0)],
            enc_h=stage1_h,
            enc_w=stage1_w,
            spatial_dims=(f_lat, h_lat, w_lat),
            video_encoder=self.vae_encoder,
            frame_rate=frame_rate,
        )

        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=conditionings_1,
            spatial_dims=(f_lat, h_lat, w_lat),
            positions=video_positions,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(f_lat, h_lat, w_lat),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state = prefix_audio_reference_state(
            audio_state,
            patchified=ref_audio_tokens,
            positions=ref_audio_positions,
            strength=1.0,
        )

        # Negative-position sanity: ref max < 0 and no overlap with target min.
        if audio_state.positions is not None and ref_audio_len > 0:
            ref_max = float(mx.max(audio_state.positions[:, :ref_audio_len, :]).item())
            tgt_min = float(mx.min(audio_state.positions[:, ref_audio_len:, :]).item())
            if not (ref_max < 0.0 <= tgt_min or ref_max < tgt_min):
                logger.warning(
                    "ID-LoRA: unexpected audio position ranges ref_max=%.4f tgt_min=%.4f",
                    ref_max,
                    tgt_min,
                )

        num_tokens = f_lat * h_lat * w_lat
        sigmas_1 = ltx2_schedule(stage1_steps, num_tokens=num_tokens)

        video_guider_params = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=DEFAULT_RESCALE_SCALE,
            modality_scale=DEFAULT_MODALITY_SCALE,
            stg_blocks=[28],
        )
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=audio_cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=DEFAULT_RESCALE_SCALE,
            modality_scale=DEFAULT_MODALITY_SCALE,
            stg_blocks=[28],
        )
        video_factory = create_multimodal_guider_factory(video_guider_params, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_guider_params, negative_context=neg_audio_embeds)

        x0_model = X0Model(self.dit)
        self._pre_denoise_flush(video_state, audio_state)
        logger.info(
            "ID-LoRA stage 1: %dx%d, %d frames, %d steps, ref_audio_tokens=%d, "
            "video_cfg=%.1f audio_cfg=%.1f identity=%.1f",
            stage1_h,
            stage1_w,
            num_frames,
            stage1_steps,
            ref_audio_len,
            cfg_scale,
            audio_cfg_scale,
            identity_guidance_scale,
        )
        output_1 = id_lora_identity_guidance_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas_1,
            ref_audio_len=ref_audio_len,
            identity_guidance_scale=identity_guidance_scale,
        )
        if self.low_memory:
            aggressive_cleanup()

        gen_video = output_1.video_latent[:, : num_tokens, :]
        s1_audio = strip_prefix_audio_tokens(output_1.audio_latent, ref_audio_len)
        s1_audio = s1_audio[:, :audio_t, :]

        video_half = self.video_patchifier.unpatchify(gen_video, (f_lat, h_lat, w_lat))
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx).transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = self.vae_encoder.normalize_latent(video_upscaled.transpose(0, 2, 3, 4, 1))
        video_upscaled = video_up_mlx.transpose(0, 4, 1, 2, 3)
        _mx_eval(video_upscaled)

        h_full = h_lat * 2
        w_full = w_lat * 2
        stage2_h = stage1_h * 2
        stage2_w = stage1_w * 2

        conditionings_2 = combined_image_conditionings(
            [ImageConditioningInput(path=image, frame_idx=0, strength=1.0)],
            enc_h=stage2_h,
            enc_w=stage2_w,
            spatial_dims=(f_lat, h_full, w_full),
            video_encoder=self.vae_encoder,
            frame_rate=frame_rate,
        )

        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None
            aggressive_cleanup()

        self._reload_clean_dev_with_distilled()
        assert self.dit is not None

        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]
        video_positions_2 = compute_video_positions(f_lat, h_full, w_full, frame_rate=frame_rate)
        audio_positions_2 = compute_audio_positions(audio_t)

        video_state_2 = create_noised_state(
            base_shape=video_tokens.shape,
            conditionings=conditionings_2,
            spatial_dims=(f_lat, h_full, w_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens,
            legacy_scalar_blend=True,
        )
        # Freeze stage-1 audio (noise_scale=0, denoise_mask=0).
        audio_state_2 = create_noised_state(
            base_shape=s1_audio.shape,
            conditionings=[],
            spatial_dims=(f_lat, h_full, w_full),
            positions=audio_positions_2,
            seed=seed + 3,
            sigma=0.0,
            initial_latent=s1_audio,
            legacy_scalar_blend=True,
        )
        audio_state_2 = LatentState(
            latent=audio_state_2.latent,
            clean_latent=audio_state_2.clean_latent,
            denoise_mask=mx.zeros_like(audio_state_2.denoise_mask),
            positions=audio_state_2.positions,
            attention_mask=audio_state_2.attention_mask,
        )

        from ltx_pipelines_mlx.utils.samplers import denoise_loop

        x0_model_2 = X0Model(self.dit)
        self._pre_denoise_flush(video_state_2, audio_state_2)
        logger.info(
            "ID-LoRA stage 2: refining at %dx%d (%d distilled steps, audio frozen)",
            stage2_h,
            stage2_w,
            len(sigmas_2) - 1,
        )
        output_2 = denoise_loop(
            model=x0_model_2,
            video_state=video_state_2,
            audio_state=audio_state_2,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_2,
        )
        return output_2.video_latent, output_2.audio_latent

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        image: str,
        audio_path: str,
        height: int,
        width: int,
        num_frames: int = DEFAULT_NUM_FRAMES,
        *,
        frame_rate: float = DEFAULT_FRAME_RATE,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_VIDEO_CFG,
        audio_cfg_scale: float = DEFAULT_AUDIO_CFG,
        stg_scale: float = DEFAULT_STG_SCALE,
        identity_guidance_scale: float = DEFAULT_IDENTITY_GUIDANCE_SCALE,
        num_steps: int | None = None,
        **_ignored,
    ) -> str:
        steps = int(stage1_steps if stage1_steps is not None else (num_steps or DEFAULT_STAGE1_STEPS))
        video_latent, audio_latent = self.generate_id_lora(
            prompt=prompt,
            image=image,
            audio_path=audio_path,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            stg_scale=stg_scale,
            identity_guidance_scale=identity_guidance_scale,
        )

        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.image_conditioner.free()
            self.upsampler = None
            self._loaded = False
            self._loras_fused = False
            aggressive_cleanup()

        self._load_decoders()
        result = self._decode_and_save_video(video_latent, audio_latent, output_path, frame_rate=frame_rate)
        if self.low_memory:
            self.vae_decoder = None
            self.audio_decoder = None
            self.vocoder = None
            aggressive_cleanup()
        return result
