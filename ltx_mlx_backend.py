# SPDX-License-Identifier: Apache-2.0
"""
Local LTX-2.3 generation using ``ltx-2-mlx`` (MLX on Apple Silicon).

See: https://github.com/dgrauet/ltx-2-mlx
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import functools
import inspect
import logging
import mimetypes
import os
import random
import re
import shutil
import subprocess
from ltx_paths import mk_scratch_dir, mk_scratch_file
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname, urlopen

log = logging.getLogger("fvserver")

LTX2_SPATIAL_ALIGN = 32
IC_LORA_IMAGE_CRF = 33  # ltx_pipelines_mlx.utils.media_io.DEFAULT_IMAGE_CRF
LTX2_MLX_GIT_TAG = "v0.14.19"

CHAIN_METHOD_AUTOCONTINUE = "autocontinue"
CHAIN_METHOD_NATIVE_EXTEND = "native_extend"
# ltx-2-mlx extend/retake: RetakePipeline + dev transformer + CFG (see docs/PIPELINES.md).
RETAKE_EXTEND_DEFAULT_CFG = 3.0
RETAKE_EXTEND_DEFAULT_STG = 1.0
# Standalone extend: latent tokens (~8 pixel frames each). 15 ≈ ~5s added @ 24fps.
DEFAULT_EXTEND_LATENT_FRAMES = 15
# A2V is always two-stage CFG — distilled UI defaults (8 steps) under-step it.
A2V_DEFAULT_STAGE1_STEPS = 30
A2V_MIN_STAGE1_STEPS = 20
# Keyframe interpolation requires upstream two-stage ctor filenames.
KEYFRAME_DEV_TRANSFORMER = "transformer-dev.safetensors"
KEYFRAME_DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384.safetensors"
KEYFRAME_DISTILLED_LORA_ALT = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
KEYFRAME_DEFAULT_CFG = 3.0
VALID_CHAIN_METHODS = frozenset({CHAIN_METHOD_AUTOCONTINUE, CHAIN_METHOD_NATIVE_EXTEND})

PIPE_PROFILE_DISTILLED = "distilled"
PIPE_PROFILE_TWO_STAGE = "two_stage"
PIPE_PROFILE_HQ = "hq"
PIPE_PROFILE_ONE_STAGE = "one_stage"
VALID_PIPELINE_PROFILES = frozenset(
    {PIPE_PROFILE_DISTILLED, PIPE_PROFILE_TWO_STAGE, PIPE_PROFILE_HQ, PIPE_PROFILE_ONE_STAGE}
)


def ltx2_mlx_install_hint() -> str:
    return (
        "  uv pip install \\\n"
        f'    "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@{LTX2_MLX_GIT_TAG}'
        '#subdirectory=packages/ltx-core-mlx" \\\n'
        f'    "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@{LTX2_MLX_GIT_TAG}'
        '#subdirectory=packages/ltx-pipelines-mlx"'
    )

# Hugging Face repo id: ``org/name`` (used with huggingface_hub.snapshot_download,
# same file set as ``huggingface-cli download org/name``).
_HF_REPO_ID_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"
)
REPO_ROOT = Path(__file__).resolve().parent
VIDEOFENTANYL_MODELS_ENV = "VIDEOFENTANYL_MODELS"
VIDEOFENTANYL_LORA_DIR_ENV = "VIDEOFENTANYL_LORA_DIR"
MAX_REMOTE_INPUT_BYTES = 512 * 1024 * 1024  # 512 MiB safety ceiling for remote audio/video


@dataclass
class GenerationRequest:
    prompt: str
    image_data: dict | str | None = None
    audio_data: dict | str | None = None
    source_video_data: dict | str | None = None
    seed: int = -1
    num_frames: int | None = None
    height: int | None = None
    width: int | None = None
    negative_prompt: str = ""
    mode: str = "generate"  # generate|a2v|retake|extend|ic_lora|keyframe|lipdub|face_swap|id_lora
    num_steps: int | None = None
    retake_start: int | None = None
    retake_end: int | None = None
    extend_frames: int | None = None
    extend_direction: str = "after"
    lora_specs: list[tuple[str, float]] | None = None
    video_conditioning_specs: list[tuple[dict | str, float]] | None = None
    job_id: str | None = None
    a2v_visual_i2v_continue: bool = False
    # Optional ltx-2-mlx advanced controls (see https://github.com/dgrauet/ltx-2-mlx#features)
    end_image_data: dict | str | None = None
    enhance_prompt: bool = False
    pipeline_profile: str = PIPE_PROFILE_DISTILLED
    cfg_scale: float | None = None
    stg_scale: float | None = None
    stage2_steps: int | None = None
    no_regen_audio: bool = False
    reference_strength: float | None = None
    audio_start_seconds: float | None = None
    skip_stage_2: bool = False
    upsample_only: bool = False
    modality_scale: float | None = None


# Pipelines that bind ``load_audio`` at import time (``from … import load_audio``).
_LOAD_AUDIO_STALE_IMPORTERS = (
    "ltx_pipelines_mlx.a2vid_two_stage",
    "ltx_pipelines_mlx.retake",
    "ltx_pipelines_mlx.lipdub",
)

# Pipelines that bind video probe/load at import time.
_VIDEO_IO_STALE_IMPORTERS = (
    "ltx_pipelines_mlx.iclora_utils",
    "ltx_pipelines_mlx.ic_lora",
    "ltx_pipelines_mlx.lipdub",
)


def _module_dict_attr(mod: Any, attr: str) -> Any:
    """Return a module-level attribute without triggering lazy ``__getattr__`` loaders."""
    if mod is None:
        return _MISSING
    d = vars(mod)
    if not isinstance(d, dict) or attr not in d:
        return _MISSING
    return d[attr]


_MISSING = object()


def _rebind_module_attr(
    module_name: str,
    attr: str,
    value: Any,
    *,
    stale: Any,
) -> None:
    import sys

    mod = sys.modules.get(module_name)
    if mod is None:
        return
    current = _module_dict_attr(mod, attr)
    if current is _MISSING:
        return
    if current is stale or current is value:
        if current is not value:
            setattr(mod, attr, value)


def _patch_load_audio_pyav_only() -> None:
    """Replace ltx-core-mlx ffmpeg load_audio with PyAV (pip package ``av``)."""
    from ltx_media import load_audio_for_inference

    try:
        import ltx_core_mlx.utils.audio as audio_mod
    except ImportError:
        return

    stale = getattr(audio_mod, "_ltx_ws_original_load_audio", None)
    if stale is None:
        current = audio_mod.load_audio
        if current is not load_audio_for_inference:
            audio_mod._ltx_ws_original_load_audio = current
            stale = current

    audio_mod.load_audio = load_audio_for_inference

    if stale is not None:
        for module_name in _LOAD_AUDIO_STALE_IMPORTERS:
            _rebind_module_attr(
                module_name,
                "load_audio",
                load_audio_for_inference,
                stale=stale,
            )


def _patch_ltx_pipelines_compat(*, default_fps: float = 24.0) -> None:
    """Default ``frame_rate`` for ``combined_image_conditionings`` (a2v+i2v on older pipelines)."""
    try:
        from ltx_pipelines_mlx.utils import _orchestration as orch
    except ImportError:
        return
    if getattr(orch, "_ltx_ws_frame_rate_patched", False):
        return

    original = orch.combined_image_conditionings
    frame_rate_param = inspect.signature(original).parameters.get("frame_rate")
    if frame_rate_param is None or frame_rate_param.default is not inspect.Parameter.empty:
        return

    def combined_image_conditionings(
        images,
        *,
        enc_h: int,
        enc_w: int,
        spatial_dims: tuple[int, int, int],
        video_encoder,
        frame_rate: float = default_fps,
    ):
        return original(
            images,
            enc_h=enc_h,
            enc_w=enc_w,
            spatial_dims=spatial_dims,
            video_encoder=video_encoder,
            frame_rate=frame_rate,
        )

    combined_image_conditionings.__name__ = getattr(original, "__name__", "combined_image_conditionings")
    orch.combined_image_conditionings = combined_image_conditionings
    orch._ltx_ws_frame_rate_patched = True


def _patch_video_decode_pyav_only() -> None:
    """Replace upstream ffmpeg stdin pipe video encode with PyAV."""
    try:
        from ltx_core_mlx.model.video_vae import video_vae as vv_mod
    except ImportError:
        return
    if getattr(vv_mod, "_ltx_ws_pyav_decode_patched", False):
        return

    from ltx_media import stream_decoder_latent_to_mp4

    def decode_and_stream(
        self,
        latent,
        output_path: str,
        frame_rate: float = 24.0,
        audio_path: str | None = None,
    ) -> None:
        stream_decoder_latent_to_mp4(
            self,
            latent,
            output_path,
            frame_rate=frame_rate,
            audio_path=audio_path,
        )

    vv_mod.VideoDecoder.decode_and_stream = decode_and_stream
    vv_mod._ltx_ws_pyav_decode_patched = True
    try:
        from ltx_video_decoder_pruna import VideoDecoderPruna

        VideoDecoderPruna.decode_and_stream = decode_and_stream  # type: ignore[method-assign]
    except ImportError:
        pass


ENV_VAE_DECODER = "LTX_WS_VAE_DECODER"
ENV_PRUNA_VAED_REPO = "LTX_WS_PRUNA_VAED_REPO"
PRUNA_VAED_HF_REPO = "audiohacking/pruna-vaed-mlx"
_VAE_DECODER_VARIANT: str = "stock"


def set_vae_decoder_variant(variant: str) -> None:
    """Select ``stock`` (default) or ``pruna`` VAE decoder for pipeline loads."""
    global _VAE_DECODER_VARIANT
    v = (variant or "stock").strip().lower()
    if v not in ("stock", "pruna"):
        raise ValueError(f"vae decoder variant must be 'stock' or 'pruna', got {variant!r}")
    _VAE_DECODER_VARIANT = v


def get_vae_decoder_variant() -> str:
    return _VAE_DECODER_VARIANT


def pruna_vaed_repo_id() -> str:
    return (os.environ.get(ENV_PRUNA_VAED_REPO) or PRUNA_VAED_HF_REPO).strip() or PRUNA_VAED_HF_REPO


def _pruna_vae_paths_in_dir(root: Path) -> tuple[Path, Path | None] | None:
    """Return (weights, config_or_None) if weights exist under ``root`` (or one nested level)."""
    candidates = [
        root,
        root / "pruna-vaed-mlx",
    ]
    for base in candidates:
        w = base / "vae_decoder_pruna.safetensors"
        if not w.is_file():
            continue
        c = base / "vae_decoder_pruna_config.json"
        return w, c if c.is_file() else None
    return None


def preview_pruna_vae_source() -> str:
    """Filesystem path where PrunaVAED MLX weights are or will be stored."""
    for root in (
        REPO_ROOT / "models" / "pruna-vaed-mlx",
        hf_local_weights_directory(pruna_vaed_repo_id(), None),
    ):
        found = _pruna_vae_paths_in_dir(root)
        if found is not None:
            return str(found[0].parent)
    return str(hf_local_weights_directory(pruna_vaed_repo_id(), None))


def ensure_pruna_vae_decoder_files(model_dir: Path | None = None) -> tuple[Path, Path | None]:
    """Locate or download PrunaVAED MLX weights the same way as main MLX models.

    Search order:
    1. ``model_dir`` / nested ``pruna-vaed-mlx`` (optional co-located weights)
    2. ``REPO_ROOT/models/pruna-vaed-mlx`` (local mlx-forge convert output)
    3. Cached Hub snapshot under ``models/<org>__<name>/`` (same layout as
       :func:`resolve_mlx_weights_directory`)
    4. ``huggingface_hub.snapshot_download`` of :func:`pruna_vaed_repo_id`
    """
    roots: list[Path] = []
    if model_dir is not None:
        roots.append(Path(model_dir))
    roots.append(REPO_ROOT / "models" / "pruna-vaed-mlx")

    for root in roots:
        found = _pruna_vae_paths_in_dir(root)
        if found is not None:
            w, c = found
            log.info("Using existing local PrunaVAED MLX weights at %s", w)
            return w, c

    repo = pruna_vaed_repo_id()
    dest = hf_local_weights_directory(repo, None)
    dest.mkdir(parents=True, exist_ok=True)
    found = _pruna_vae_paths_in_dir(dest)
    if found is not None:
        w, c = found
        log.info("Using existing local MLX snapshot for %r at %s", repo, dest)
        return w, c

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to download PrunaVAED MLX weights from Hugging Face. "
            "Install with:  pip install huggingface_hub\n"
            f"Or place vae_decoder_pruna.safetensors under {dest}."
        ) from e

    log.info(
        "Ensuring Hugging Face weights %r under %s "
        "(huggingface_hub.snapshot_download; same payload as `huggingface-cli download`) …",
        repo,
        dest,
    )
    _snapshot_download_weights(snapshot_download, repo, dest)
    found = _pruna_vae_paths_in_dir(dest)
    if found is None:
        raise FileNotFoundError(
            f"Downloaded {repo} to {dest} but vae_decoder_pruna.safetensors is missing"
        )
    w, c = found
    log.info("PrunaVAED MLX decoder ready at %s", w)
    return w, c


def _patch_pruna_vae_decoder_loader() -> None:
    """Swap TextToVideoPipeline._load_decoders to opt into VideoDecoderPruna."""
    try:
        from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
    except ImportError:
        return
    if getattr(TextToVideoPipeline, "_ltx_ws_pruna_vae_patched", False):
        return

    from ltx_core_mlx.model.audio_vae import AudioVAEDecoder, VocoderWithBWE
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys

    _orig = TextToVideoPipeline._load_decoders

    def _load_decoders(self) -> None:
        variant = get_vae_decoder_variant()
        if variant != "pruna":
            log.info(
                "VAE decoder: stock  (pipeline default vae_decoder.safetensors)"
            )
            return _orig(self)

        model_dir = self.model_dir
        if self.vae_decoder is None:
            from ltx_video_decoder_pruna import VideoDecoderPruna

            weights_path, cfg_path = ensure_pruna_vae_decoder_files(model_dir)
            log.info(
                "VAE decoder: pruna  (loading VideoDecoderPruna from %s)",
                weights_path,
            )
            if cfg_path is not None:
                self.vae_decoder = VideoDecoderPruna.from_config(cfg_path)
            else:
                self.vae_decoder = VideoDecoderPruna()
            vae_weights = load_split_safetensors(weights_path, prefix="vae_decoder.")
            self.vae_decoder.load_weights(list(vae_weights.items()))
            aggressive_cleanup()
            log.info("VAE decoder: pruna ready  (%s)", weights_path)
        else:
            log.info(
                "VAE decoder: pruna  (reusing already-loaded VideoDecoderPruna)"
            )

        if self.audio_decoder is None:
            self.audio_decoder = AudioVAEDecoder()
            audio_weights = load_split_safetensors(
                model_dir / "audio_vae.safetensors", prefix="audio_vae.decoder."
            )
            all_audio = load_split_safetensors(
                model_dir / "audio_vae.safetensors", prefix="audio_vae."
            )
            for k, v in all_audio.items():
                if k.startswith("per_channel_statistics."):
                    audio_weights[k] = v
            audio_weights = remap_audio_vae_keys(audio_weights)
            self.audio_decoder.load_weights(list(audio_weights.items()))
            aggressive_cleanup()

        if self.vocoder is None:
            self.vocoder = VocoderWithBWE()
            vocoder_weights = load_split_safetensors(
                model_dir / "vocoder.safetensors", prefix="vocoder."
            )
            self.vocoder.load_weights(list(vocoder_weights.items()))
            self.vocoder.upcast_weights_to_fp32()
            aggressive_cleanup()

    TextToVideoPipeline._load_decoders = _load_decoders  # type: ignore[method-assign]
    TextToVideoPipeline._ltx_ws_pruna_vae_patched = True


def _patch_media_io_pyav_only() -> None:
    """Replace ltx-pipelines ffmpeg I2V image preprocess with PyAV libx264."""
    try:
        from ltx_pipelines_mlx.utils import media_io as media_mod
    except ImportError:
        return

    from ltx_media import decode_single_frame as pyav_decode_single_frame
    from ltx_media import encode_single_frame as pyav_encode_single_frame

    first = not getattr(media_mod, "_ltx_ws_pyav_media_patched", False)
    if first:
        media_mod._ltx_ws_orig_encode = media_mod.encode_single_frame
        media_mod._ltx_ws_orig_decode = media_mod.decode_single_frame

    media_mod.encode_single_frame = pyav_encode_single_frame
    media_mod.decode_single_frame = pyav_decode_single_frame

    media_mod._ltx_ws_pyav_media_patched = True
    if media_mod.encode_single_frame.__module__ != "ltx_media":
        raise RuntimeError(
            "Failed to replace ltx_pipelines_mlx image encode with PyAV "
            "(encode_single_frame still bound to ffmpeg)"
        )
    if first:
        log.debug("PyAV media_io patch applied (I2V image preprocess)")


def _patch_video_io_pyav_only() -> None:
    """Replace upstream ffprobe/ffmpeg video probe and decode with PyAV."""
    try:
        import ltx_core_mlx.utils.ffmpeg as ffmpeg_mod
        import ltx_core_mlx.utils.video as video_mod
    except ImportError:
        return

    from ltx_media import load_video_frames_normalized as pyav_load_video_frames
    from ltx_media import probe_video_info as pyav_probe_video_info

    first = not getattr(ffmpeg_mod, "_ltx_ws_pyav_video_io_patched", False)

    stale_probe = getattr(ffmpeg_mod, "_ltx_ws_original_probe_video_info", None)
    if stale_probe is None:
        current_probe = ffmpeg_mod.probe_video_info
        if current_probe is not pyav_probe_video_info:
            ffmpeg_mod._ltx_ws_original_probe_video_info = current_probe
            stale_probe = current_probe
    ffmpeg_mod.probe_video_info = pyav_probe_video_info

    stale_load = getattr(video_mod, "_ltx_ws_original_load_video_frames", None)
    if stale_load is None:
        current_load = video_mod.load_video_frames_normalized
        if current_load is not pyav_load_video_frames:
            video_mod._ltx_ws_original_load_video_frames = current_load
            stale_load = current_load
    video_mod.load_video_frames_normalized = pyav_load_video_frames

    if stale_probe is not None:
        for module_name in _VIDEO_IO_STALE_IMPORTERS:
            _rebind_module_attr(
                module_name,
                "probe_video_info",
                pyav_probe_video_info,
                stale=stale_probe,
            )
    if stale_load is not None:
        for module_name in _VIDEO_IO_STALE_IMPORTERS:
            _rebind_module_attr(
                module_name,
                "load_video_frames_normalized",
                pyav_load_video_frames,
                stale=stale_load,
            )

    ffmpeg_mod._ltx_ws_pyav_video_io_patched = True
    if first:
        log.debug("PyAV video_io patch applied (IC-LoRA reference video probe/decode)")


def _patch_iclora_stage2_x0_model() -> None:
    """Recreate X0Model after Stage 2 clean transformer reload.

    Upstream ``ICLoraPipeline.generate`` builds ``x0_model = X0Model(self.dit)``
    before Stage 1, then reloads a clean DiT for Stage 2 but keeps denoising
    with the old (LoRA-fused) ``x0_model``. Rebuild the wrapper so Stage 2
    matches the documented clean-distilled refine.
    """
    try:
        from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
        from ltx_core_mlx.model.transformer.model import X0Model
    except ImportError:
        return
    if getattr(ICLoraPipeline, "_ltx_ws_stage2_x0_patched", False):
        return

    original = ICLoraPipeline._reload_clean_transformer

    def _reload_and_flag(self: Any) -> None:
        original(self)
        # Stash flag so generate can rebuild x0_model; generate itself is long
        # so we patch denoise by wrapping generate instead when possible.
        self._ltx_ws_need_stage2_x0_refresh = True

    ICLoraPipeline._reload_clean_transformer = _reload_and_flag  # type: ignore[method-assign]

    original_generate = ICLoraPipeline.generate

    def _generate_with_stage2_x0(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Wrap denoise_loop usage by monkeypatching X0Model construction mid-generate
        # is fragile; instead post-process: if generate returns after stage2 with
        # stale x0, we patch at the call site inside generate via local rewrite.
        # Practical approach: wrap denoise_loop for this call.
        import ltx_pipelines_mlx.ic_lora as ic_mod

        self._ltx_ws_need_stage2_x0_refresh = False
        real_denoise = ic_mod.denoise_loop
        x0_holder: dict[str, Any] = {"model": None}

        def denoise_loop(*, model, **kw):  # type: ignore[no-untyped-def]
            if getattr(self, "_ltx_ws_need_stage2_x0_refresh", False):
                # Stage 2: rebuild X0 around the clean reloaded dit.
                model = X0Model(self.dit)
                self._ltx_ws_need_stage2_x0_refresh = False
                log.info("IC-LoRA Stage 2: rebuilt X0Model on clean transformer")
            x0_holder["model"] = model
            return real_denoise(model=model, **kw)

        ic_mod.denoise_loop = denoise_loop  # type: ignore[assignment]
        try:
            return original_generate(self, *args, **kwargs)
        finally:
            ic_mod.denoise_loop = real_denoise  # type: ignore[assignment]

    ICLoraPipeline.generate = _generate_with_stage2_x0  # type: ignore[method-assign]
    ICLoraPipeline._ltx_ws_stage2_x0_patched = True
    log.info("Patched ICLoraPipeline Stage 2 to use clean X0Model after LoRA reload")


def _pad_audio_latent_to_token_count(audio_latent: Any, audio_T: int) -> Any:
    """Pad/truncate audio VAE latent time dim to ``audio_T`` (silence = zeros)."""
    import mlx.core as mx

    t = int(audio_latent.shape[2])
    target = int(audio_T)
    if t == target:
        return audio_latent
    if t > target:
        return audio_latent[:, :, :target, :]
    pad = mx.zeros(
        (audio_latent.shape[0], audio_latent.shape[1], target - t, audio_latent.shape[3]),
        dtype=audio_latent.dtype,
    )
    return mx.concatenate([audio_latent, pad], axis=2)


def _patch_a2v_short_audio_pad() -> None:
    """Pad short A2V audio latents to the video token length (RoPE shape match).

    Upstream ``A2VidPipelineTwoStage`` computes ``audio_T`` from ``num_frames`` but
    only truncates encoded audio — it does not pad when the clip is shorter than
    the requested video. That yields ``audio_tokens`` shorter than
    ``compute_audio_positions(audio_T)`` and crashes in RoPE.
    """
    try:
        from ltx_pipelines_mlx.a2vid_two_stage import A2VidPipelineTwoStage
        import ltx_pipelines_mlx.a2vid_two_stage as a2v_mod
    except ImportError:
        return
    if getattr(A2VidPipelineTwoStage, "_ltx_ws_short_audio_padded", False):
        return

    from ltx_core_mlx.utils.positions import compute_audio_token_count

    orig = A2VidPipelineTwoStage.generate_and_save

    @functools.wraps(orig)
    def generate_and_save(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = inspect.signature(orig).bind(self, *args, **kwargs)
        bound.apply_defaults()
        num_frames = int(bound.arguments["num_frames"])
        frame_rate = float(bound.arguments["frame_rate"])
        audio_T = compute_audio_token_count(num_frames, frame_rate)
        real_encode = a2v_mod.encode_audio

        def encode_audio(waveform: Any, sample_rate: Any, encoder: Any, processor: Any = None) -> Any:
            lat = real_encode(waveform, sample_rate, encoder, processor)
            t = int(lat.shape[2])
            if t < audio_T:
                log.info(
                    "a2v: audio shorter than video — padded latent tokens %d → %d (silence)",
                    t,
                    audio_T,
                )
            return _pad_audio_latent_to_token_count(lat, audio_T)

        a2v_mod.encode_audio = encode_audio  # type: ignore[assignment]
        try:
            return orig(self, *args, **kwargs)
        finally:
            a2v_mod.encode_audio = real_encode  # type: ignore[assignment]

    A2VidPipelineTwoStage.generate_and_save = generate_and_save  # type: ignore[method-assign]
    A2VidPipelineTwoStage._ltx_ws_short_audio_padded = True


def _mlx_strided_scatter_add_broken() -> bool | None:
    """Return True if Metal ``.at[strided].add()`` mis-indexes (mlx 0.31.2 bug).

    ``None`` if mlx is unavailable. See ltx-2-mlx#34 / ml-explore/mlx#3477.
    """
    try:
        import mlx.core as mx
        import numpy as np
    except ImportError:
        return None
    try:
        b, t, c = 2, 64, 8
        x = (mx.arange(b * t * c, dtype=mx.float32).reshape(b, t, c) % 7 - 3) / 3.0
        y = mx.zeros((b, t * 2, c)).at[:, ::2, :].add(x)
        mx.eval(y)
        expected = np.zeros((b, t * 2, c), dtype=np.float32)
        expected[:, ::2, :] = np.array(x)
        return bool((np.array(y) != expected).any())
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("mlx scatter canary failed: %s", exc)
        return None


def _source_uses_strided_at_add(fn: Any) -> bool:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return ".at[" in src and ".add(" in src


def _patch_mlx_audio_upsample_scatter() -> None:
    """Work around mlx 0.31.2 Metal vocoder collapse (hiss / near-silent audio).

    On broken mlx, ``UpSample1d`` / ``HannSincResampler`` zero-insert via
    ``.at[::k].add()`` corrupts the waveform (~−50 dB hiss). Destination rows
    are freshly zeroed, so strided assignment is equivalent and correct on all
    Metal GPUs (M3/M4/M5). Idempotent when ltx-core-mlx ≥ 0.14.19 already fixed.
    """
    try:
        from ltx_core_mlx.model.audio_vae.vocoder import UpSample1d
        from ltx_core_mlx.model.audio_vae.bwe import HannSincResampler
        import mlx.core as mx
    except ImportError:
        return
    if getattr(UpSample1d, "_ltx_ws_scatter_patched", False):
        return

    scatter_broken = _mlx_strided_scatter_add_broken()
    need_up = _source_uses_strided_at_add(UpSample1d.__call__)
    need_bwe = _source_uses_strided_at_add(HannSincResampler.__call__)

    if scatter_broken and (need_up or need_bwe):
        log.warning(
            "MLX Metal strided .at[].add() is broken on this GPU/driver "
            "(known mlx 0.31.2 bug → vocoder hiss). "
            "ltx-ws will force the slice-assignment workaround."
        )
    elif scatter_broken:
        log.info(
            "MLX Metal strided .at[].add() is broken (mlx 0.31.2), "
            "but audio vocoder already uses the safe slice-assign workaround"
        )
    elif scatter_broken is False:
        log.info("MLX strided-scatter canary OK")

    if not need_up and not need_bwe:
        log.info(
            "Audio vocoder upsample already uses safe slice assign "
            "(ltx-core-mlx ≥ 0.14.19 or previously patched)"
        )
        UpSample1d._ltx_ws_scatter_patched = True
        return

    if need_up:
        def _upsample1d_call(self: Any, x: Any) -> Any:
            b, t, c = x.shape
            x_up = mx.zeros((b, t * 2, c))
            # Safe on mlx 0.31.2 Metal (dest is zeros ⇒ assign ≡ add).
            x_up[:, ::2, :] = x
            x_up = x_up.transpose(0, 2, 1).reshape(b * c, t * 2, 1)
            k = self.filter.shape[1]
            pad = k // 2
            left_edge = mx.repeat(x_up[:, :1, :], pad, axis=1)
            right_edge = mx.repeat(x_up[:, -1:, :], pad - 1, axis=1)
            x_up = mx.concatenate([left_edge, x_up, right_edge], axis=1)
            x_up = mx.conv1d(x_up, self.filter)
            t_out = x_up.shape[1]
            return x_up.reshape(b, c, t_out).transpose(0, 2, 1) * 2.0

        UpSample1d.__call__ = _upsample1d_call  # type: ignore[method-assign]
        log.info("Patched UpSample1d zero-insert to slice assign (mlx#3477 / ltx-2-mlx#34)")

    if need_bwe:
        def _hann_sinc_call(self: Any, x: Any) -> Any:
            b, t = x.shape
            ratio = self.upsample_factor
            first = mx.repeat(x[:, :1], self._pad, axis=1)
            last = mx.repeat(x[:, -1:], self._pad, axis=1)
            x_padded = mx.concatenate([first, x, last], axis=1)
            t_padded = x_padded.shape[1]
            zi_len = (t_padded - 1) * ratio + 1
            upsampled = mx.zeros((b, zi_len))
            upsampled[:, ::ratio] = x_padded
            upsampled = upsampled[:, :, None]
            k = self.kernel.shape[0]
            upsampled = mx.pad(upsampled, [(0, 0), (k - 1, k - 1), (0, 0)])
            filt = self.kernel[None, :, :]
            result = mx.conv1d(upsampled, filt, padding=0)
            result = result.squeeze(-1)
            result = result * ratio
            result = result[:, self._pad_left : -self._pad_right]
            return result[:, : t * ratio]

        HannSincResampler.__call__ = _hann_sinc_call  # type: ignore[method-assign]
        log.info(
            "Patched HannSincResampler zero-insert to slice assign "
            "(mlx#3477 / ltx-2-mlx#34)"
        )

    UpSample1d._ltx_ws_scatter_patched = True


def _apply_ltx_mlx_patches(*, default_fps: float = 24.0) -> None:
    """Apply all ltx-ws runtime patches (PyAV-only media, pipeline compat)."""
    _patch_media_io_pyav_only()
    _patch_load_audio_pyav_only()
    _patch_video_io_pyav_only()
    _patch_ltx_pipelines_compat(default_fps=default_fps)
    _patch_video_decode_pyav_only()
    _patch_pruna_vae_decoder_loader()
    _patch_iclora_stage2_x0_model()
    _patch_a2v_short_audio_pad()
    _patch_mlx_audio_upsample_scatter()


def looks_like_hf_repo_id(model: str) -> bool:
    """True if ``model`` looks like ``author/repo`` and is not an existing directory path."""
    s = (model or "").strip()
    if not s or _HF_REPO_ID_RE.match(s) is None:
        return False
    p = Path(s).expanduser()
    if p.is_dir():
        return False
    return True


def _snapshot_download_weights(snapshot_download: Any, repo_id: str, dest: Path) -> str:
    """Call ``snapshot_download`` with kwargs compatible across huggingface_hub versions."""
    import inspect

    kw: dict[str, Any] = {"repo_id": repo_id, "local_dir": str(dest)}
    sig = inspect.signature(snapshot_download)
    if "resume_download" in sig.parameters:
        kw["resume_download"] = True
    if "local_dir_use_symlinks" in sig.parameters:
        kw["local_dir_use_symlinks"] = False
    out = snapshot_download(**kw)
    return str(Path(out).resolve())


def _model_snapshot_present(dest: Path) -> bool:
    """
    Heuristic to detect an already materialized HF snapshot in ``dest``.
    """
    if not dest.is_dir():
        return False
    try:
        has_config = (dest / "config.json").is_file() or (dest / "embedded_config.json").is_file()
        has_weights = any(dest.glob("*.safetensors"))
    except OSError:
        return False
    return bool(has_config and has_weights)


def hf_local_weights_directory(repo_id: str, explicit_model_dir: str | None) -> Path:
    """
    Directory where we store a full ``snapshot_download`` for ``repo_id``.

    If ``explicit_model_dir`` is set, that path is used. Otherwise:
    ``$VIDEOFENTANYL_MODELS/<org>__<name>/`` when the env var is set, else
    ``<repo_root>/models/<org>__<name>/``.
    """
    rid = repo_id.strip()
    if explicit_model_dir:
        return Path(explicit_model_dir).expanduser().resolve()
    env = os.environ.get(VIDEOFENTANYL_MODELS_ENV, "").strip()
    root = Path(env).expanduser().resolve() if env else (REPO_ROOT / "models")
    safe = rid.replace("/", "__")
    return (root / safe).resolve()


def _looks_like_models_dir_leaf(name: str) -> bool:
    """True if ``name`` is a single path segment (safe to join under ``models/``)."""
    s = (name or "").strip()
    if not s or s in (".", "..") or s.startswith(".."):
        return False
    if "/" in s or "\\" in s:
        return False
    return Path(s).name == s


def _path_candidates_for_user_string(user_path: str) -> list[Path]:
    """For a filesystem path string: absolutes resolve once; relatives try cwd then repo root.

    This fixes ``python /path/to/server.py`` started from ``$HOME`` where
    ``./models/foo`` must resolve next to the checkout, not under ``$HOME``.
    """
    raw = (user_path or "").strip()
    if not raw:
        return []
    p = Path(raw).expanduser()
    if p.is_absolute():
        return [p.resolve()]
    return [(Path.cwd() / p).resolve(), (REPO_ROOT / p).resolve()]


def _first_existing_dir(user_path: str) -> Path | None:
    for c in _path_candidates_for_user_string(user_path):
        if c.is_dir():
            return c
    return None


def _resolve_non_hf_disk_path(model: str, explicit_model_dir: str | None) -> str | None:
    """
    Resolve to an existing weights directory without calling the Hub.

    Tries: ``--model`` as a directory path (cwd, then repo root for relatives),
    then ``--model-dir`` the same way, then ``models/<model>/`` under cwd and
    under repo root for a shorthand leaf (e.g. ``ltx-2.3-mlx``).
    """
    raw = (model or "").strip()
    if not raw:
        return None

    hit = _first_existing_dir(raw)
    if hit is not None:
        return str(hit)

    md = (explicit_model_dir or "").strip()
    if md:
        hit = _first_existing_dir(md)
        if hit is not None:
            return str(hit)

    if _looks_like_models_dir_leaf(raw):
        leaf = Path(raw).name
        for base in (Path.cwd(), REPO_ROOT):
            candidate = (base / "models" / leaf).resolve()
            try:
                candidate.relative_to(base.resolve())
            except ValueError:
                continue
            if candidate.is_dir():
                return str(candidate)

    return None


def preview_mlx_weights_source(model: str, explicit_model_dir: str | None) -> str:
    """Where weights are expected on disk (for UI); may not exist yet for fresh HF pulls."""
    raw = (model or "").strip()
    got = _resolve_non_hf_disk_path(raw, explicit_model_dir)
    if got is not None:
        return got
    if looks_like_hf_repo_id(raw):
        return str(hf_local_weights_directory(raw, explicit_model_dir))
    return raw


def resolve_mlx_weights_directory(model: str, explicit_model_dir: str | None) -> str:
    """Resolve ``model`` and optional ``explicit_model_dir`` to an on-disk MLX weights tree."""
    raw = (model or "").strip()
    disk = _resolve_non_hf_disk_path(raw, explicit_model_dir)
    if disk is not None:
        return disk

    if looks_like_hf_repo_id(raw):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required to download MLX weights from Hugging Face. "
                "Install with:  pip install huggingface_hub\n"
                "Or use a local directory for --model."
            ) from e
        dest = hf_local_weights_directory(raw, explicit_model_dir)
        dest.mkdir(parents=True, exist_ok=True)
        if _model_snapshot_present(dest):
            log.info("Using existing local MLX snapshot for %r at %s", raw, dest)
            return str(dest)
        log.info(
            "Ensuring Hugging Face weights %r under %s "
            "(huggingface_hub.snapshot_download; same payload as `huggingface-cli download`) …",
            raw,
            dest,
        )
        _snapshot_download_weights(snapshot_download, raw, dest)
        return str(dest)

    return raw


def _spill_slug(prompt: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^\w\s-]+", "", prompt.lower().strip())[:maxlen]
    s = re.sub(r"[\s_]+", "_", s).strip("_")
    return s or "clip"


def _largest_mp4_under(root: Path) -> Path | None:
    best: Path | None = None
    best_mtime = -1.0
    try:
        for p in root.rglob("*.mp4"):
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size <= 0:
                continue
            if st.st_mtime >= best_mtime:
                best_mtime = st.st_mtime
                best = p
    except OSError:
        return None
    return best


def _align_ltx2_spatial(n: int, align: int = LTX2_SPATIAL_ALIGN) -> int:
    if n < align:
        return align
    lower = (n // align) * align
    upper = lower + align
    return lower if (n - lower) <= (upper - n) else upper


def _nearest_valid_frames(n: int) -> int:
    if n < 9:
        return 9
    remainder = (n - 1) % 8
    if remainder == 0:
        return n
    lower = n - remainder
    upper = lower + 8
    return lower if (n - lower) <= (upper - n) else upper


def _decode_initial_image_dict(image_data: dict) -> str:
    """Data URL / path / base64 → path or URL (same contract as ``server._decode_initial_image``)."""
    data_url: str = (image_data.get("data_url") or "").strip()
    if data_url.startswith(("http://", "https://")):
        return data_url
    if data_url.startswith("file://"):
        from urllib.parse import unquote
        from urllib.request import url2pathname

        path = url2pathname(unquote(data_url[7:]))
        if os.path.isfile(path):
            return path
    if data_url and os.path.isfile(data_url):
        return data_url

    if data_url.startswith("data:"):
        header, encoded = data_url.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
    else:
        mime = image_data.get("mime_type", "image/jpeg")
        encoded = data_url

    ext = mimetypes.guess_extension(mime) or ".jpg"
    if ext == ".jpe":
        ext = ".jpg"

    fd, path = mk_scratch_file(prefix="fvserver_img_", suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(base64.b64decode(encoded))
    return path


def _download_remote_to_temp(
    url: str,
    prefix: str,
    suffix_hint: str = "",
    max_bytes: int | None = MAX_REMOTE_INPUT_BYTES,
) -> str:
    req_url = (url or "").strip()
    if not req_url.startswith(("http://", "https://")):
        raise ValueError(f"Unsupported remote input URL: {url!r}")
    with urlopen(req_url, timeout=180) as resp:
        if max_bytes is None:
            payload = resp.read()
        else:
            payload = resp.read(max_bytes + 1)
    if max_bytes is not None and len(payload) > max_bytes:
        raise RuntimeError(
            f"Remote media exceeds {max_bytes // (1024 * 1024)} MiB limit"
        )
    fd, path = mk_scratch_file(prefix=prefix, suffix=suffix_hint)
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
    return path


def _local_lora_cache_dir() -> Path:
    env = (os.environ.get(VIDEOFENTANYL_LORA_DIR_ENV) or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "loras").resolve()


def _normalize_lora_spec(spec: str) -> str:
    """Normalize LoRA specs: HF URL variants, and repair Path()-mangled URLs.

    On macOS/Linux, ``Path("https://…").resolve()`` collapses ``://`` to ``:/`` and
    may prefix the cwd (e.g. ``/repo/https:/huggingface.co/…``). Recover those.
    """
    raw = (spec or "").strip()
    if not raw:
        return raw

    # Recover cwd-prefixed / Path-collapsed URLs before any Path() usage.
    for marker in ("https://", "http://", "https:/", "http:/"):
        idx = raw.find(marker)
        if idx > 0 and ("huggingface.co" in raw or "hf.co" in raw):
            raw = raw[idx:]
            break
    if raw.startswith("https:/") and not raw.startswith("https://"):
        raw = "https://" + raw[len("https:/") :]
    elif raw.startswith("http:/") and not raw.startswith("http://"):
        raw = "http://" + raw[len("http:/") :]

    if not raw.startswith(("http://", "https://")):
        return raw
    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    if host in ("hf.co", "www.hf.co"):
        raw = f"https://huggingface.co{parsed.path}"
        if parsed.query:
            raw += f"?{parsed.query}"
    if "huggingface.co" in raw and "/blob/" in raw:
        raw = raw.replace("/blob/", "/resolve/", 1)
    return raw


def _is_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _pick_safetensors_file(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*.safetensors"))
    if not candidates:
        return None
    # Prefer explicit loras/ subdir when present.
    for c in candidates:
        if "loras" in {p.lower() for p in c.parts}:
            return c
    # Prefer main IC-LoRA weight over auxiliary embeddings when both exist.
    non_emb = [c for c in candidates if "scene-emb" not in c.name.lower()]
    if non_emb:
        return non_emb[0]
    return candidates[0]


class _HfLoraResolve(NamedTuple):
    cache_dir_name: str
    filename: str
    repo_id: str | None
    revision: str | None
    url: str


def _parse_hf_lora_resolve_url(url: str) -> _HfLoraResolve | None:
    """Parse Hugging Face model or bucket resolve URLs into cache/download targets."""
    parsed = urlparse(url)
    if not parsed.netloc.endswith("huggingface.co") or "/resolve/" not in parsed.path:
        return None
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) >= 5 and parts[0] == "buckets" and parts[3] == "resolve":
        return _HfLoraResolve(
            cache_dir_name=f"{parts[1]}__{parts[2]}",
            filename="/".join(parts[4:]),
            repo_id=None,
            revision=None,
            url=url,
        )
    if len(parts) >= 5 and parts[2] == "resolve":
        repo_id = f"{parts[0]}/{parts[1]}"
        return _HfLoraResolve(
            cache_dir_name=repo_id.replace("/", "__"),
            filename="/".join(parts[4:]),
            repo_id=repo_id,
            revision=parts[3],
            url=url,
        )
    return None


def _hf_lora_cache_file(resolved: _HfLoraResolve) -> Path:
    return (_local_lora_cache_dir() / resolved.cache_dir_name / resolved.filename).resolve()


def _is_usable_lora_file(path: Path | None) -> bool:
    """True when path is a non-empty .safetensors (or any) weight file on disk."""
    if path is None:
        return False
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _materialize_lora_cache(src: Path, dest: Path) -> Path:
    """Place src at the canonical cache path (hardlink, then copy)."""
    src = src.resolve()
    dest = dest.resolve()
    if src == dest and _is_usable_lora_file(dest):
        return dest
    if _is_usable_lora_file(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        import shutil

        shutil.copy2(src, dest)
    return dest


def _find_lora_by_basename(filename: str) -> Path | None:
    """Search the persistent LoRA cache for an existing file by basename."""
    name = Path(filename).name
    if not name:
        return None
    root = _local_lora_cache_dir()
    if not root.is_dir():
        return None
    # Prefer exact relative layout when present (e.g. loras/foo.safetensors).
    exact = (root / filename).resolve()
    if _is_usable_lora_file(exact):
        return exact
    matches: list[Path] = []
    for match in root.rglob(name):
        if not _is_usable_lora_file(match):
            continue
        # Skip HF download metadata / incomplete sidecars mistaken as weights.
        if match.suffix.lower() != ".safetensors" and not name.endswith(match.suffix):
            continue
        if ".cache" in match.parts:
            continue
        matches.append(match.resolve())
    if not matches:
        return None
    # Prefer paths under a repo-named cache dir that include the relative filename.
    for m in matches:
        if filename in str(m).replace("\\", "/"):
            return m
    return matches[0]


def _find_lora_in_hf_hub_cache(resolved: _HfLoraResolve) -> Path | None:
    """Return a hub-cache hit without network when repo_id is known."""
    if not resolved.repo_id:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        cached = try_to_load_from_cache(
            repo_id=resolved.repo_id,
            filename=resolved.filename,
            revision=resolved.revision,
        )
    except Exception:
        return None
    if cached is None or cached is False:
        return None
    path = Path(str(cached))
    return path if _is_usable_lora_file(path) else None


def _resolve_existing_hf_lora(resolved: _HfLoraResolve) -> Path | None:
    """Locate an already-downloaded LoRA and promote it to the canonical cache path."""
    dest = _hf_lora_cache_file(resolved)
    if _is_usable_lora_file(dest):
        return dest

    # Same repo cache dir, possibly nested (hf_hub_download local_dir layouts).
    cache_dir = (_local_lora_cache_dir() / resolved.cache_dir_name).resolve()
    if cache_dir.is_dir():
        for match in cache_dir.rglob(Path(resolved.filename).name):
            if ".cache" in match.parts:
                continue
            if _is_usable_lora_file(match):
                return _materialize_lora_cache(match, dest)

    # Any prior download under ./loras (or VIDEOFENTANYL_LORA_DIR) by basename.
    found = _find_lora_by_basename(resolved.filename)
    if found is not None:
        log.info("Reusing cached LoRA %s → %s", found, dest)
        return _materialize_lora_cache(found, dest)

    # Hugging Face hub cache from a previous hf_hub_download / CLI pull.
    hub_hit = _find_lora_in_hf_hub_cache(resolved)
    if hub_hit is not None:
        log.info("Reusing Hugging Face hub LoRA cache %s → %s", hub_hit, dest)
        return _materialize_lora_cache(hub_hit, dest)

    # Offline hub probe: may resolve snapshot paths try_to_load_from_cache misses.
    if resolved.repo_id and resolved.revision:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return None
        try:
            local = hf_hub_download(
                repo_id=resolved.repo_id,
                filename=resolved.filename,
                revision=resolved.revision,
                local_files_only=True,
            )
        except Exception:
            return None
        path = Path(local)
        if _is_usable_lora_file(path):
            log.info("Reusing local-only hub LoRA %s → %s", path, dest)
            return _materialize_lora_cache(path, dest)
    return None


def _download_hf_lora_resolve(resolved: _HfLoraResolve) -> Path:
    """Download a Hugging Face resolve URL into the persistent LoRA cache."""
    existing = _resolve_existing_hf_lora(resolved)
    if existing is not None:
        return existing
    dest = _hf_lora_cache_file(resolved)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if resolved.repo_id and resolved.revision:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required to download LoRA from Hugging Face"
            ) from e
        log.info(
            "Downloading LoRA %s (%s @ %s) …",
            resolved.repo_id,
            resolved.filename,
            resolved.revision,
        )
        local = hf_hub_download(
            repo_id=resolved.repo_id,
            filename=resolved.filename,
            revision=resolved.revision,
            local_dir=str(_local_lora_cache_dir() / resolved.cache_dir_name),
        )
        return Path(local).resolve()
    log.info("Downloading public LoRA from %s …", resolved.url)
    with urlopen(resolved.url, timeout=300) as resp:
        with dest.open("wb") as handle:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    return dest


def _lora_cached_path(spec: str) -> Path | None:
    """Return local path when spec is already on disk; None if download may be needed."""
    raw = _normalize_lora_spec(spec)
    if not raw:
        return None

    # URLs must not go through pathlib — Path("https://…") collapses :// → :/ .
    if _is_http_url(raw):
        resolved = _parse_hf_lora_resolve_url(raw)
        if resolved is not None:
            return _resolve_existing_hf_lora(resolved)
        # Non-HF URL: still reuse a prior basename hit under the LoRA cache.
        name = Path(urlparse(raw).path).name
        if name.endswith(".safetensors"):
            return _find_lora_by_basename(name)
        return None

    p = Path(raw).expanduser()
    if _is_usable_lora_file(p):
        return p.resolve()

    if looks_like_hf_repo_id(raw):
        dest = (_local_lora_cache_dir() / raw.replace("/", "__")).resolve()
        if dest.is_dir():
            picked = _pick_safetensors_file(dest)
            if picked is not None and _is_usable_lora_file(picked):
                return picked.resolve()
        # Basename fallback when a resolve-URL download already placed the weights.
        found = _find_lora_by_basename(raw.rsplit("/", 1)[-1])
        if found is not None:
            return found

    return None


def format_lora_download_error(exc: BaseException, spec: str = "") -> str:
    """User-facing LoRA download failure (gated Hugging Face repos, etc.)."""
    msg = str(exc)
    lower = msg.lower()
    gated = (
        "403" in msg
        or "gated" in lower
        or "authorized list" in lower
        or "cannot access gated repo" in lower
    )
    if gated or "LTX-2.3-22b-IC-LoRA-LipDub" in (spec or msg):
        return (
            "LipDub LoRA is gated on Hugging Face. Accept access at "
            "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub, "
            "set HF_TOKEN (or run huggingface-cli login), then add a custom LoRA "
            "with the official resolve URL or set LTX_WS_LIPDUB_LORA to a local "
            ".safetensors path."
        )
    return f"LoRA download failed: {exc}"


def _resolve_lora_path(spec: str) -> tuple[str, str | None]:
    """
    Resolve LoRA spec to a local safetensors path.
    Returns (path, cleanup_temp_path_or_none).
    """
    raw = _normalize_lora_spec(spec)
    if not raw:
        raise ValueError("Empty LoRA spec")

    cached = _lora_cached_path(raw)
    if cached is not None:
        log.debug("Using cached LoRA at %s", cached)
        return str(cached), None

    if _is_http_url(raw):
        resolved = _parse_hf_lora_resolve_url(raw)
        if resolved is not None:
            return str(_download_hf_lora_resolve(resolved)), None

        # Generic URL fallback (no 512MiB cap for LoRA artifacts).
        tmp = _download_remote_to_temp(
            raw,
            "fvserver_lora_",
            ".safetensors",
            max_bytes=None,
        )
        return tmp, tmp

    p = Path(raw).expanduser()
    if p.is_file():
        return str(p.resolve()), None

    if looks_like_hf_repo_id(raw):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required to download LoRA from Hugging Face"
            ) from e
        dest_root = _local_lora_cache_dir()
        dest = (dest_root / raw.replace("/", "__")).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        snap = _snapshot_download_weights(snapshot_download, raw, dest)
        snap_path = Path(snap)
        lora_file = _pick_safetensors_file(snap_path)
        if lora_file is None:
            raise RuntimeError(f"No .safetensors LoRA file found under {snap_path}")
        return str(lora_file.resolve()), None

    raise FileNotFoundError(f"LoRA spec not found or unsupported: {raw}")


def _decode_media_input(
    media_data: dict | str | None,
    *,
    temp_prefix: str,
    default_suffix: str,
) -> tuple[str | None, str | None]:
    """
    Resolve media input to a local path or URL.

    Returns: (resolved_path_or_url, temp_file_to_cleanup_or_none)
    """
    if media_data is None:
        return None, None

    if isinstance(media_data, str):
        raw = media_data.strip()
        if not raw:
            return None, None
        if raw.startswith(("http://", "https://")):
            tmp = _download_remote_to_temp(raw, temp_prefix, default_suffix)
            return tmp, tmp
        if raw.startswith("file://"):
            path = url2pathname(unquote(raw[7:]))
            if os.path.isfile(path):
                return path, None
            raise FileNotFoundError(f"File URL does not exist: {raw}")
        if os.path.isfile(raw):
            return raw, None
        raise FileNotFoundError(f"Media input not found: {raw}")

    if isinstance(media_data, dict):
        data_url = str(media_data.get("data_url") or "").strip()
        if not data_url:
            return None, None
        name_hint = str(
            media_data.get("name") or media_data.get("filename") or ""
        ).strip()
        name_suffix = Path(name_hint).suffix if name_hint else ""
        if data_url.startswith(("http://", "https://")):
            tmp = _download_remote_to_temp(data_url, temp_prefix, default_suffix)
            return tmp, tmp
        if data_url.startswith("file://"):
            path = url2pathname(unquote(data_url[7:]))
            if os.path.isfile(path):
                return path, None
            raise FileNotFoundError(f"File URL does not exist: {data_url}")
        if os.path.isfile(data_url):
            return data_url, None
        if data_url.startswith("data:"):
            header, encoded = data_url.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
        else:
            mime = str(media_data.get("mime_type") or "")
            encoded = data_url
        ext = name_suffix or mimetypes.guess_extension(mime) or default_suffix
        if ext == ".jpe":
            ext = ".jpg"
        fd, path = mk_scratch_file(prefix=temp_prefix, suffix=ext)
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(encoded))
        return path, path

    return None, None


def _decode_weighted_media_inputs(
    items: list[tuple[dict | str, float]] | None,
    *,
    temp_prefix: str,
    default_suffix: str,
) -> tuple[list[tuple[str, float]], list[str]]:
    decoded: list[tuple[str, float]] = []
    temps: list[str] = []
    for src, weight in (items or []):
        path, cleanup = _decode_media_input(
            src,
            temp_prefix=temp_prefix,
            default_suffix=default_suffix,
        )
        if path:
            decoded.append((path, float(weight)))
        if cleanup:
            temps.append(cleanup)
    return decoded, temps


def _apply_pending_loras(pipe: Any, lora_paths: list[tuple[str, float]] | None) -> None:
    """Attach request LoRAs so the next transformer load fuses them.

    ``ltx_pipelines_mlx`` only fuses ``_pending_loras`` inside
    ``BasePipeline._load_transformer_with_optional_streaming``. The attribute is
    optional (the CLI assigns it directly). We must always set it — a previous
    ``hasattr`` guard silently dropped Web UI / MCP LoRAs because Distilled /
    T2V pipes never pre-declare the attribute.

    IC-LoRA / HDR / LipDub pipelines fuse via constructor ``_lora_paths`` inside
    ``generate()``; leave those alone.

    When the DiT is already loaded under a different LoRA set (typical: ``load()``
    during ``_get_pipe`` with no LoRAs, then request LoRAs arrive here), reload so
    fusion actually happens.
    """
    # IC / FaceSwap own fusion via constructor LoRAs — never reload a clean DiT
    # here or the adapter is silently washed out (CrossView Stage-2 lesson).
    owned = getattr(pipe, "_lora_paths", None) or getattr(pipe, "_head_swap_lora", None)
    if owned:
        return

    desired = [(str(p), float(s)) for p, s in (lora_paths or [])]
    previous = [(str(p), float(s)) for p, s in (getattr(pipe, "_pending_loras", None) or [])]
    pipe._pending_loras = desired

    dit_loaded = getattr(pipe, "dit", None) is not None
    if not dit_loaded:
        if desired:
            log.info("Queued %d LoRA(s) for next transformer load", len(desired))
        return
    if desired == previous:
        return

    log.info(
        "Reloading transformer to apply %d LoRA(s) (previously %d)",
        len(desired),
        len(previous),
    )
    pipe.dit = None
    if hasattr(pipe, "_loaded"):
        pipe._loaded = False
    load_fn = getattr(pipe, "load", None)
    if callable(load_fn):
        load_fn()


def _is_crossview_lora_path(path: str) -> bool:
    lowered = (path or "").lower()
    return "crossview" in lowered or "cross-view" in lowered


def _tune_ic_lora_strengths(
    resolved_loras: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Raise CrossView LoRA strength on distilled workflows (author tip: 1.2–1.5)."""
    tuned: list[tuple[str, float]] = []
    for path, scale in resolved_loras:
        s = float(scale)
        if _is_crossview_lora_path(path) and s < 1.2:
            log.warning(
                "CrossView IC-LoRA on distilled is weak below 1.2 — raising strength "
                "%.2f → 1.25 (author tip: 1.2–1.5). Set scale explicitly to override.",
                s,
            )
            s = 1.25
        tuned.append((path, s))
    return tuned


def _clamp_conditioning_attention_strength(value: float | None) -> float | None:
    """ICLoraPipeline requires conditioning_attention_strength in [0, 1]."""
    if value is None:
        return None
    v = float(value)
    if v < 0.0 or v > 1.0:
        clamped = max(0.0, min(1.0, v))
        log.warning(
            "conditioning_attention_strength %.3f out of [0,1] — clamping to %.3f",
            v,
            clamped,
        )
        return clamped
    return v


def _pipeline_load_state_inconsistent(pipe: Any) -> bool:
    """True when ``_loaded`` is set but core weights the next job needs were freed."""
    if not getattr(pipe, "_loaded", False):
        return False
    if getattr(pipe, "dit", None) is None:
        return True
    if getattr(pipe, "vae_encoder", None) is None:
        return True
    return False


def _sync_pipeline_load_flag(pipe: Any) -> None:
    """Clear ``_loaded`` when freed blocks would make :meth:`load` skip a required reload."""
    if _pipeline_load_state_inconsistent(pipe):
        pipe._loaded = False


def _free_pipeline_blocks(pipe: Any) -> None:
    """Drop per-job MLX weights held on a cached pipeline instance."""
    if getattr(pipe, "dit", None) is not None:
        pipe.dit = None
    for block_name in (
        "prompt_encoder",
        "image_conditioner",
        "audio_conditioner",
        "video_decoder_block",
        "audio_decoder_block",
    ):
        block = getattr(pipe, block_name, None)
        if block is not None and hasattr(block, "free"):
            try:
                block.free()
            except Exception as exc:
                log.debug("Pipeline block free %s failed: %s", block_name, exc)
    if hasattr(pipe, "vae_encoder"):
        pipe.vae_encoder = None
    _sync_pipeline_load_flag(pipe)


def _mlx_aggressive_cleanup() -> None:
    try:
        from ltx_core_mlx.utils.memory import aggressive_cleanup

        aggressive_cleanup()
    except ImportError:
        pass


def _release_pipe_after_generation(pipe: Any) -> None:
    """Reset per-request pipeline state and free MLX blocks between jobs."""
    if hasattr(pipe, "_pending_loras"):
        pipe._pending_loras = []
    _free_pipeline_blocks(pipe)
    _mlx_aggressive_cleanup()


def _unlink_fvserver_temp(path: str | None, marker: str) -> None:
    if path and os.path.isfile(path) and marker in path:
        try:
            os.unlink(path)
        except OSError:
            pass


def _export_output_mp4(source_path: str) -> str:
    """Copy generation output to a standalone temp file (outside per-job workdirs).

    When HDR IC-LoRA wrote ``<stem>.hdr.npz`` next to the MP4, copy that sidecar too.
    """
    fd, final_path = mk_scratch_file(prefix="fvserver_out_", suffix=".mp4")
    os.close(fd)
    shutil.copy2(source_path, final_path)
    hdr_src = Path(source_path).with_suffix(".hdr.npz")
    if hdr_src.is_file():
        hdr_dest = Path(final_path).with_suffix(".hdr.npz")
        try:
            shutil.copy2(hdr_src, hdr_dest)
            log.info("Exported HDR tensor sidecar → %s", hdr_dest)
        except OSError as exc:
            log.warning("Could not export HDR .npz sidecar: %s", exc)
    return final_path


def _normalize_pipeline_profile(raw: str | None) -> str:
    profile = (raw or PIPE_PROFILE_DISTILLED).strip().lower()
    if profile in VALID_PIPELINE_PROFILES:
        return profile
    return PIPE_PROFILE_DISTILLED


def _maybe_enhance_prompt(
    prompt: str,
    *,
    mode: str,
    model_dir: str,
    enabled: bool,
) -> str:
    """Run ltx-2-mlx Gemma prompt enhancement when available and requested."""
    text = (prompt or "").strip()
    if not enabled or not text:
        return prompt
    enhance_mode = "i2v" if mode in ("generate", "i2v", "keyframe") else "t2v"
    try:
        import ltx_pipelines_mlx as lpm
    except ImportError:
        log.warning("enhance_prompt requested but ltx_pipelines_mlx is not installed")
        return prompt
    for attr in ("enhance_prompt", "enhance"):
        fn = getattr(lpm, attr, None)
        if callable(fn):
            try:
                out = fn(text, mode=enhance_mode, model_dir=model_dir)
                if isinstance(out, str) and out.strip():
                    log.info("Prompt enhanced via ltx_pipelines_mlx.%s", attr)
                    return out.strip()
            except TypeError:
                try:
                    out = fn(text, enhance_mode, model_dir)
                    if isinstance(out, str) and out.strip():
                        log.info("Prompt enhanced via ltx_pipelines_mlx.%s (legacy signature)", attr)
                        return out.strip()
                except Exception as exc:
                    log.warning("Prompt enhance via %s failed: %s", attr, exc)
            except Exception as exc:
                log.warning("Prompt enhance via %s failed: %s", attr, exc)
    log.warning(
        "enhance_prompt requested but no enhance API found in ltx_pipelines_mlx; using original prompt"
    )
    return prompt


def _apply_optional_generate_kwargs(call_kwargs: dict[str, Any], req: GenerationRequest) -> None:
    """Attach optional CFG / stage-2 / audio-regen flags when the pipeline accepts them."""
    if req.cfg_scale is not None:
        call_kwargs["cfg_scale"] = float(req.cfg_scale)
    if req.stg_scale is not None:
        call_kwargs["stg_scale"] = float(req.stg_scale)
    if req.stage2_steps is not None:
        call_kwargs["stage2_steps"] = int(req.stage2_steps)
    if req.no_regen_audio:
        # Upstream RetakePipeline uses regenerate_audio= (default True).
        call_kwargs["regenerate_audio"] = False
    if req.reference_strength is not None:
        call_kwargs["reference_strength"] = float(req.reference_strength)
    if req.audio_start_seconds is not None and float(req.audio_start_seconds) > 0:
        call_kwargs["audio_start_time"] = float(req.audio_start_seconds)
    neg = (req.negative_prompt or "").strip()
    if neg:
        call_kwargs["negative_prompt"] = neg


def _keyframe_pipe_kwargs(model_dir: str | Path | None) -> dict[str, Any]:
    """Ctor kwargs for KeyframeInterpolationPipeline (dev DiT + distilled LoRA)."""
    root = Path(model_dir) if model_dir else None
    distilled = KEYFRAME_DISTILLED_LORA
    if root is not None:
        if not (root / KEYFRAME_DEV_TRANSFORMER).is_file():
            raise RuntimeError(
                "Keyframe interpolation requires transformer-dev.safetensors in the model "
                "directory (dgrauet/ltx-2.3-mlx or ltx-2.3-mlx-q8 — not distilled-only)."
            )
        if (root / KEYFRAME_DISTILLED_LORA).is_file():
            distilled = KEYFRAME_DISTILLED_LORA
        elif (root / KEYFRAME_DISTILLED_LORA_ALT).is_file():
            distilled = KEYFRAME_DISTILLED_LORA_ALT
        else:
            raise RuntimeError(
                "Keyframe interpolation requires a distilled LoRA for stage 2 "
                f"({KEYFRAME_DISTILLED_LORA} or {KEYFRAME_DISTILLED_LORA_ALT}) in the model directory."
            )
    return {
        "dev_transformer": KEYFRAME_DEV_TRANSFORMER,
        "distilled_lora": distilled,
        "distilled_lora_strength": 1.0,
    }


def _clamp_a2v_stage1_steps(steps: int) -> int:
    """A2V is CFG two-stage; distilled UI defaults (8) are too low."""
    n = max(1, int(steps))
    if n < A2V_MIN_STAGE1_STEPS:
        log.warning(
            "a2v: raising stage1 steps from %d to %d (CFG two-stage; UI distilled defaults under-step)",
            n,
            A2V_DEFAULT_STAGE1_STEPS,
        )
        return A2V_DEFAULT_STAGE1_STEPS
    return n


def _a2v_effective_stage1_steps(steps: int, *, visual_i2v_continue: bool) -> int:
    """Steps for an a2v job.

    Native A2V (``A2VidPipelineTwoStage``) needs CFG stage-1 depth (~30).
    Visual autocontinue uses the *generate* profile pipe (distilled I2V by
    default) + audio mux — keep the caller's steps so we do not re-introduce
    the slow CFG/one-stage path the I2V profile fix removed.
    """
    if visual_i2v_continue:
        return max(1, int(steps))
    return _clamp_a2v_stage1_steps(steps)


def _frame_rate_from_kwargs(kwargs: dict[str, Any], default: float) -> float:
    if "frame_rate" in kwargs:
        return float(kwargs.pop("frame_rate"))
    if "fps" in kwargs:
        return float(kwargs.pop("fps"))
    return float(default)


def _decode_latents_to_mp4(
    pipe: Any,
    video_latent: Any,
    audio_latent: Any,
    output_path: str,
    frame_rate: float,
) -> None:
    if getattr(pipe, "low_memory", False):
        pipe.dit = None
        if hasattr(pipe, "prompt_encoder"):
            pipe.prompt_encoder.free()
        if hasattr(pipe, "image_conditioner"):
            pipe.image_conditioner.free()
        _sync_pipeline_load_flag(pipe)
        try:
            from ltx_core_mlx.utils.memory import aggressive_cleanup

            aggressive_cleanup()
        except ImportError:
            pass
    pipe._load_decoders()
    fn = getattr(pipe, "_decode_and_save_video", None)
    if fn is None:
        raise RuntimeError(f"{type(pipe).__name__} has no _decode_and_save_video()")
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    decode_kwargs: dict[str, Any] = {}
    if "frame_rate" in accepted:
        decode_kwargs["frame_rate"] = float(frame_rate)
    elif "fps" in accepted:
        decode_kwargs["fps"] = float(frame_rate)
    fn(video_latent, audio_latent, output_path, **decode_kwargs)


def _filter_call_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_varkw:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in accepted}


def _invoke_retake_and_save(pipe: Any, *, default_fps: float, **kwargs: Any) -> None:
    output_path = kwargs.pop("output_path")
    frame_rate = _frame_rate_from_kwargs(kwargs, default_fps)
    lora_paths = kwargs.pop("lora_paths", None)
    for drop_key in ("height", "width", "num_frames"):
        kwargs.pop(drop_key, None)
    _apply_pending_loras(pipe, lora_paths)
    video_latent, audio_latent = pipe.retake_from_video(
        **_filter_call_kwargs(pipe.retake_from_video, kwargs)
    )
    _decode_latents_to_mp4(pipe, video_latent, audio_latent, output_path, frame_rate)


def _invoke_extend_and_save(pipe: Any, *, default_fps: float, **kwargs: Any) -> None:
    output_path = kwargs.pop("output_path")
    frame_rate = _frame_rate_from_kwargs(kwargs, default_fps)
    lora_paths = kwargs.pop("lora_paths", None)
    for drop_key in ("height", "width", "num_frames"):
        kwargs.pop(drop_key, None)
    _apply_pending_loras(pipe, lora_paths)
    video_latent, audio_latent = pipe.extend_from_video(
        **_filter_call_kwargs(pipe.extend_from_video, kwargs)
    )
    _decode_latents_to_mp4(pipe, video_latent, audio_latent, output_path, frame_rate)


def _invoke_generate_and_save(pipe: Any, **kwargs: Any) -> None:
    """
    Call ``pipe.generate_and_save`` while tolerating API drift between ltx-2-mlx versions.

    - Drops unsupported kwargs.
    - Maps ``num_steps`` -> ``stage1_steps`` / ``steps`` when needed.
    - Maps ``fps`` -> ``frame_rate`` when upstream uses that name.
    - Applies request LoRAs via ``pipe._pending_loras`` when supported.
    """
    fn = getattr(pipe, "generate_and_save", None)
    if fn is None:
        raise RuntimeError(f"{type(pipe).__name__} has no generate_and_save()")

    sig = inspect.signature(fn)
    params = sig.parameters
    accepted = set(params.keys())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    call_kwargs = dict(kwargs)
    lora_paths = call_kwargs.pop("lora_paths", None)
    _apply_pending_loras(pipe, lora_paths)

    if "num_steps" in call_kwargs:
        steps = call_kwargs["num_steps"]
        if "stage1_steps" in accepted and "stage1_steps" not in call_kwargs and "num_steps" not in accepted:
            call_kwargs["stage1_steps"] = steps
        if "num_steps" not in accepted and "steps" in accepted:
            call_kwargs["steps"] = call_kwargs.pop("num_steps")
    if "fps" in call_kwargs and "fps" not in accepted and "frame_rate" in accepted:
        call_kwargs["frame_rate"] = float(call_kwargs.pop("fps"))
    elif "fps" in call_kwargs and "fps" not in accepted and "frame_rate" not in accepted:
        call_kwargs.pop("fps", None)
    if "frame_rate" in call_kwargs and "frame_rate" not in accepted and "fps" in accepted:
        call_kwargs["fps"] = float(call_kwargs.pop("frame_rate"))

    img = call_kwargs.get("image")
    if img and "image" not in accepted:
        for alias in (
            "image_path",
            "input_image",
            "reference_image",
            "init_image",
            "first_frame_image",
            "start_image",
            "start",
        ):
            if alias in accepted:
                call_kwargs[alias] = call_kwargs.pop("image")
                break

    end_img = call_kwargs.get("end_image")
    if end_img and "end_image" not in accepted:
        for alias in ("end_image_path", "end", "target_image", "last_frame_image"):
            if alias in accepted:
                call_kwargs[alias] = call_kwargs.pop("end_image")
                break

    vid = call_kwargs.get("video_path") or call_kwargs.get("reference_video")
    if vid:
        for primary, aliases in (
            ("video_path", ("video", "source_video", "source_video_path", "input_video")),
            ("reference_video", ("video_path", "video", "source_video")),
        ):
            if primary in call_kwargs and primary not in accepted:
                for alias in aliases:
                    if alias in accepted:
                        call_kwargs[alias] = call_kwargs.pop(primary)
                        break

    if not has_varkw:
        dropped_image = img and "image" not in call_kwargs and not any(
            k in call_kwargs for k in ("image_path", "input_image", "reference_image", "init_image")
        )
        if dropped_image:
            log.warning(
                "Pipeline %s.generate_and_save does not accept image= — I2V conditioning disabled",
                type(pipe).__name__,
            )
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted}

    fn(**call_kwargs)


def _mux_audio_into_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    *,
    duration_s: float,
) -> None:
    """Mux an audio track into a silent video (a2v chain visual continuation)."""
    from ltx_media import media_available, mux_audio_into_video

    if not media_available():
        raise RuntimeError("PyAV is required to mux audio into a2v autocontinue clips")
    mux_audio_into_video(video_path, audio_path, output_path, duration_s=duration_s)


class _ModelProgressStore:
    """Thread-safe denoising / download progress for WebSocket keepalives."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] | None = None

    def set(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._data = dict(data)

    def clear(self) -> None:
        with self._lock:
            self._data = None

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._data:
                return None
            snap = dict(self._data)
        step = snap.get("step")
        total = snap.get("total")
        if (
            snap.get("pct") is None
            and isinstance(step, (int, float))
            and isinstance(total, (int, float))
            and total > 0
        ):
            snap["pct"] = round(100 * float(step) / float(total), 0)
        return snap


class GenerationCancelledError(RuntimeError):
    """Raised when generation is cancelled via ``request_cancel()``."""


def _ic_lora_primary_lora(
    resolved_loras: list[tuple[str, float]],
) -> tuple[str, float] | None:
    if not resolved_loras:
        return None
    return resolved_loras[0]


def _ic_lora_uses_hdr_pipeline(resolved_loras: list[tuple[str, float]]) -> bool:
    """True when the primary LoRA is HDR (HDRICLoraPipeline + raw RGB ref video)."""
    primary = _ic_lora_primary_lora(resolved_loras)
    if primary is None:
        return False
    path, _ = primary
    try:
        from ltx_core_mlx.loader.hdr_metadata import read_hdr_lora_config

        if read_hdr_lora_config(path):
            return True
    except ImportError:
        pass
    lowered = path.lower()
    return "ic-lora-hdr" in lowered or "ic_lora_hdr" in lowered


def _ic_lora_reference_downscale_factor(lora_path: str) -> int:
    """IC-LoRA ref scale from safetensors metadata (Union Control = 2, HDR = 1)."""
    try:
        from ltx_pipelines_mlx.iclora_utils import read_lora_reference_downscale_factor

        return int(read_lora_reference_downscale_factor(lora_path))
    except ImportError:
        lowered = lora_path.lower()
        if "ref0.5" in lowered or "union-control" in lowered:
            return 2
        return 1


def _needs_pose_control_preprocessing(
    resolved_loras: list[tuple[str, float]],
    vc_items: list[tuple[str, float]],
) -> bool:
    """Control IC-LoRAs (ref downscale > 1) need pose/canny/depth — not raw RGB."""
    if not vc_items:
        return False
    primary = _ic_lora_primary_lora(resolved_loras)
    if primary is None:
        return False
    if _ic_lora_uses_hdr_pipeline(resolved_loras):
        return False
    return _ic_lora_reference_downscale_factor(primary[0]) != 1


def _build_ic_lora_image_conditionings(
    image_path: str,
    num_frames: int,
) -> list[tuple[str, int, float, int]]:
    """I2V frame-0 anchor only.

    IC-LoRA stage 2 accepts ``VideoConditionByLatentIndex`` (frame_idx==0) but
    not ``VideoConditionByKeyframeIndex`` — a last-frame keyframe appends extra
    tokens and breaks stage-2 ``unpatchify``.
    """
    del num_frames  # frame-0 only; see docstring
    return [(image_path, 0, 1.0, IC_LORA_IMAGE_CRF)]


def _prepare_ic_lora_video_conditioning(
    vc_items: list[tuple[str, float]],
    *,
    resolved_loras: list[tuple[str, float]],
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    tmpdir: str,
) -> tuple[list[tuple[str, float]], list[str]]:
    """Build IC-LoRA ``video_conditioning`` list (pose maps for motion transfer)."""
    cleanup: list[str] = []
    combined = [(str(p), float(s)) for p, s in vc_items]
    if not _needs_pose_control_preprocessing(resolved_loras, vc_items):
        log.info(
            "IC-LoRA video conditioning: %d raw reference clip(s) (primary=%s)",
            len(combined),
            _ic_lora_primary_lora(resolved_loras)[0] if _ic_lora_primary_lora(resolved_loras) else "?",
        )
        return combined, cleanup

    from ltx_ic_lora_preprocess import render_pose_control_video, require_pose_control

    require_pose_control()
    motion_path, motion_scale = combined[0]
    pose_path = os.path.join(tmpdir, "ic_lora_pose_control.mp4")
    render_pose_control_video(
        motion_path,
        pose_path,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
    )
    cleanup.append(pose_path)
    log.info(
        "IC-LoRA motion transfer: OpenPose control from %s (Union Control, %dx%d, %d frames)",
        motion_path,
        width,
        height,
        num_frames,
    )
    return [(pose_path, motion_scale)], cleanup


def _maybe_preserve_reference_audio(
    output_path: str,
    reference_video_paths: list[str],
    *,
    job_id: str | None = None,
) -> None:
    """Replace generated audio with the first reference clip that has an audio track."""
    from ltx_media import media_available, probe_video_info, replace_output_audio_from_source

    if not media_available():
        return
    for ref in reference_video_paths:
        if not ref or not os.path.isfile(ref):
            continue
        try:
            info = probe_video_info(ref)
        except Exception:
            continue
        if not info.has_audio:
            continue
        try:
            replace_output_audio_from_source(output_path, ref)
            log.info(
                "Preserved reference audio from %s (job=%s)",
                ref,
                (job_id or "?")[:8],
            )
            return
        except Exception as exc:
            log.warning("Could not preserve reference audio from %s: %s", ref, exc)


def _prepare_lipdub_reference_video(
    reference_video: str,
    voice_audio: str | None,
    *,
    tmpdir: str,
) -> tuple[str, list[str]]:
    """Build the LipDub reference clip with voice-tone audio for VAE reference tokens.

    LipDubPipeline reads audio from ``reference_video_path`` only. When the user
    supplies a separate voice-tone track, mux it onto a temp copy of the reference
    video (Comfy-style Load Audio). Otherwise use embedded video audio.
    """
    from ltx_media import media_available, mux_audio_into_video, probe_video_info

    cleanup: list[str] = []
    if not media_available():
        if voice_audio:
            raise RuntimeError("lipdub voice-tone audio mux requires PyAV (pip install av)")
        return reference_video, cleanup

    info = probe_video_info(reference_video)
    duration_s = float(info.duration or 0.0)
    if duration_s <= 0 and info.num_frames > 0 and info.fps > 0:
        duration_s = float(info.num_frames) / float(info.fps)
    duration_s = max(0.1, duration_s)

    if voice_audio:
        out = os.path.join(tmpdir, "lipdub_ref_with_voice_audio.mp4")
        mux_audio_into_video(reference_video, voice_audio, out, duration_s=duration_s)
        cleanup.append(out)
        log.info("LipDub: muxed voice-tone audio onto reference video (%s)", out)
        return out, cleanup

    if not info.has_audio:
        raise RuntimeError(
            "lipdub requires voice-tone audio: upload an audio file or use a reference "
            "video with an audio track"
        )
    return reference_video, cleanup


def _invoke_lipdub_style(
    pipe: Any,
    *,
    common_gen_kwargs: dict[str, Any],
    reference_video: str,
    tmp_image: str | None,
    num_frames: int,
    req: GenerationRequest,
) -> None:
    """LipDub: reference video (visual IC-LoRA + voice-tone audio) + optional I2V anchor."""
    _apply_ltx_mlx_patches(default_fps=float(common_gen_kwargs.get("frame_rate") or 24.0))
    lip_kwargs = dict(common_gen_kwargs)
    lip_kwargs["reference_video_path"] = reference_video
    if tmp_image:
        lip_kwargs["images"] = _build_ic_lora_image_conditionings(tmp_image, num_frames)
    if req.reference_strength is not None:
        lip_kwargs["reference_strength"] = float(req.reference_strength)
    _apply_optional_generate_kwargs(lip_kwargs, req)
    _invoke_generate_and_save(pipe, **lip_kwargs)


def _prepare_face_swap_guide_video(
    reference_path: str,
    identity_image: str,
    *,
    tmpdir: str,
    num_frames: int,
    width: int,
    height: int,
    fps: float,
) -> tuple[str, Any, int, int, int]:
    """Trim reference footage and build the BFS V3 composite guide clip.

    Returns ``(guide_path, layout, num_frames, canvas_width, canvas_height)``.
    Canvas size preserves source aspect (longer-edge resize); it is **not** stretched
    to the UI preset box.
    """
    from ltx_face_swap_compose import (
        FaceSwapGuideLayout,
        compose_bfs_v3_guide_video,
        compute_bfs_guide_layout,
        resolve_face_swap_canvas_size,
    )
    from ltx_ltxv_add_guide import vae_compatible_frame_count
    from ltx_media import (
        media_available,
        probe_video_info,
        trim_video_fit_aspect,
    )

    if not media_available():
        raise RuntimeError("face_swap requires PyAV to prepare guide video (pip install av)")

    info = probe_video_info(reference_path)
    canvas_w, canvas_h = resolve_face_swap_canvas_size(
        info.width,
        info.height,
        request_width=width,
        request_height=height,
    )
    if (canvas_w, canvas_h) != (width, height):
        log.info(
            "Face swap: canvas %dx%d from source %dx%d (aspect preserved; UI preset was %dx%d)",
            canvas_w,
            canvas_h,
            info.width,
            info.height,
            width,
            height,
        )
    vae_frames = vae_compatible_frame_count(num_frames, info.num_frames)
    if vae_frames != num_frames:
        log.info(
            "Face swap: adjusting target frames %d -> %d for LTX VAE (8k+1)",
            num_frames,
            vae_frames,
        )
    trimmed_path = os.path.join(tmpdir, "face_swap_ref_trimmed.mp4")
    guide_layout = compute_bfs_guide_layout(
        canvas_w,
        canvas_h,
        src_width=info.width,
        src_height=info.height,
        region_size_px=256,
    )
    if info.num_frames > vae_frames or abs(info.fps - fps) > 0.05:
        log.info(
            "Face swap: trimming reference video %d frames at %.1f fps -> %d frames at %.1f fps "
            "(main panel %dx%d, aspect preserved)",
            info.num_frames,
            info.fps,
            vae_frames,
            fps,
            guide_layout.video_w,
            guide_layout.video_h,
        )
    trim_video_fit_aspect(
        reference_path,
        trimmed_path,
        num_frames=vae_frames,
        max_width=guide_layout.video_w,
        max_height=guide_layout.video_h,
        fps=fps,
    )
    guide_path = os.path.join(tmpdir, "face_swap_bfs_v3_guide.mp4")
    layout = compose_bfs_v3_guide_video(
        trimmed_path,
        identity_image,
        guide_path,
        width=canvas_w,
        height=canvas_h,
        num_frames=vae_frames,
        fps=fps,
        region_size_px=256,
        layout=guide_layout,
    )
    return guide_path, layout, vae_frames, canvas_w, canvas_h


def _iclora_supports_control_aware_refine() -> bool:
    """True when installed ltx-2-mlx has ``upsample_only`` / ``refine_steps`` (0.14.17+)."""
    try:
        from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
    except ImportError:
        return False
    try:
        params = inspect.signature(ICLoraPipeline.generate_and_save).parameters
    except (TypeError, ValueError):
        return False
    return "upsample_only" in params and "refine_steps" in params


def _should_use_control_aware_refine(resolved_loras: list[tuple[str, float]]) -> bool:
    """Use upsample-only + control-aware refine for raw-RGB V2V.

    Empty LoRAs (pure reference conditioning) and community raw-RGB IC-LoRAs
    (ref_downscale=1, e.g. CrossView) use upstream 0.14.17+ ``upsample_only`` +
    ``refine_steps`` so Stage 2 does not wipe control / adapter effect.

    HDR and Union Control keep the legacy clean Stage 2 path.
    """
    if not resolved_loras:
        # Pure V2V / motion transfer without an adapter — still benefit from
        # full-res control re-append during refine.
        return True
    if _ic_lora_uses_hdr_pipeline(resolved_loras):
        return False
    primary = resolved_loras[0][0]
    if _ic_lora_reference_downscale_factor(primary) != 1:
        return False
    return True


def _run_ic_lora_generation(
    gen: "LocalVideoGenerator",
    *,
    req: GenerationRequest,
    prompt: str,
    resolved_loras: list[tuple[str, float]],
    vc_items: list[tuple[str, float]],
    tmp_image: str | None,
    tmpdir: str,
    out_path: str,
    width: int,
    height: int,
    nf: int,
    seed: int,
    steps: int,
    tmp_video_conditioning_cleanup: list[str],
    audio_reference_paths: list[str] | None = None,
    guide_images: list[tuple[str, int, float, int]] | None = None,
) -> Any:
    """Shared IC-LoRA invoke path for ``ic_lora`` / V2V (and face_swap guides).

    LoRAs are optional when a reference video is present (pure V2V). HDR LoRAs
    may omit video conditioning (pure T2V HDR), matching upstream ``hdr-ic-lora``.
    """
    is_hdr = bool(resolved_loras) and _ic_lora_uses_hdr_pipeline(resolved_loras)
    if not vc_items and not is_hdr:
        raise RuntimeError(
            "IC-LoRA / V2V requires a reference video "
            "(or an HDR LoRA for pure text-to-HDR)"
        )
    if not resolved_loras and not vc_items:
        raise RuntimeError("IC-LoRA / V2V requires a reference video and/or LoRA")

    ic_pipe_key = "hdr_ic_lora" if is_hdr else "ic_lora"
    # CrossView was trained on full LTX-2.3; strengthen on distilled few-step.
    resolved_loras = _tune_ic_lora_strengths(resolved_loras)
    primary_lora = _ic_lora_primary_lora(resolved_loras)
    uses_pose = _needs_pose_control_preprocessing(resolved_loras, vc_items)
    log.info(
        "IC-LoRA invoke: pipe=%s primary=%s vcond_in=%d image=%s (%d) pose_preprocess=%s",
        ic_pipe_key,
        primary_lora[0] if primary_lora else "(none)",
        len(vc_items),
        "guide" if guide_images else ("i2v" if tmp_image else "no"),
        len(guide_images) if guide_images else (1 if tmp_image else 0),
        "yes" if uses_pose else "no",
    )
    pipe = gen._get_pipe(
        ic_pipe_key,
        pipe_kwargs={
            "lora_paths": [(str(p), float(s)) for p, s in resolved_loras],
        },
    )
    ic_vcond, ic_vcond_cleanup = _prepare_ic_lora_video_conditioning(
        vc_items,
        resolved_loras=resolved_loras,
        width=width,
        height=height,
        num_frames=nf,
        fps=float(gen.fps),
        tmpdir=tmpdir,
    )
    tmp_video_conditioning_cleanup.extend(ic_vcond_cleanup)
    attn = _clamp_conditioning_attention_strength(
        float(req.reference_strength)
        if req.reference_strength is not None
        else 1.0
    )
    ic_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "output_path": out_path,
        "video_conditioning": ic_vcond,
        "height": height,
        "width": width,
        "num_frames": nf,
        "frame_rate": float(gen.fps),
        "seed": seed,
        "stage1_steps": int(steps),
        "conditioning_attention_strength": 1.0 if attn is None else float(attn),
    }
    if guide_images:
        ic_kwargs["images"] = guide_images
    elif tmp_image:
        ic_kwargs["images"] = _build_ic_lora_image_conditionings(tmp_image, nf)

    # Control-aware refine needs a reference clip; never pass upsample_only to HDR.
    use_control_aware = (
        bool(vc_items)
        and (not is_hdr)
        and _should_use_control_aware_refine(resolved_loras)
        and _iclora_supports_control_aware_refine()
    )
    if use_control_aware:
        # Upstream 0.14.17+: re-append control at full res; keep LoRA fused when present.
        refine_n = int(req.stage2_steps) if req.stage2_steps is not None else 3
        refine_n = max(1, min(refine_n, 8))
        ic_kwargs["upsample_only"] = True
        ic_kwargs["refine_steps"] = refine_n
        log.info(
            "IC-LoRA control-aware refine: upsample_only=True refine_steps=%d "
            "(ltx-2-mlx >= 0.14.17; LoRAs=%d)",
            refine_n,
            len(resolved_loras),
        )
    elif req.stage2_steps is not None:
        ic_kwargs["stage2_steps"] = int(req.stage2_steps)
    elif uses_pose and tmp_image:
        ic_kwargs["stage2_steps"] = 1
        log.info(
            "IC-LoRA Union motion transfer: stage2_steps=1 "
            "(override with stage2_steps in API)"
        )
    if bool(getattr(req, "skip_stage_2", False)):
        # Upstream hdr-ic-lora / ICLoraPipeline: half-res output, no upscale refine.
        ic_kwargs.pop("upsample_only", None)
        ic_kwargs.pop("refine_steps", None)
        ic_kwargs.pop("stage2_steps", None)
        ic_kwargs["skip_stage_2"] = True
        log.info("IC-LoRA skip_stage_2=True (half-res output, no upscale stage)")
    if primary_lora and _is_crossview_lora_path(primary_lora[0]):
        log.info(
            "CrossView V2V: prompt must use the fixed vocabulary "
            "(e.g. 'crossview. new camera angle: to the right, lower, closer.'); "
            "LoRA strength=%.2f attention=%.2f — see "
            "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Prompt",
            float(resolved_loras[0][1]),
            float(ic_kwargs["conditioning_attention_strength"]),
        )
        if not use_control_aware:
            log.warning(
                "CrossView without control-aware refine (upgrade ltx-2-mlx to "
                "%s+ for upsample_only/refine_steps). Legacy clean Stage 2 will "
                "weaken the camera-angle effect on distilled.",
                LTX2_MLX_GIT_TAG,
            )
    if not resolved_loras and vc_items:
        log.info(
            "V2V / IC-LoRA with no adapter LoRA — reference-video conditioning only "
            "(pure motion / structure transfer)"
        )
    if ic_vcond_cleanup:
        log.info(
            "IC-LoRA pose control video: %s — verify colored "
            "OpenPose skeletons before cleanup",
            ic_vcond_cleanup[0],
        )
    _invoke_generate_and_save(pipe, **ic_kwargs)
    if uses_pose and ic_vcond_cleanup and gen.spill_dir and req.job_id:
        try:
            gen.spill_dir.mkdir(parents=True, exist_ok=True)
            slug = _spill_slug(req.prompt)
            dest = gen.spill_dir / f"{req.job_id}_{slug}_pose_control.mp4"
            shutil.copy2(ic_vcond_cleanup[0], dest)
            log.info("IC-LoRA pose control saved → %s", dest)
        except OSError as exc:
            log.warning("Could not save pose control debug copy: %s", exc)
    if audio_reference_paths:
        _maybe_preserve_reference_audio(
            out_path,
            audio_reference_paths,
            job_id=req.job_id,
        )
    return pipe


# Comfy V3 Face Swap LoRA stack (see docs/FACESWAP_COMFY_GRAPH.md).
ID_LORA_CELEBVHQ_SPEC = (
    "https://huggingface.co/AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K/"
    "resolve/main/lora_weights.safetensors"
)
ID_LORA_TALKVID_SPEC = (
    "https://huggingface.co/AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K/"
    "resolve/main/lora_weights.safetensors"
)
ID_LORA_DEFAULT_SPEC = ID_LORA_CELEBVHQ_SPEC
ID_LORA_DEFAULT_SCALE = 1.0
DEFAULT_ID_LORA_STAGE1_STEPS = 20
DEFAULT_ID_LORA_STAGE1_STEPS_FAITHFUL = 30


FACE_SWAP_DISTILLED_DYNAMIC_SPEC = (
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/"
    "loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"
)
FACE_SWAP_DISTILLED_DYNAMIC_SCALE = 1.0
FACE_SWAP_DISTILLED_DYNAMIC_NAMES = (
    "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "ltx-2.3-22b-distilled-lora-dynamic.safetensors",
)
ENV_FACE_SWAP_DISTILLED_LORA = "LTX_WS_FACE_SWAP_DISTILLED_LORA"
ENV_FACE_SWAP_NO_DISTILLED_LORA = "LTX_WS_FACE_SWAP_NO_DISTILLED_LORA"


def _find_local_face_swap_distilled_dynamic(model_dir: Path | None = None) -> Path | None:
    """Prefer an already-downloaded distilled-dynamic LoRA over a fresh HF fetch."""
    candidates: list[Path] = []
    if model_dir is not None:
        root = Path(model_dir)
        for name in FACE_SWAP_DISTILLED_DYNAMIC_NAMES:
            candidates.append(root / name)
            candidates.append(root / "loras" / name)
    cache = _local_lora_cache_dir()
    for name in FACE_SWAP_DISTILLED_DYNAMIC_NAMES:
        candidates.append(cache / name)
        candidates.append(cache / "loras" / name)
        candidates.append(cache / "Kijai__LTX2.3_comfy" / "loras" / name)
        candidates.append(cache / "Kijai__LTX2.3_comfy" / name)
    for hit in candidates:
        if hit.is_file():
            return hit.resolve()
    return None


def _build_face_swap_lora_stack(
    head_swap_loras: list[tuple[str, float]],
    *,
    model_dir: Path | None = None,
) -> list[tuple[str, float]]:
    """Comfy V3 order: distilled-dynamic @1.0, then user head-swap LoRA(s).

    Skip with ``LTX_WS_FACE_SWAP_NO_DISTILLED_LORA=1``. Override path/URL via
    ``LTX_WS_FACE_SWAP_DISTILLED_LORA``.
    """
    if not head_swap_loras:
        raise RuntimeError("Face swap requires exactly one head-swap LoRA")

    stack: list[tuple[str, float]] = []
    skip = os.environ.get(ENV_FACE_SWAP_NO_DISTILLED_LORA, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not skip:
        override = os.environ.get(ENV_FACE_SWAP_DISTILLED_LORA, "").strip()
        try:
            if override:
                path, _cleanup = _resolve_lora_path(override)
                stack.append((path, FACE_SWAP_DISTILLED_DYNAMIC_SCALE))
            else:
                local = _find_local_face_swap_distilled_dynamic(model_dir)
                if local is not None:
                    stack.append((str(local), FACE_SWAP_DISTILLED_DYNAMIC_SCALE))
                else:
                    path, _cleanup = _resolve_lora_path(FACE_SWAP_DISTILLED_DYNAMIC_SPEC)
                    stack.append((path, FACE_SWAP_DISTILLED_DYNAMIC_SCALE))
            log.info(
                "Face swap LoRA stack: distilled-dynamic @%.2f → %s",
                stack[-1][1],
                Path(stack[-1][0]).name,
            )
        except Exception as exc:
            log.warning(
                "Face swap: distilled-dynamic LoRA unavailable (%s: %s); "
                "continuing with head-swap only (identity transfer may be weak)",
                type(exc).__name__,
                exc,
            )
    else:
        log.info("Face swap: distilled-dynamic LoRA skipped (%s=1)", ENV_FACE_SWAP_NO_DISTILLED_LORA)

    stack.extend((str(p), float(s)) for p, s in head_swap_loras)
    return stack


def _resolve_id_lora_stage1_size(
    image_path: str,
    *,
    requested_height: int | None = None,
    requested_width: int | None = None,
) -> tuple[int, int]:
    """Stage-1 HxW from first-frame aspect (long side ≤512), matching upstream ID-LoRA."""
    from ltx_id_lora_pipeline import compute_id_lora_stage1_resolution, snap_to_divisor

    if requested_height and requested_width:
        long_side = max(int(requested_height), int(requested_width))
        if long_side <= 512:
            return snap_to_divisor(int(requested_height)), snap_to_divisor(int(requested_width))

    try:
        from PIL import Image as PILImage

        with PILImage.open(image_path) as im:
            src_w, src_h = im.size
    except Exception as exc:
        raise RuntimeError(f"id_lora could not read reference image: {image_path}") from exc
    return compute_id_lora_stage1_resolution(src_h, src_w)


def _run_id_lora_generation(
    gen: "LocalVideoGenerator",
    *,
    req: GenerationRequest,
    prompt: str,
    resolved_loras: list[tuple[str, float]],
    image_path: str,
    audio_path: str,
    out_path: str,
    nf: int,
    seed: int,
    steps: int,
    requested_height: int,
    requested_width: int,
) -> Any:
    """Two-stage ID-LoRA: identity audio IC + first-frame I2V → distilled refine."""
    if not image_path:
        raise RuntimeError("id_lora mode requires a first-frame reference image")
    if not audio_path:
        raise RuntimeError("id_lora mode requires reference audio")
    if len(resolved_loras) != 1:
        raise RuntimeError("id_lora requires exactly one ID-LoRA adapter")

    stage1_h, stage1_w = _resolve_id_lora_stage1_size(
        image_path,
        requested_height=requested_height,
        requested_width=requested_width,
    )
    skip_stage_2 = bool(getattr(req, "skip_stage_2", False))
    upsample_only = bool(getattr(req, "upsample_only", False))
    if skip_stage_2 and upsample_only:
        raise RuntimeError("id_lora: skip_stage_2 and upsample_only are mutually exclusive")
    out_h = stage1_h if skip_stage_2 else stage1_h * 2
    out_w = stage1_w if skip_stage_2 else stage1_w * 2
    stage1_steps = int(steps)
    if req.num_steps is None:
        stage1_steps = DEFAULT_ID_LORA_STAGE1_STEPS

    audio_bytes = 0
    try:
        audio_bytes = Path(audio_path).stat().st_size
    except OSError:
        pass
    log.info(
        "ID-LoRA invoke: stage1=%dx%d → output=%dx%d frames=%d steps=%d lora=%s "
        "audio_path=%s audio_bytes=%d skip_stage_2=%s upsample_only=%s "
        "(ref audio = voice identity only; spoken words come from [SPEECH] in the prompt, not the WAV)",
        stage1_h,
        stage1_w,
        out_h,
        out_w,
        nf,
        stage1_steps,
        Path(resolved_loras[0][0]).name,
        str(Path(audio_path).resolve()) if audio_path else audio_path,
        audio_bytes,
        skip_stage_2,
        upsample_only,
    )
    pipe = gen._get_pipe(
        "id_lora",
        pipe_kwargs={"lora_paths": [(str(p), float(s)) for p, s in resolved_loras]},
    )
    from ltx_id_lora_pipeline import (
        DEFAULT_AUDIO_CFG,
        DEFAULT_IDENTITY_GUIDANCE_SCALE,
        DEFAULT_MODALITY_SCALE,
        DEFAULT_STG_SCALE,
        DEFAULT_VIDEO_CFG,
        IDLoraTwoStagesPipeline,
    )

    if not isinstance(pipe, IDLoraTwoStagesPipeline):
        raise RuntimeError(
            f"id_lora expected IDLoraTwoStagesPipeline, got {type(pipe).__name__}"
        )

    id_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "output_path": out_path,
        "image": image_path,
        "audio_path": audio_path,
        "height": stage1_h,
        "width": stage1_w,
        "num_frames": nf,
        "frame_rate": float(gen.fps),
        "seed": seed,
        "stage1_steps": stage1_steps,
        "cfg_scale": float(DEFAULT_VIDEO_CFG),
        "audio_cfg_scale": float(DEFAULT_AUDIO_CFG),
        "stg_scale": float(DEFAULT_STG_SCALE),
        "modality_scale": float(DEFAULT_MODALITY_SCALE),
        "identity_guidance_scale": float(DEFAULT_IDENTITY_GUIDANCE_SCALE),
        "skip_stage_2": skip_stage_2,
        "upsample_only": upsample_only,
    }
    if req.cfg_scale is not None:
        id_kwargs["cfg_scale"] = float(req.cfg_scale)
    if req.stg_scale is not None:
        id_kwargs["stg_scale"] = float(req.stg_scale)
    if getattr(req, "modality_scale", None) is not None:
        id_kwargs["modality_scale"] = float(req.modality_scale)
    if req.stage2_steps is not None:
        id_kwargs["stage2_steps"] = int(req.stage2_steps)
    _invoke_generate_and_save(pipe, **id_kwargs)
    return pipe


def _run_face_swap_generation(
    gen: "LocalVideoGenerator",
    *,
    req: GenerationRequest,
    prompt: str,
    resolved_loras: list[tuple[str, float]],
    guide_path: str,
    tmpdir: str,
    out_path: str,
    width: int,
    height: int,
    nf: int,
    seed: int,
    steps: int,
) -> Any:
    """BFS V3 face swap via Comfy LTXVAddGuide + distilled-dynamic + head-swap LoRA."""
    if len(resolved_loras) != 1:
        raise RuntimeError("Face swap requires exactly one head-swap LoRA")

    model_dir = Path(getattr(gen, "model_dir", "") or "")
    lora_stack = _build_face_swap_lora_stack(
        [(str(p), float(s)) for p, s in resolved_loras],
        model_dir=model_dir if model_dir.is_dir() else None,
    )
    log.info(
        "Face swap invoke: FaceSwapPipeline loras=%s guide=%s (%dx%d, %d frames) "
        "add_guide=full_composite crop_guides=yes schedule=DISTILLED_SIGMAS",
        [Path(p).name for p, _ in lora_stack],
        guide_path,
        width,
        height,
        nf,
    )
    pipe = gen._get_pipe(
        "face_swap",
        pipe_kwargs={"lora_paths": lora_stack},
    )
    from ltx_face_swap_pipeline import DEFAULT_FACE_SWAP_CFG, FaceSwapPipeline

    if not isinstance(pipe, FaceSwapPipeline):
        raise RuntimeError(
            f"face_swap expected FaceSwapPipeline (dev+Comfy LoRA stack), got {type(pipe).__name__}; "
            "update ltx-ws faceswap branch"
        )
    from ltx_ltxv_add_guide import DEFAULT_GUIDE_CRF

    swap_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "output_path": out_path,
        "guide_video_path": guide_path,
        "height": height,
        "width": width,
        "num_frames": nf,
        "frame_rate": float(gen.fps),
        "seed": seed,
        "num_steps": int(steps),
        "guide_crf": DEFAULT_GUIDE_CRF,
        # Comfy V3 default; request override still wins below.
        "cfg_scale": float(DEFAULT_FACE_SWAP_CFG),
    }
    if req.reference_strength is not None:
        swap_kwargs["guide_strength"] = float(req.reference_strength)
    if req.stage2_steps is not None:
        log.info("Face swap: stage2_steps ignored (single-stage V3; keep LoRAs fused)")
    if req.cfg_scale is not None:
        swap_kwargs["cfg_scale"] = float(req.cfg_scale)
    _apply_optional_generate_kwargs(swap_kwargs, req)
    _invoke_generate_and_save(pipe, **swap_kwargs)
    return pipe


def _stage_from_tqdm_desc(desc: str) -> str:
    d = (desc or "").strip().lower()
    if "denois" in d or "id-lora" in d or "id_lora" in d:
        return "denoising"
    if "download" in d:
        return "downloading"
    if any(k in d for k in ("encod", "decod", "vae", "latent")):
        return "encoding"
    if "upscal" in d:
        return "upscaling"
    return "generating"


class LocalVideoGenerator:
    """
    MLX pipeline adapter for ``ltx-2-mlx``: text/image/audio/video generation modes.
    Weights are resolved once at ``load()``; individual pipelines lazy-load on demand.
    """

    def __init__(
        self,
        model: str,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        model_dir: str | None,
        inference_steps: int,
        default_lora_specs: list[tuple[str, float]] | None = None,
        spill_dir: Path | None = None,
        low_memory: bool = False,
        *,
        upscale: bool = False,
    ) -> None:
        self.model = model
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.fps = float(fps)
        self.model_dir = model_dir
        self.inference_steps = max(1, int(inference_steps))
        self.default_lora_specs = list(default_lora_specs or [])
        self.spill_dir = spill_dir
        self.low_memory = bool(low_memory)
        # Backward-compatible ctor arg used by server.py CLI.
        self.upscale = bool(upscale)
        self._model_path: str | None = None
        self._pipe_classes: dict[str, Any] = {}
        self._pipes: dict[str, Any] = {}
        self._resolved_default_loras: list[tuple[str, float]] | None = None
        self._lpm_module: Any | None = None
        self._model_progress = _ModelProgressStore()
        self._cancel_requested = threading.Event()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ltx-gen",
        )
        self._executor_shutdown = False

    def shutdown(self, *, wait: bool = False) -> None:
        """Release the generation thread pool (call on server exit)."""
        self.request_cancel()
        if self._executor_shutdown:
            return
        self._executor.shutdown(wait=wait, cancel_futures=True)
        self._executor_shutdown = True

    def clear_cancel(self) -> None:
        self._cancel_requested.clear()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def _check_cancel(self) -> None:
        if self._cancel_requested.is_set():
            raise GenerationCancelledError("Generation cancelled")

    @contextmanager
    def _track_model_progress(self):
        """Patch tqdm so denoising step bars update ``model_progress_for_ws``."""
        try:
            import tqdm as tqdm_mod
        except ImportError:
            yield
            return

        generator = self
        orig_tqdm = tqdm_mod.tqdm
        orig_auto = getattr(tqdm_mod.auto, "tqdm", orig_tqdm)

        class _TrackingTqdm(orig_tqdm):  # type: ignore[misc,valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                generator._check_cancel()
                super().__init__(*args, **kwargs)
                self._publish(generator)

            def __iter__(self):
                # Prefer driving update(1) ourselves: upstream tqdm.__iter__ keeps a
                # local ``n`` and only calls update() when mininterval elapses, which
                # can starve WebSocket keepalives on fast bars. Explicit update()
                # always publishes model_progress for the UI.
                if self.disable:
                    yield from self.iterable
                    return
                try:
                    for item in self.iterable:
                        generator._check_cancel()
                        yield item
                        self.update(1)
                finally:
                    self.close()

            def refresh(self, *args: Any, **kwargs: Any) -> None:
                generator._check_cancel()
                super().refresh(*args, **kwargs)
                self._publish(generator)

            def update(self, n: float = 1) -> bool | None:
                generator._check_cancel()
                result = super().update(n)
                self._publish(generator)
                return result

            def _publish(self, gen: LocalVideoGenerator) -> None:
                desc = str(self.desc or "")
                fd = getattr(self, "format_dict", None) or {}
                n = int(self.n)
                total = int(self.total) if self.total is not None else None
                rate = fd.get("rate")
                tqdm_elapsed = fd.get("elapsed")
                eta_s: float | None = None
                avg_step_s: float | None = None
                if isinstance(rate, (int, float)) and rate > 0:
                    avg_step_s = round(1.0 / float(rate), 2)
                    if total is not None:
                        eta_s = round((total - n) / float(rate), 1)
                gen._model_progress.set(
                    {
                        "stage": _stage_from_tqdm_desc(desc),
                        "step": n,
                        "total": total,
                        "eta_s": eta_s,
                        "avg_step_s": avg_step_s,
                        "elapsed_s": (
                            round(float(tqdm_elapsed), 1)
                            if isinstance(tqdm_elapsed, (int, float))
                            else None
                        ),
                        "label": desc.strip() or None,
                    }
                )

        tqdm_mod.tqdm = _TrackingTqdm
        tqdm_mod.auto.tqdm = _TrackingTqdm
        # Modules that did ``from tqdm import tqdm`` keep a stale class binding;
        # patch those names too (samplers stage-2 path, and any leftover ID-LoRA binds).
        stale_bindings: list[tuple[Any, str, Any]] = []
        for mod_name, attr in (
            ("ltx_pipelines_mlx.utils.samplers", "tqdm"),
            ("ltx_id_lora_pipeline", "tqdm"),
        ):
            try:
                mod = __import__(mod_name, fromlist=["*"])
            except ImportError:
                continue
            if hasattr(mod, attr):
                stale_bindings.append((mod, attr, getattr(mod, attr)))
                setattr(mod, attr, _TrackingTqdm)
        try:
            yield
        finally:
            tqdm_mod.tqdm = orig_tqdm
            tqdm_mod.auto.tqdm = orig_auto
            for mod, attr, orig in stale_bindings:
                setattr(mod, attr, orig)
            self._model_progress.clear()

    def _resolve_model_dir(self) -> str:
        return resolve_mlx_weights_directory(self.model, self.model_dir)

    def load(self) -> None:
        _apply_ltx_mlx_patches(default_fps=self.fps)
        if self._model_path is not None:
            return
        try:
            import ltx_pipelines_mlx as lpm
        except ImportError as e:
            raise RuntimeError(
                "Missing ltx_pipelines_mlx. Install the MLX monorepo packages, e.g.:\n"
                f"{ltx2_mlx_install_hint()}"
            ) from e
        path = self._resolve_model_dir()
        self._model_path = path
        self._lpm_module = lpm

        if get_vae_decoder_variant() == "pruna":
            log.info("Resolving PrunaVAED MLX decoder weights (same Hub flow as --model) …")
            ensure_pruna_vae_decoder_files(Path(path))

        generate_cls = getattr(lpm, "DistilledPipeline", None)
        if generate_cls is None:
            generate_cls = getattr(lpm, "TextToVideoPipeline", None)
        if self.upscale:
            upscale_cls = getattr(lpm, "TI2VidTwoStagesPipeline", None)
            if upscale_cls is not None:
                generate_cls = upscale_cls
                log.info("Using TI2VidTwoStagesPipeline for --upscale generate jobs")

        legacy_t2v_cls = getattr(lpm, "TextToVideoPipeline", None)
        legacy_i2v_cls = getattr(lpm, "ImageToVideoPipeline", None)
        one_stage_cls = getattr(lpm, "TI2VidOneStagePipeline", None)

        a2v_cls = getattr(lpm, "A2VidPipelineTwoStage", None)
        if a2v_cls is None:
            a2v_cls = getattr(lpm, "AudioToVideoPipeline", None)

        retake_cls = getattr(lpm, "RetakePipeline", None)
        extend_cls = retake_cls if retake_cls is not None else getattr(lpm, "ExtendPipeline", None)

        self._pipe_classes: dict[str, Any] = {}
        if legacy_t2v_cls is not None:
            self._pipe_classes["t2v"] = legacy_t2v_cls
        elif generate_cls is not None:
            self._pipe_classes["t2v"] = generate_cls
        # Upstream: I2V is ``image=`` on the same generate pipeline — not a separate
        # slow TI2VidOneStagePipeline. Keep ``i2v`` as an alias of the fast/default
        # generate class; ``one_stage`` is opt-in via pipeline_profile.
        if legacy_i2v_cls is not None:
            self._pipe_classes["i2v"] = legacy_i2v_cls
            log.info("Using ImageToVideoPipeline for legacy i2v alias")
        elif generate_cls is not None:
            self._pipe_classes["i2v"] = generate_cls
            log.info(
                "I2V uses %s via image= (same pipeline as T2V; profile selects speed/quality)",
                getattr(generate_cls, "__name__", generate_cls),
            )
        if one_stage_cls is not None:
            self._pipe_classes["one_stage"] = one_stage_cls
            log.info("Registered MLX pipeline one_stage (TI2VidOneStagePipeline)")
        if generate_cls is not None:
            self._pipe_classes["gen"] = generate_cls
        if a2v_cls is not None:
            self._pipe_classes["a2v"] = a2v_cls
        if retake_cls is not None:
            self._pipe_classes["retake"] = retake_cls
        if extend_cls is not None:
            self._pipe_classes["extend"] = extend_cls

        ic_cls = getattr(lpm, "ICLoraPipeline", None)
        if ic_cls is not None:
            self._pipe_classes["ic_lora"] = ic_cls
        hdr_ic_cls = getattr(lpm, "HDRICLoraPipeline", None)
        if hdr_ic_cls is not None:
            self._pipe_classes["hdr_ic_lora"] = hdr_ic_cls

        for key, cls_name in (
            ("two_stage", "TI2VidTwoStagesPipeline"),
            ("hq", "TI2VidTwoStagesHQPipeline"),
            ("keyframe", "KeyframeInterpolationPipeline"),
            ("lipdub", "LipDubPipeline"),
        ):
            cls = getattr(lpm, cls_name, None)
            if cls is not None:
                self._pipe_classes[key] = cls
                log.info("Registered MLX pipeline %s (%s)", key, cls_name)

        try:
            from ltx_face_swap_pipeline import FaceSwapPipeline

            self._pipe_classes["face_swap"] = FaceSwapPipeline
            log.info("Registered MLX pipeline face_swap (FaceSwapPipeline)")
        except ImportError as exc:
            log.warning("FaceSwapPipeline unavailable: %s", exc)

        try:
            from ltx_id_lora_pipeline import IDLoraTwoStagesPipeline

            self._pipe_classes["id_lora"] = IDLoraTwoStagesPipeline
            log.info("Registered MLX pipeline id_lora (IDLoraTwoStagesPipeline)")
        except ImportError as exc:
            log.warning("IDLoraTwoStagesPipeline unavailable: %s", exc)

        # Legacy standalone spatial upscaler classes (pre-v0.14 monolith pipelines).
        for cls_name in (
            "SpatialUpscalerX2V11Pipeline",
            "SpatialUpscalerX2Pipeline",
            "SpatialUpscalerPipeline",
            "LTXSpatialUpscalerPipeline",
        ):
            up_cls = getattr(lpm, cls_name, None)
            if up_cls is not None:
                self._pipe_classes["spatial_upscaler"] = up_cls
                log.info("Detected spatial upscaler pipeline class: %s", cls_name)
                break
        log.info("MLX model path resolved ✓ %s", path)

    def _get_pipe(self, key: str, *, pipe_kwargs: dict[str, Any] | None = None) -> Any:
        _apply_ltx_mlx_patches(default_fps=self.fps)
        if not pipe_kwargs and key in self._pipes:
            return self._pipes[key]
        self.load()
        if self._model_path is None:
            raise RuntimeError("MLX model path not initialized")
        cls = self._pipe_classes.get(key)
        if cls is None:
            raise RuntimeError(
                f"Unsupported pipeline key: {key} (installed ltx-2-mlx may be too old; "
                f"expected {LTX2_MLX_GIT_TAG}+)"
            )
        log.info("Loading MLX pipeline %s from %s …", key, self._model_path)
        ctor_kwargs: dict[str, Any] = {"model_dir": self._model_path, "low_memory": self.low_memory}
        if pipe_kwargs:
            ctor_kwargs.update(pipe_kwargs)
        pipe = cls(**ctor_kwargs)
        if key not in ("retake", "extend") and hasattr(pipe, "load"):
            pipe.load()
        if not pipe_kwargs:
            self._pipes[key] = pipe
        log.info("MLX pipeline ready ✓ (%s)", key)
        return pipe

    def _resolve_generate_pipe_key(self, profile: str, *, has_image: bool = False) -> str:
        """Pick generate pipeline from ``pipeline_profile`` (not from image presence).

        Upstream ltx-2-mlx supports I2V on every generate pipeline via ``image=``.
        Switching to TI2VidOneStagePipeline just because a start image is set was
        causing I2V to run the slow full-res CFG/dev path while T2V stayed on
        DistilledPipeline.
        """
        del has_image  # kept for call-site compatibility; image does not change pipe
        profile = _normalize_pipeline_profile(profile)
        if profile == PIPE_PROFILE_TWO_STAGE and "two_stage" in self._pipe_classes:
            return "two_stage"
        if profile == PIPE_PROFILE_HQ and "hq" in self._pipe_classes:
            return "hq"
        if profile == PIPE_PROFILE_ONE_STAGE and "one_stage" in self._pipe_classes:
            return "one_stage"
        if self.upscale and "two_stage" in self._pipe_classes:
            return "two_stage"
        return "t2v"

    def _resolve_lora_specs(self, specs: list[tuple[str, float]]) -> tuple[list[tuple[str, float]], list[str]]:
        resolved: list[tuple[str, float]] = []
        temps: list[str] = []
        for lora_spec, lora_scale in specs:
            lora_path, cleanup = _resolve_lora_path(str(lora_spec))
            resolved.append((lora_path, float(lora_scale)))
            if cleanup:
                temps.append(cleanup)
        return resolved, temps

    def ensure_default_loras_ready(self) -> None:
        """
        Resolve/download default LoRAs at startup when LoRA mode is enabled.
        """
        self.load()
        if not self.default_lora_specs:
            self._resolved_default_loras = []
            return
        resolved, temps = self._resolve_lora_specs(self.default_lora_specs)
        for tmp in temps:
            if tmp and os.path.isfile(tmp) and "fvserver_lora_" in tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        self._resolved_default_loras = resolved
        log.info("Resolved %d default LoRA(s) for global use", len(resolved))

    def model_progress_for_ws(self) -> dict[str, Any] | None:
        return self._model_progress.snapshot()

    def cleanup_after_generation(self, pipe: Any | None = None) -> None:
        """Clear progress state and MLX memory after each job (success or failure)."""
        self._model_progress.clear()
        if pipe is not None:
            _release_pipe_after_generation(pipe)
        else:
            _mlx_aggressive_cleanup()

    def default_lora_count(self) -> int:
        if self._resolved_default_loras is not None:
            return len(self._resolved_default_loras)
        return len(self.default_lora_specs)

    def _calculate_stage1_dimensions(self, height: int, width: int) -> tuple[int, int]:
        base_h = _align_ltx2_spatial(max(LTX2_SPATIAL_ALIGN, int(round(height / 2.0))))
        base_w = _align_ltx2_spatial(max(LTX2_SPATIAL_ALIGN, int(round(width / 2.0))))
        return base_h, base_w

    def _run_spatial_upscaler_stage(
        self,
        *,
        prompt: str,
        source_video_path: str,
        output_path: str,
        height: int,
        width: int,
        num_frames: int,
        seed: int,
        num_steps: int,
        lora_paths: list[tuple[str, float]],
    ) -> bool:
        try:
            pipe = self._get_pipe("spatial_upscaler")
        except Exception as exc:
            log.warning(
                "Spatial upscaler pipeline unavailable; using first-stage output only: %s",
                exc,
            )
            return False

        try:
            sig = inspect.signature(pipe.generate_and_save)
            accepted = set(sig.parameters.keys())
            call_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "output_path": output_path,
                "num_frames": num_frames,
                "fps": float(self.fps),
                "seed": seed,
                "num_steps": num_steps,
                "lora_paths": lora_paths,
            }
            # Stage-2 source size comes from source_video_path; these are output target dimensions.
            if "target_height" in accepted and "target_width" in accepted:
                call_kwargs["target_height"] = height
                call_kwargs["target_width"] = width
            elif "height" in accepted and "width" in accepted:
                call_kwargs["height"] = height
                call_kwargs["width"] = width

            # Backend compatibility: select only one supported input-video arg name.
            for name in (
                "video",
                "video_path",
                "source_video",
                "source_video_path",
                "input_video",
                "input_video_path",
            ):
                if name in accepted:
                    call_kwargs[name] = source_video_path
                    break

            # Backend compatibility: pick the first recognized control by preference:
            # explicit boolean flags first, then string sampler-name style controls.
            for name, value in (
                ("use_tiled_sampler", True),
                ("tiled", True),
                ("sampler", "tiled"),
                ("sampler_name", "tiled"),
                ("sampling_method", "tiled"),
                ("second_sampler", "tiled"),
            ):
                if name in accepted:
                    call_kwargs[name] = value
                    break

            _invoke_generate_and_save(
                pipe,
                **call_kwargs,
            )
        except Exception as exc:
            log.warning(
                "Spatial upscaler second stage failed; using first-stage output only: %s",
                exc,
            )
            return False

        out = Path(output_path)
        if not (out.is_file() and out.stat().st_size > 0):
            log.warning(
                "Spatial upscaler produced no output; using first-stage output only: %s",
                output_path,
            )
            return False
        return True

    async def generate(
        self,
        prompt: str,
        image_data: dict | str | None = None,
        audio_data: dict | str | None = None,
        source_video_data: dict | str | None = None,
        seed: int = -1,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        negative_prompt: str = "",
        mode: str = "generate",
        num_steps: int | None = None,
        retake_start: int | None = None,
        retake_end: int | None = None,
        extend_frames: int | None = None,
        extend_direction: str = "after",
        lora_specs: list[tuple[str, float]] | None = None,
        video_conditioning_specs: list[tuple[dict | str, float]] | None = None,
        *,
        job_id: str | None = None,
        a2v_visual_i2v_continue: bool = False,
        end_image_data: dict | str | None = None,
        enhance_prompt: bool = False,
        pipeline_profile: str = PIPE_PROFILE_DISTILLED,
        cfg_scale: float | None = None,
        stg_scale: float | None = None,
        stage2_steps: int | None = None,
        no_regen_audio: bool = False,
        reference_strength: float | None = None,
        audio_start_seconds: float | None = None,
        skip_stage_2: bool = False,
        upsample_only: bool = False,
        modality_scale: float | None = None,
    ) -> str:
        self.clear_cancel()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(
                self._generate_sync,
                GenerationRequest(
                    prompt=prompt,
                    image_data=image_data,
                    audio_data=audio_data,
                    source_video_data=source_video_data,
                    seed=seed,
                    num_frames=num_frames or self.num_frames,
                    height=height or self.height,
                    width=width or self.width,
                    negative_prompt=negative_prompt,
                    mode=mode or "generate",
                    num_steps=num_steps,
                    retake_start=retake_start,
                    retake_end=retake_end,
                    extend_frames=extend_frames,
                    extend_direction=extend_direction or "after",
                    lora_specs=lora_specs,
                    video_conditioning_specs=video_conditioning_specs,
                    job_id=job_id,
                    a2v_visual_i2v_continue=a2v_visual_i2v_continue,
                    end_image_data=end_image_data,
                    enhance_prompt=enhance_prompt,
                    pipeline_profile=pipeline_profile,
                    cfg_scale=cfg_scale,
                    stg_scale=stg_scale,
                    stage2_steps=stage2_steps,
                    no_regen_audio=no_regen_audio,
                    reference_strength=reference_strength,
                    audio_start_seconds=audio_start_seconds,
                    skip_stage_2=bool(skip_stage_2),
                    upsample_only=bool(upsample_only),
                    modality_scale=modality_scale,
                ),
            ),
        )

    def _salvage_mp4_to_spill(
        self,
        tmpdir: str,
        preferred_out: str,
        job_id: str | None,
        prompt: str,
        tag: str,
    ) -> None:
        if not self.spill_dir or not job_id:
            return
        root = Path(tmpdir)
        src = Path(preferred_out)
        if not (src.is_file() and src.stat().st_size > 0):
            alt = _largest_mp4_under(root)
            if alt is None:
                log.warning(
                    "  ◆ no MP4 found to salvage under %s (job %s)",
                    tmpdir,
                    job_id[:8],
                )
                return
            src = alt
        try:
            self.spill_dir.mkdir(parents=True, exist_ok=True)
            slug = _spill_slug(prompt)
            dest = self.spill_dir / f"{job_id}_{slug}_{tag}.mp4"
            shutil.copy2(src, dest)
            log.info("  ◆ spill-salvaged (%s) → %s", tag, dest)
        except OSError as exc:
            log.error("  ✗ spill salvage failed: %s", exc)

    def _log_generation_e2e_grandtotal(
        self,
        *,
        status: str,
        req: GenerationRequest,
        t_e2e0: float,
        mode: str,
        height: int | str,
        width: int | str,
        frames: int | str,
        steps: int | str,
        seed: int | str,
        output_path: str | None = None,
        error: str | None = None,
    ) -> None:
        """Single greppable wall-clock line for every generation attempt."""
        e2e_s = time.time() - t_e2e0
        e2e_ms = int(round(e2e_s * 1000))
        out_kb: int | str = "-"
        if output_path and os.path.isfile(output_path):
            out_kb = os.path.getsize(output_path) // 1024
        err_bit = f" error={error!r}" if error else ""
        log.info(
            "Generation e2e grandtotal: status=%s job=%s mode=%s vae_decoder=%s "
            "seed=%s size=%sx%s frames=%s steps=%s e2e=%.3fs (%dms) output_kb=%s%s",
            status,
            (req.job_id[:8] if req.job_id else "-"),
            mode,
            get_vae_decoder_variant(),
            seed,
            height,
            width,
            frames,
            steps,
            e2e_s,
            e2e_ms,
            out_kb,
            err_bit,
        )

    def _generate_sync(self, req: GenerationRequest) -> str:
        t_e2e0 = time.time()
        ctx: dict[str, Any] = {
            "mode": (req.mode or "generate").strip().lower(),
            "height": "?",
            "width": "?",
            "frames": "?",
            "steps": "?",
            "seed": req.seed,
            "status": "failed",
            "output_path": None,
            "error": None,
        }
        try:
            result = self._generate_sync_timed(req, ctx=ctx)
            ctx["status"] = "ok"
            ctx["output_path"] = result
            return result
        except GenerationCancelledError as exc:
            ctx["status"] = "cancelled"
            ctx["error"] = str(exc)[:200]
            raise
        except BaseException as exc:
            ctx["status"] = "failed"
            ctx["error"] = str(exc)[:200]
            raise
        finally:
            self._log_generation_e2e_grandtotal(
                status=str(ctx["status"]),
                req=req,
                t_e2e0=t_e2e0,
                mode=str(ctx["mode"]),
                height=ctx["height"],
                width=ctx["width"],
                frames=ctx["frames"],
                steps=ctx["steps"],
                seed=ctx["seed"],
                output_path=ctx.get("output_path"),
                error=ctx.get("error"),
            )

    def _generate_sync_timed(self, req: GenerationRequest, *, ctx: dict[str, Any]) -> str:
        self._check_cancel()
        self.load()
        self._check_cancel()

        assert self._model_path is not None
        requested_height = int(req.height or self.height)
        requested_width = int(req.width or self.width)
        ah = _align_ltx2_spatial(requested_height)
        aw = _align_ltx2_spatial(requested_width)
        if ah != requested_height or aw != requested_width:
            log.warning(
                "LTX requires H×W divisible by %s; adjusted %s×%s → %s×%s",
                LTX2_SPATIAL_ALIGN,
                requested_height,
                requested_width,
                ah,
                aw,
            )
        height, width = ah, aw

        requested_num_frames = int(req.num_frames or self.num_frames)
        nf = _nearest_valid_frames(requested_num_frames)
        if nf != requested_num_frames:
            log.warning(
                "LTX requires (frames-1)%%8==0; adjusted frames %s → %s",
                requested_num_frames,
                nf,
            )
        mode = (req.mode or "generate").strip().lower()
        requested_steps = int(req.num_steps or self.inference_steps)
        steps = max(1, requested_steps)
        if steps != requested_steps:
            log.warning("LTX steps must be >=1; adjusted steps %s → %s", requested_steps, steps)
        requested_seed = int(req.seed)
        seed = requested_seed
        if seed < 0:
            # videofentanyl commonly sends -1 for "auto/random seed".
            seed = random.randint(0, 2**31 - 1)
            log.info("LTX random seed requested (%s); using generated seed %s", requested_seed, seed)
        ctx.update(
            mode=mode,
            height=height,
            width=width,
            frames=nf,
            steps=steps,
            seed=seed,
        )
        effective_loras: list[tuple[str, float]] = []
        if self._resolved_default_loras is not None:
            effective_loras.extend(self._resolved_default_loras)
        else:
            effective_loras.extend(self.default_lora_specs)
        effective_loras.extend(req.lora_specs or [])
        resolved_loras: list[tuple[str, float]] = []

        tmp_image: str | None = None
        tmp_end_image: str | None = None
        tmp_audio: str | None = None
        tmp_video: str | None = None
        tmp_video_conditioning_cleanup: list[str] = []
        tmp_lora_cleanup: list[str] = []
        prefix = f"fv_{req.job_id[:8]}_" if req.job_id else "fvserver_work_"
        tmpdir = str(mk_scratch_dir(prefix=prefix))
        out_path = os.path.join(tmpdir, "output.mp4")
        last_pipe: Any | None = None
        media_cleanups: list[str] = []

        try:
            tmp_image, tmp_image_cleanup = _decode_media_input(
                req.image_data,
                temp_prefix="fvserver_img_",
                default_suffix=".jpg",
            )
            if not tmp_image and isinstance(req.image_data, dict):
                tmp_image = _decode_initial_image_dict(req.image_data)
                tmp_image_cleanup = tmp_image
            tmp_audio, tmp_audio_cleanup = _decode_media_input(
                req.audio_data,
                temp_prefix="fvserver_audio_",
                default_suffix=".wav",
            )
            tmp_video, tmp_video_cleanup = _decode_media_input(
                req.source_video_data,
                temp_prefix="fvserver_video_",
                default_suffix=".mp4",
            )
            tmp_end_image, tmp_end_image_cleanup = _decode_media_input(
                req.end_image_data,
                temp_prefix="fvserver_end_img_",
                default_suffix=".jpg",
            )
            if not tmp_end_image and isinstance(req.end_image_data, dict):
                tmp_end_image = _decode_initial_image_dict(req.end_image_data)
                tmp_end_image_cleanup = tmp_end_image
            vc_items, vc_cleanup = _decode_weighted_media_inputs(
                req.video_conditioning_specs,
                temp_prefix="fvserver_vcond_",
                default_suffix=".mp4",
            )
            tmp_video_conditioning_cleanup = vc_cleanup
            for path, cleanup, marker in (
                (tmp_image, tmp_image_cleanup, "fvserver_img_"),
                (tmp_end_image, tmp_end_image_cleanup, "fvserver_end_img_"),
                (tmp_audio, tmp_audio_cleanup, "fvserver_audio_"),
                (tmp_video, tmp_video_cleanup, "fvserver_video_"),
            ):
                if cleanup:
                    media_cleanups.append(cleanup)
                elif path and marker in path:
                    media_cleanups.append(path)
            if mode in (
                "face_swap",
                "face-swap",
                "lipdub",
                "lip_dub",
                "ic_lora",
                "keyframe",
                "id_lora",
                "id-lora",
            ):
                # Exclusive adapter modes: never stack global OmniNFT defaults.
                # V2V maps to ic_lora — empty request = pure reference conditioning.
                # CrossView / HDR / Union / keyframe / ID-LoRA come only from the request.
                mode_loras = list(req.lora_specs or [])
                if mode in ("id_lora", "id-lora") and not mode_loras:
                    mode_loras = [(ID_LORA_DEFAULT_SPEC, ID_LORA_DEFAULT_SCALE)]
                    log.info("ID-LoRA: auto-injecting default CelebV-HQ adapter")
                for lora_spec, lora_scale in mode_loras:
                    lora_path, lora_cleanup = _resolve_lora_path(str(lora_spec))
                    resolved_loras.append((lora_path, float(lora_scale)))
                    if lora_cleanup:
                        tmp_lora_cleanup.append(lora_cleanup)
            elif self._resolved_default_loras is not None and not req.lora_specs:
                resolved_loras = list(self._resolved_default_loras)
            else:
                for lora_spec, lora_scale in effective_loras:
                    lora_path, lora_cleanup = _resolve_lora_path(str(lora_spec))
                    resolved_loras.append((lora_path, float(lora_scale)))
                    if lora_cleanup:
                        tmp_lora_cleanup.append(lora_cleanup)
            effective_prompt = _maybe_enhance_prompt(
                req.prompt,
                mode=mode,
                model_dir=str(self._model_path),
                enabled=bool(req.enhance_prompt) and mode not in ("extend", "retake"),
            )
            profile = _normalize_pipeline_profile(req.pipeline_profile)
            log.info(
                "Generation effective params: mode=%s profile=%s enhance=%s seed=%s (requested=%s) "
                "size=%sx%s frames=%s steps=%s fps=%s (requested size=%sx%s frames=%s steps=%s) "
                "image=%s end_image=%s audio=%s video=%s retake=%s-%s extend=%s/%s vcond=%s loras=%s "
                "id_lora=%s vae_decoder=%s model_path=%s",
                mode,
                profile if mode not in ("extend", "retake", "id_lora", "id-lora") else (
                    "id_lora_two_stage" if mode in ("id_lora", "id-lora") else "dev+CFG"
                ),
                "yes" if req.enhance_prompt else "no",
                seed,
                requested_seed,
                height,
                width,
                nf,
                steps,
                float(self.fps),
                requested_height,
                requested_width,
                requested_num_frames,
                requested_steps,
                "yes" if tmp_image else "no",
                "yes" if tmp_end_image else "no",
                "yes" if tmp_audio else "no",
                "yes" if tmp_video else "no",
                req.retake_start if req.retake_start is not None else "-",
                req.retake_end if req.retake_end is not None else "-",
                req.extend_frames if req.extend_frames is not None else "-",
                (req.extend_direction or "after").strip().lower(),
                len(vc_items),
                len(resolved_loras),
                "yes" if mode in ("id_lora", "id-lora") else "no",
                get_vae_decoder_variant(),
                self._model_path,
            )
            if resolved_loras:
                log.info(
                    "Applying %d LoRA(s) for mode=%s (request=%d, defaults=%d)",
                    len(resolved_loras),
                    mode,
                    len(req.lora_specs or []),
                    self.default_lora_count(),
                )

            self._check_cancel()
            try:
                with self._track_model_progress():
                    common_gen_kwargs = dict(
                        prompt=effective_prompt,
                        output_path=out_path,
                        height=height,
                        width=width,
                        num_frames=nf,
                        frame_rate=float(self.fps),
                        seed=seed,
                        num_steps=steps,
                        lora_paths=resolved_loras,
                    )
                    _apply_optional_generate_kwargs(common_gen_kwargs, req)
                    if mode == "a2v":
                        if not tmp_audio:
                            raise RuntimeError("a2v mode requires audio input")
                        visual_continue = bool(req.a2v_visual_i2v_continue and tmp_image)
                        steps = _a2v_effective_stage1_steps(
                            steps, visual_i2v_continue=visual_continue
                        )
                        common_gen_kwargs["num_steps"] = steps
                        _apply_ltx_mlx_patches(default_fps=self.fps)
                        from ltx_media import load_audio_for_inference

                        audio_probe = load_audio_for_inference(
                            tmp_audio,
                            target_sample_rate=16000,
                            start_time=float(req.audio_start_seconds or 0),
                            max_duration=0.25,
                        )
                        if audio_probe is None:
                            raise RuntimeError(
                                f"Could not decode audio for a2v (PyAV): {tmp_audio}"
                            )
                        video_duration_s = nf / float(self.fps)
                        if visual_continue:
                            silent_path = os.path.join(tmpdir, "output_silent.mp4")
                            # Same profile pipe as T2V/I2V (distilled by default), not
                            # A2V CFG clamp and not TI2VidOneStagePipeline.
                            pipe_key = self._resolve_generate_pipe_key(
                                profile, has_image=True
                            )
                            pipe = self._get_pipe(pipe_key)
                            last_pipe = pipe
                            log.info(
                                "A2V visual continue: I2V profile pipe key=%s (%s) "
                                "steps=%s image=yes + audio mux "
                                "(skips A2V CFG clamp / one_stage)",
                                pipe_key,
                                type(pipe).__name__,
                                steps,
                            )
                            _invoke_generate_and_save(
                                pipe,
                                **common_gen_kwargs,
                                output_path=silent_path,
                                image=tmp_image,
                            )
                            _mux_audio_into_video(
                                silent_path,
                                tmp_audio,
                                out_path,
                                duration_s=video_duration_s,
                            )
                        else:
                            pipe = self._get_pipe("a2v")
                            last_pipe = pipe
                            log.info(
                                "A2V native: pipe=a2v (%s) steps=%s image=%s "
                                "(start image is conditioning inside A2VidPipelineTwoStage; "
                                "not TI2VidOneStagePipeline)",
                                type(pipe).__name__,
                                steps,
                                "yes" if tmp_image else "no",
                            )
                            _invoke_generate_and_save(
                                pipe,
                                **common_gen_kwargs,
                                audio_path=tmp_audio,
                                image=tmp_image,
                            )
                    elif mode == "retake":
                        if not tmp_video:
                            raise RuntimeError("retake mode requires source video input")
                        start_frame = int(req.retake_start if req.retake_start is not None else 0)
                        end_frame = int(
                            req.retake_end if req.retake_end is not None else start_frame + 1
                        )
                        if end_frame <= start_frame:
                            raise RuntimeError(
                                f"retake requires end_frame > start_frame "
                                f"(got start={start_frame}, end={end_frame}; end is exclusive)"
                            )
                        pipe = self._get_pipe("retake")
                        last_pipe = pipe
                        retake_steps = steps
                        retake_cfg = float(
                            req.cfg_scale
                            if req.cfg_scale is not None
                            else RETAKE_EXTEND_DEFAULT_CFG
                        )
                        retake_stg = float(
                            req.stg_scale
                            if req.stg_scale is not None
                            else RETAKE_EXTEND_DEFAULT_STG
                        )
                        retake_kwargs = dict(
                            prompt=effective_prompt,
                            output_path=out_path,
                            video_path=tmp_video,
                            start_frame=start_frame,
                            end_frame=end_frame,
                            seed=seed,
                            num_steps=retake_steps,
                            cfg_scale=retake_cfg,
                            stg_scale=retake_stg,
                            lora_paths=resolved_loras,
                            fps=float(self.fps),
                        )
                        _apply_optional_generate_kwargs(retake_kwargs, req)
                        if not callable(getattr(pipe, "retake_from_video", None)):
                            raise RuntimeError(
                                f"{type(pipe).__name__} has no retake_from_video(); "
                                "update ltx-2-mlx"
                            )
                        log.info(
                            "Retake via retake_from_video (frames %s-%s exclusive end, "
                            "steps=%s, cfg=%.1f, stg=%.1f, regenerate_audio=%s)",
                            start_frame,
                            end_frame,
                            retake_steps,
                            retake_cfg,
                            retake_stg,
                            retake_kwargs.get("regenerate_audio", True),
                        )
                        _invoke_retake_and_save(
                            pipe,
                            default_fps=float(self.fps),
                            **retake_kwargs,
                        )
                    elif mode == "extend":
                        if not tmp_video:
                            raise RuntimeError("extend mode requires source video input")
                        ext_frames = int(
                            req.extend_frames
                            if req.extend_frames is not None
                            else DEFAULT_EXTEND_LATENT_FRAMES
                        )
                        direction = (req.extend_direction or "after").strip().lower()
                        pipe = self._get_pipe("extend")
                        last_pipe = pipe
                        extend_steps = steps
                        extend_cfg = float(
                            req.cfg_scale
                            if req.cfg_scale is not None
                            else RETAKE_EXTEND_DEFAULT_CFG
                        )
                        extend_stg = float(
                            req.stg_scale
                            if req.stg_scale is not None
                            else RETAKE_EXTEND_DEFAULT_STG
                        )
                        extend_kwargs = dict(
                            prompt=effective_prompt,
                            output_path=out_path,
                            video_path=tmp_video,
                            extend_frames=ext_frames,
                            direction=direction,
                            seed=seed,
                            num_steps=extend_steps,
                            cfg_scale=extend_cfg,
                            stg_scale=extend_stg,
                            lora_paths=resolved_loras,
                            fps=float(self.fps),
                        )
                        _apply_optional_generate_kwargs(extend_kwargs, req)
                        if not callable(getattr(pipe, "extend_from_video", None)):
                            raise RuntimeError(
                                f"{type(pipe).__name__} has no extend_from_video(); "
                                "update ltx-2-mlx"
                            )
                        log.info(
                            "Extend via extend_from_video "
                            "(latent_frames=%s ≈ ~%d pixel frames, direction=%s, "
                            "steps=%s, cfg=%.1f, stg=%.1f)",
                            ext_frames,
                            ext_frames * 8,
                            direction,
                            extend_steps,
                            extend_cfg,
                            extend_stg,
                        )
                        _invoke_extend_and_save(
                            pipe,
                            default_fps=float(self.fps),
                            **extend_kwargs,
                        )
                        try:
                            from videofentanyl import count_video_frames

                            src_frames = count_video_frames(tmp_video) if tmp_video else None
                            out_frames = count_video_frames(out_path)
                            if src_frames is not None and out_frames is not None:
                                log.info(
                                    "Extend output: %d frames (source %d, +%d, ~%.2fs @ %.1f fps)",
                                    out_frames,
                                    src_frames,
                                    out_frames - src_frames,
                                    max(0.0, (out_frames - 1) / float(self.fps)),
                                    float(self.fps),
                                )
                                if out_frames <= src_frames:
                                    log.warning(
                                        "Extend did not lengthen the video — verify duration "
                                        "(5s = 121 frames) and extend_frames=%s",
                                        ext_frames,
                                    )
                        except Exception:
                            pass
                    elif mode == "keyframe":
                        if not tmp_image or not tmp_end_image:
                            raise RuntimeError("keyframe mode requires start and end images")
                        pipe = self._get_pipe(
                            "keyframe",
                            pipe_kwargs=_keyframe_pipe_kwargs(self._model_path),
                        )
                        last_pipe = pipe
                        last_idx = max(0, int(nf) - 1)
                        kf_cfg = float(
                            req.cfg_scale if req.cfg_scale is not None else KEYFRAME_DEFAULT_CFG
                        )
                        kf_kwargs = dict(common_gen_kwargs)
                        kf_kwargs.pop("image", None)
                        kf_kwargs.pop("end_image", None)
                        kf_kwargs["cfg_scale"] = kf_cfg
                        log.info(
                            "Keyframe interpolation: images=2 → indices [0, %d], cfg=%.1f",
                            last_idx,
                            kf_cfg,
                        )
                        _invoke_generate_and_save(
                            pipe,
                            **kf_kwargs,
                            keyframe_images=[tmp_image, tmp_end_image],
                            keyframe_indices=[0, last_idx],
                        )
                    elif mode in ("lipdub", "lip_dub"):
                        if not tmp_video:
                            raise RuntimeError("lipdub mode requires reference video")
                        if len(resolved_loras) != 1:
                            raise RuntimeError("lipdub mode requires exactly one LoRA spec")
                        pipe = self._get_pipe(
                            "lipdub",
                            pipe_kwargs={
                                "lora_paths": [(str(p), float(s)) for p, s in resolved_loras],
                            },
                        )
                        last_pipe = pipe
                        lipdub_ref, lipdub_tmp = _prepare_lipdub_reference_video(
                            tmp_video,
                            tmp_audio,
                            tmpdir=tmpdir,
                        )
                        tmp_video_conditioning_cleanup.extend(lipdub_tmp)
                        _invoke_lipdub_style(
                            pipe,
                            common_gen_kwargs=common_gen_kwargs,
                            reference_video=lipdub_ref,
                            tmp_image=tmp_image,
                            num_frames=nf,
                            req=req,
                        )
                    elif mode in ("id_lora", "id-lora"):
                        if not tmp_image:
                            raise RuntimeError("id_lora mode requires a first-frame reference image")
                        if not tmp_audio:
                            raise RuntimeError("id_lora mode requires reference audio")
                        if len(resolved_loras) != 1:
                            raise RuntimeError("id_lora mode requires exactly one ID-LoRA spec")
                        last_pipe = _run_id_lora_generation(
                            self,
                            req=req,
                            prompt=effective_prompt,
                            resolved_loras=resolved_loras,
                            image_path=tmp_image,
                            audio_path=tmp_audio,
                            out_path=out_path,
                            nf=nf,
                            seed=seed,
                            steps=steps,
                            requested_height=height,
                            requested_width=width,
                        )
                    elif mode in ("face_swap", "face-swap"):
                        if not tmp_video:
                            raise RuntimeError("face_swap mode requires reference video")
                        if not tmp_image:
                            raise RuntimeError("face_swap mode requires face identity image")
                        if len(resolved_loras) != 1:
                            raise RuntimeError("face_swap mode requires exactly one LoRA spec")
                        from ltx_face_swap_compose import (
                            crop_face_swap_output_to_main_video,
                            format_head_swap_prompt,
                        )

                        ref_path = tmp_video
                        guide_path, guide_layout, face_swap_nf, canvas_w, canvas_h = (
                            _prepare_face_swap_guide_video(
                                str(ref_path),
                                tmp_image,
                                tmpdir=tmpdir,
                                num_frames=nf,
                                width=width,
                                height=height,
                                fps=float(self.fps),
                            )
                        )
                        tmp_video_conditioning_cleanup.extend(
                            [
                                os.path.join(tmpdir, "face_swap_ref_trimmed.mp4"),
                                os.path.join(tmpdir, "face_swap_bfs_v3_guide.mp4"),
                                guide_path,
                            ]
                        )
                        face_swap_prompt = format_head_swap_prompt(effective_prompt)
                        log.info(
                            "Face swap: BFS V3 composite %dx%d (%d frames) "
                            "main panel %dx%d + LTXVAddGuide full composite",
                            canvas_w,
                            canvas_h,
                            face_swap_nf,
                            guide_layout.video_w,
                            guide_layout.video_h,
                        )
                        if self.spill_dir and req.job_id:
                            try:
                                self.spill_dir.mkdir(parents=True, exist_ok=True)
                                slug = _spill_slug(req.prompt)
                                dest = self.spill_dir / f"{req.job_id}_{slug}_face_swap_guide.mp4"
                                shutil.copy2(guide_path, dest)
                                log.info("Face swap guide saved → %s", dest)
                            except OSError as exc:
                                log.warning("Could not save face swap guide debug copy: %s", exc)
                        last_pipe = _run_face_swap_generation(
                            self,
                            req=req,
                            prompt=face_swap_prompt,
                            resolved_loras=resolved_loras,
                            guide_path=guide_path,
                            tmpdir=tmpdir,
                            out_path=out_path,
                            width=canvas_w,
                            height=canvas_h,
                            nf=face_swap_nf,
                            seed=seed,
                            steps=steps,
                        )
                        try:
                            crop_face_swap_output_to_main_video(out_path, guide_layout)
                            log.info(
                                "Face swap: cropped output to main video area %dx%d",
                                guide_layout.video_w,
                                guide_layout.video_h,
                            )
                        except Exception as exc:
                            log.warning("Face swap output crop failed (using full frame): %s", exc)
                        _maybe_preserve_reference_audio(
                            out_path,
                            [tmp_video],
                            job_id=req.job_id,
                        )
                    elif mode == "ic_lora":
                        if not vc_items and not (
                            resolved_loras and _ic_lora_uses_hdr_pipeline(resolved_loras)
                        ):
                            raise RuntimeError(
                                "ic_lora / v2v requires a reference video "
                                "(or an HDR LoRA for pure text-to-HDR)"
                            )
                        last_pipe = _run_ic_lora_generation(
                            self,
                            req=req,
                            prompt=req.prompt,
                            resolved_loras=resolved_loras,
                            vc_items=vc_items,
                            tmp_image=tmp_image,
                            tmpdir=tmpdir,
                            out_path=out_path,
                            width=width,
                            height=height,
                            nf=nf,
                            seed=seed,
                            steps=steps,
                            tmp_video_conditioning_cleanup=tmp_video_conditioning_cleanup,
                            audio_reference_paths=[str(p) for p, _ in vc_items],
                        )
                    elif tmp_image:
                        try:
                            from PIL import Image as PILImage

                            with PILImage.open(tmp_image) as im:
                                log.info(
                                    "I2V conditioning image: %s (%dx%d) → generation %dx%d",
                                    tmp_image,
                                    im.size[0],
                                    im.size[1],
                                    width,
                                    height,
                                )
                        except Exception:
                            log.info(
                                "I2V conditioning image: %s → generation %dx%d",
                                tmp_image,
                                width,
                                height,
                            )
                        # I2V: same generate pipeline as T2V for this profile; pass image=.
                        pipe_key = self._resolve_generate_pipe_key(profile, has_image=True)
                        pipe = self._get_pipe(pipe_key)
                        last_pipe = pipe
                        log.info(
                            "I2V using pipeline key=%s (%s) with image conditioning",
                            pipe_key,
                            type(pipe).__name__,
                        )
                        _invoke_generate_and_save(
                            pipe,
                            **common_gen_kwargs,
                            image=tmp_image,
                        )
                    else:
                        pipe_key = self._resolve_generate_pipe_key(profile, has_image=False)
                        pipe = self._get_pipe(pipe_key)
                        last_pipe = pipe
                        if (
                            profile == PIPE_PROFILE_DISTILLED
                            and self.upscale
                            and "spatial_upscaler" in self._pipe_classes
                            and pipe_key == "t2v"
                        ):
                            base_h, base_w = self._calculate_stage1_dimensions(height, width)
                            lowres_out_path = os.path.join(tmpdir, "output_lowres.mp4")
                            log.info(
                                "Legacy two-stage upscale enabled: stage1=%sx%s -> stage2=%sx%s",
                                base_h,
                                base_w,
                                height,
                                width,
                            )
                            _invoke_generate_and_save(
                                pipe,
                                prompt=req.prompt,
                                output_path=lowres_out_path,
                                height=base_h,
                                width=base_w,
                                num_frames=nf,
                                fps=float(self.fps),
                                seed=seed,
                                num_steps=steps,
                                lora_paths=resolved_loras,
                            )
                            upscaled = self._run_spatial_upscaler_stage(
                                prompt=req.prompt,
                                source_video_path=lowres_out_path,
                                output_path=out_path,
                                height=height,
                                width=width,
                                num_frames=nf,
                                seed=seed,
                                num_steps=steps,
                                lora_paths=resolved_loras,
                            )
                            if not upscaled:
                                shutil.copy2(lowres_out_path, out_path)
                        else:
                            _invoke_generate_and_save(
                                pipe,
                                **common_gen_kwargs,
                            )
            except BaseException as exc:
                if not isinstance(exc, GenerationCancelledError):
                    log.exception(
                        "Generation failed (job %s, mode=%s): %s",
                        req.job_id[:8] if req.job_id else "?",
                        mode,
                        exc,
                    )
                    self._salvage_mp4_to_spill(
                        tmpdir, out_path, req.job_id, req.prompt, "ENCODE_FAIL"
                    )
                raise

            video_path = out_path
            if not os.path.exists(video_path):
                self._salvage_mp4_to_spill(
                    tmpdir, out_path, req.job_id, req.prompt, "MISSING_OUTPUT",
                )
                raise RuntimeError(
                    f"Generation completed but output file not found: {video_path}"
                )
            return _export_output_mp4(video_path)

        finally:
            if last_pipe is not None:
                self.cleanup_after_generation(last_pipe)
            else:
                self.cleanup_after_generation()
            for tmp in media_cleanups:
                _unlink_fvserver_temp(tmp, "fvserver_")
            for tmp in tmp_video_conditioning_cleanup:
                _unlink_fvserver_temp(tmp, "fvserver_vcond_")
            for tmp in tmp_lora_cleanup:
                _unlink_fvserver_temp(tmp, "fvserver_lora_")
            shutil.rmtree(tmpdir, ignore_errors=True)


# Patch before any ltx-pipelines import binds ffmpeg media helpers.
_apply_ltx_mlx_patches()
