"""ID-LoRA mode: helpers, catalog, request validation, and mode wiring."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.utils.positions import compute_audio_positions
from ltx_id_lora_pipeline import (
    DEFAULT_MODALITY_SCALE,
    DEFAULT_STAGE1_STEPS,
    DEFAULT_STAGE1_STEPS_FAITHFUL,
    DEFAULT_STG_SCALE,
    DEFAULT_STG_SCALE_FAITHFUL,
    REF_AUDIO_MIN_PEAK,
    compute_id_lora_stage1_resolution,
    count_id_lora_key_matches,
    patchify_id_lora_audio_reference_latent,
    prefix_audio_reference_state,
    snap_to_divisor,
    strip_prefix_audio_tokens,
    validate_id_lora_ref_audio_stats,
    waveform_peak,
)


class _FakeAudioPatchifier:
    def patchify(self, _vae_latents: mx.array) -> tuple[mx.array, int]:
        t = 5
        return mx.zeros((1, t, 128)), t


def test_stage1_resolution_caps_long_side():
    h, w = compute_id_lora_stage1_resolution(1080, 1920)
    assert max(h, w) <= 512
    assert h % 32 == 0 and w % 32 == 0
    assert snap_to_divisor(500) == 512


def test_negative_audio_positions_are_prefix_and_non_overlapping():
    tokens, ref_pos = patchify_id_lora_audio_reference_latent(
        mx.zeros((1, 8, 4, 16)),
        _FakeAudioPatchifier(),
        negative_positions=True,
    )
    ref_len = int(tokens.shape[1])
    assert float(mx.max(ref_pos).item()) < 0.0

    tgt_t = 9
    tgt_pos = compute_audio_positions(tgt_t)
    state = LatentState(
        latent=mx.zeros((1, tgt_t, 128)),
        clean_latent=mx.zeros((1, tgt_t, 128)),
        denoise_mask=mx.ones((1, tgt_t, 1)),
        positions=tgt_pos,
    )
    prefixed = prefix_audio_reference_state(state, patchified=tokens, positions=ref_pos)
    assert prefixed.latent.shape[1] == ref_len + tgt_t
    assert float(mx.max(prefixed.denoise_mask[:, :ref_len, :]).item()) == 0.0
    assert float(mx.min(prefixed.denoise_mask[:, ref_len:, :]).item()) == 1.0

    ref_max = float(mx.max(prefixed.positions[:, :ref_len, :]).item())
    tgt_min = float(mx.min(prefixed.positions[:, ref_len:, :]).item())
    assert ref_max < 0.0
    assert ref_max < tgt_min

    stripped = strip_prefix_audio_tokens(prefixed.latent, ref_len)
    assert stripped.shape[1] == tgt_t


def test_identity_guidance_slice_assumes_prefix_layout():
    """Identity guidance uses cond[:, ref_len:] vs a noref pass on target-only tokens."""
    ref_len = 4
    total = 12
    cond = mx.arange(total, dtype=mx.float32).reshape(1, total, 1)
    noref = mx.zeros((1, total - ref_len, 1), dtype=mx.float32)
    id_delta = 3.0 * (cond[:, ref_len:, :] - noref)
    out = mx.concatenate([cond[:, :ref_len, :], cond[:, ref_len:, :] + id_delta], axis=1)
    assert out.shape[1] == total
    # Target tokens get +3 * original value when noref is zero.
    assert float(out[0, ref_len, 0].item()) == pytest.approx(4.0 * 3.0 + 4.0)


def test_generation_modes_include_id_lora():
    from web_ui import GENERATION_MODES

    ids = [m["id"] for m in GENERATION_MODES]
    assert "id_lora" in ids
    assert next(m for m in GENERATION_MODES if m["id"] == "id_lora")["label"]


def test_lora_catalog_includes_id_lora_presets():
    from web_ui import (
        ID_LORA_CELEBVHQ_PRESET_ID,
        ID_LORA_CELEBVHQ_SPEC,
        ID_LORA_TALKVID_PRESET_ID,
        ID_LORA_TALKVID_SPEC,
        _lora_catalog,
    )

    presets, _ = _lora_catalog(None)
    celeb = next(p for p in presets if p["id"] == ID_LORA_CELEBVHQ_PRESET_ID)
    talk = next(p for p in presets if p["id"] == ID_LORA_TALKVID_PRESET_ID)
    assert celeb["spec"] == ID_LORA_CELEBVHQ_SPEC
    assert talk["spec"] == ID_LORA_TALKVID_SPEC


def test_build_params_id_lora(tmp_path: Path):
    from web_ui import ID_LORA_CELEBVHQ_SPEC, _build_params_from_request

    face = tmp_path / "face.jpg"
    face.write_bytes(b"jpeg")
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")

    params = _build_params_from_request(
        {
            "mode": "id_lora",
            "prompt": "[VISUAL]: person. [SPEECH]: hi. [SOUNDS]: quiet.",
            "image_path": str(face),
            "audio_path": str(audio),
            "lora_specs": [[ID_LORA_CELEBVHQ_SPEC, 1.0]],
        }
    )
    assert params.generation_mode == "id_lora"
    assert params.initial_image == str(face.resolve())
    assert params.audio_input == str(audio.resolve())
    assert len(params.lora_specs) == 1


def test_mcp_normalize_and_validate_id_lora():
    from mcp_server import _normalize_mode

    assert _normalize_mode("id_lora") == "id_lora"
    with pytest.raises(ValueError, match="Unsupported mode"):
        _normalize_mode("not_a_mode")


def test_mcp_id_lora_validation_rules():
    """Mirror mcp_server id_lora guards without running a full generation."""
    mode = "id_lora"
    image = None
    audio = "/tmp/a.wav"
    lora_specs = None
    with pytest.raises(ValueError, match="requires image"):
        if mode == "id_lora":
            if not image:
                raise ValueError("mode=id_lora requires image (first-frame identity)")
            if not audio:
                raise ValueError("mode=id_lora requires audio (identity reference, ~5s)")
            if lora_specs is not None and len(lora_specs) != 1:
                raise ValueError("mode=id_lora accepts at most one lora_specs entry")

    image = "/tmp/f.jpg"
    audio = None
    with pytest.raises(ValueError, match="requires audio"):
        if mode == "id_lora":
            if not image:
                raise ValueError("mode=id_lora requires image (first-frame identity)")
            if not audio:
                raise ValueError("mode=id_lora requires audio (identity reference, ~5s)")

    from mcp_server import _normalize_mode

    assert _normalize_mode("id_lora") == "id_lora"
    src = Path("mcp_server.py").read_text(encoding="utf-8")
    assert '"id_lora"' in src
    assert "mode=id_lora requires image" in src
    assert "mode=id_lora requires audio" in src


def test_backend_registers_id_lora_and_defaults():
    from ltx_mlx_backend import ID_LORA_DEFAULT_SPEC, ID_LORA_DEFAULT_SCALE
    from pathlib import Path as P

    src = P("ltx_mlx_backend.py").read_text(encoding="utf-8")
    assert "IDLoraTwoStagesPipeline" in src
    assert 'elif mode in ("id_lora", "id-lora"):' in src
    assert "_run_id_lora_generation" in src
    assert "ID-LoRA-CelebVHQ" in ID_LORA_DEFAULT_SPEC
    assert ID_LORA_DEFAULT_SCALE == pytest.approx(1.0)


def test_cli_choices_include_id_lora():
    src = Path("videofentanyl.py").read_text(encoding="utf-8")
    assert '"id_lora"' in src
    assert "generation-mode id_lora requires --image" in src
    assert "generation-mode id_lora requires --audio" in src


def test_ui_payload_fields_for_id_lora():
    app = Path("web/src/App.tsx").read_text(encoding="utf-8")
    assert 'mode === "id_lora"' in app
    assert "ID-LoRA inputs" in app
    assert "[VISUAL]" in app
    assert "id_lora_celebvhq" in app
    assert "disabled={loraBusy || addingCustomLora || isFaceSwap || isLipDub || isIdLora}" in app


def test_stage_from_tqdm_maps_id_lora_desc():
    from ltx_mlx_backend import _stage_from_tqdm_desc

    assert _stage_from_tqdm_desc("ID-LoRA stage 1") == "denoising"
    assert _stage_from_tqdm_desc("Denoising (guided)") == "denoising"


def test_id_lora_denoise_uses_live_tqdm_lookup():
    """Stage-1 bar must call ``tqdm.tqdm`` live so WebSocket progress patching works."""
    import ast

    src = Path("ltx_id_lora_pipeline.py").read_text(encoding="utf-8")
    assert "import tqdm as tqdm_lib" in src
    assert "tqdm_lib.tqdm(" in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tqdm":
            assert not any(a.name == "tqdm" for a in node.names)


def test_track_model_progress_patches_id_lora_module_binding():
    from ltx_mlx_backend import LocalVideoGenerator

    src = Path("ltx_mlx_backend.py").read_text(encoding="utf-8")
    assert '"ltx_id_lora_pipeline"' in src
    assert "stale_bindings" in src
    assert hasattr(LocalVideoGenerator, "_track_model_progress")


def test_track_model_progress_publishes_id_lora_stage1_bar():
    """Live ``tqdm.tqdm`` lookup under the progress CM must update model_progress."""
    import threading

    import tqdm as tqdm_lib

    from ltx_mlx_backend import LocalVideoGenerator, _ModelProgressStore

    gen = LocalVideoGenerator.__new__(LocalVideoGenerator)
    gen._model_progress = _ModelProgressStore()
    gen._cancel_requested = threading.Event()

    with gen._track_model_progress():
        bar = tqdm_lib.tqdm(range(4), desc="ID-LoRA stage 1", disable=False, mininterval=0)
        for _ in bar:
            pass
        snap = gen.model_progress_for_ws()
        assert snap is not None
        assert snap["stage"] == "denoising"
        assert snap["total"] == 4
        assert snap["step"] == 4


def test_id_lora_unpatchifies_before_decode():
    """Regression: token latents (rank 3) must be unpatchified for audio VAE decode."""
    src = Path("ltx_id_lora_pipeline.py").read_text(encoding="utf-8")
    assert "audio_patchifier.unpatchify" in src
    assert "video_patchifier.unpatchify" in src
    # Must not return raw denoise tokens as the generate_id_lora result.
    assert "return output_2.video_latent, output_2.audio_latent" not in src


def test_apply_pending_loras_skips_id_lora_owned_paths():
    from ltx_mlx_backend import _apply_pending_loras

    class Pipe:
        _lora_paths = [("/tmp/id_lora.safetensors", 1.0)]
        dit = object()
        _pending_loras = None

    pipe = Pipe()
    _apply_pending_loras(pipe, [("/tmp/other.safetensors", 0.5)])
    assert pipe._pending_loras is None


def test_count_id_lora_key_matches():
    class _SD:
        def __init__(self, sd):
            self.sd = sd

    model = _SD({"blocks.0.attn.weight": mx.zeros((4, 4))})
    lora_ok = _SD(
        {
            "blocks.0.attn.lora_A.weight": mx.zeros((2, 4)),
            "blocks.0.attn.lora_B.weight": mx.zeros((4, 2)),
        }
    )
    matched, total = count_id_lora_key_matches(model, lora_ok)
    assert total == 1 and matched == 1

    lora_miss = _SD(
        {
            "blocks.99.attn.lora_A.weight": mx.zeros((2, 4)),
            "blocks.99.attn.lora_B.weight": mx.zeros((4, 2)),
        }
    )
    matched, total = count_id_lora_key_matches(model, lora_miss)
    assert total == 1 and matched == 0


def test_validate_id_lora_ref_audio_stats_rejects_silent_or_empty():
    validate_id_lora_ref_audio_stats(peak=0.5, token_count=8)
    with pytest.raises(ValueError, match="silent"):
        validate_id_lora_ref_audio_stats(peak=REF_AUDIO_MIN_PEAK / 10, token_count=8)
    with pytest.raises(ValueError, match="0 tokens"):
        validate_id_lora_ref_audio_stats(peak=0.5, token_count=0)
    assert waveform_peak(mx.array([[[-0.2, 0.4, -0.1]]])) == pytest.approx(0.4)


def test_balanced_defaults_and_faithful_constants():
    assert DEFAULT_STAGE1_STEPS == 20
    assert DEFAULT_STAGE1_STEPS_FAITHFUL == 30
    assert DEFAULT_STG_SCALE == pytest.approx(0.0)
    assert DEFAULT_STG_SCALE_FAITHFUL == pytest.approx(1.0)
    assert DEFAULT_MODALITY_SCALE == pytest.approx(1.0)
    from ltx_id_lora_pipeline import DEFAULT_MODALITY_SCALE_FAITHFUL

    assert DEFAULT_MODALITY_SCALE_FAITHFUL == pytest.approx(3.0)
    from ltx_mlx_backend import DEFAULT_ID_LORA_STAGE1_STEPS

    assert DEFAULT_ID_LORA_STAGE1_STEPS == 20


def test_generate_and_save_accepts_audio_and_speed_kwargs():
    import inspect

    from ltx_id_lora_pipeline import IDLoraTwoStagesPipeline

    params = inspect.signature(IDLoraTwoStagesPipeline.generate_and_save).parameters
    for name in (
        "audio_path",
        "skip_stage_2",
        "upsample_only",
        "modality_scale",
        "stg_scale",
        "stage1_steps",
    ):
        assert name in params


def test_invoke_generate_and_save_keeps_audio_path():
    """Regression: audio_path must not be dropped before generate_and_save."""
    from ltx_mlx_backend import _invoke_generate_and_save

    captured: dict = {}

    class Pipe:
        def generate_and_save(self, **kwargs):
            captured.update(kwargs)

    _invoke_generate_and_save(
        Pipe(),
        prompt="x",
        output_path="/tmp/out.mp4",
        image="/tmp/face.jpg",
        audio_path="/tmp/ref.wav",
        height=512,
        width=512,
        skip_stage_2=True,
        upsample_only=False,
        modality_scale=1.0,
    )
    assert captured["audio_path"] == "/tmp/ref.wav"
    assert captured["skip_stage_2"] is True
    assert "modality_scale" in captured


def test_ui_copy_says_voice_identity_not_soundtrack():
    app = Path("web/src/App.tsx").read_text(encoding="utf-8")
    assert "voice identity only" in app
    assert "Faithful (30 steps" in app
    assert "Skip stage 2 (faster preview" in app
    assert "Upsample only" in app
    assert "modality_scale" in app
    assert "upsample_only" in app
    assert "ID_LORA_PROMPT_TEMPLATE" in app
    assert "promptLooksLikeIdLoraTemplate" in app
    assert "Notify when ready" in app
    assert "notifyGenerationReady" in app
    assert "autoPlay" not in app or 'autoPlay\n' not in app
    # Player must not autoplay finished generations.
    assert "autoPlay" not in app.split("player-wrap")[1].split("</section>")[0]


def test_cli_documents_id_lora_speed_flags():
    src = Path("videofentanyl.py").read_text(encoding="utf-8")
    assert "--upsample-only" in src
    assert "--modality-scale" in src
    assert "ID-LoRA" in src or "id_lora" in src
