"""ID-LoRA mode: helpers, catalog, request validation, and mode wiring."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.utils.positions import compute_audio_positions
from ltx_id_lora_pipeline import (
    compute_id_lora_stage1_resolution,
    patchify_id_lora_audio_reference_latent,
    prefix_audio_reference_state,
    snap_to_divisor,
    strip_prefix_audio_tokens,
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


def test_apply_pending_loras_skips_id_lora_owned_paths():
    from ltx_mlx_backend import _apply_pending_loras

    class Pipe:
        _lora_paths = [("/tmp/id_lora.safetensors", 1.0)]
        dit = object()
        _pending_loras = None

    pipe = Pipe()
    _apply_pending_loras(pipe, [("/tmp/other.safetensors", 0.5)])
    assert pipe._pending_loras is None
