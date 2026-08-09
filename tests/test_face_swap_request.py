"""Face swap mode request wiring."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_lora_catalog_includes_face_swap_preset():
    from web_ui import FACE_SWAP_DEFAULT_SPEC, FACE_SWAP_PRESET_ID, _lora_catalog

    presets, _ = _lora_catalog(None)
    match = next(p for p in presets if p["id"] == FACE_SWAP_PRESET_ID)
    assert match["spec"] == FACE_SWAP_DEFAULT_SPEC
    assert match["scale"] == pytest.approx(0.98)


def test_build_params_face_swap(tmp_path: Path):
    from web_ui import _build_params_from_request

    face = tmp_path / "face.jpg"
    face.write_bytes(b"jpeg")
    video = tmp_path / "ref.mp4"
    video.write_bytes(b"mp4")

    params = _build_params_from_request(
        {
            "mode": "face_swap",
            "prompt": "person speaking to camera",
            "image_path": str(face),
            "video_path": str(video),
            "lora_specs": [[
                "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video/"
                "resolve/main/ltx-2.3/head_swap_v3_rank_adaptive_fro_098.safetensors",
                0.98,
            ]],
        }
    )
    assert params.generation_mode == "face_swap"
    assert params.initial_image == str(face.resolve())
    assert params.source_video == str(video.resolve())
    assert len(params.lora_specs) == 1


def test_face_swap_ui_payload_fields_are_mapped():
    """UI contract: face → image_path, performance → video_path/source_clip_id."""
    from pathlib import Path

    app = Path("web/src/App.tsx").read_text(encoding="utf-8")
    assert 'mode === "face_swap"' in app
    assert "Face identity image" in app
    assert "Reference video (required)" in app
    # Face swap video lives in the dedicated panel (not shared retake/extend upload).
    assert 'mode === "retake" || mode === "extend" || mode === "lipdub"' in app
    assert 'mode === "face_swap" && selectedLoras.length !== 1' in app
    assert "body.image_path = imagePath" in app
    assert "body.video_path = videoPath" in app
    assert "body.source_clip_id = sourceClipId" in app
    assert "disabled={loraBusy || addingCustomLora || isFaceSwap || isLipDub || isIdLora}" in app


def test_face_swap_mode_uses_bfs_face_swap_pipeline():
    """Face swap uses BFS composite + FaceSwapPipeline (stage-2 LoRA + ref), not LipDub."""
    from pathlib import Path

    src = Path("ltx_mlx_backend.py").read_text(encoding="utf-8")
    face_block_start = src.index('elif mode in ("face_swap", "face-swap"):')
    face_block_end = src.index("elif mode == \"ic_lora\":", face_block_start)
    face_block = src[face_block_start:face_block_end]
    assert '"lipdub"' not in face_block
    assert "_invoke_lipdub_style" not in face_block
    assert "_run_face_swap_generation" in face_block
    assert "_run_ic_lora_generation" not in face_block
    assert "FaceSwapPipeline" in src or "face_swap" in src
    assert "compose_bfs_v3_guide_video" in src
