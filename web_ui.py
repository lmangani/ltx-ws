"""
web_ui.py — WebUI API, static assets, and generation orchestration for ltx-ws.

Used by server.py (--web-ui) and optionally by web_server.py (standalone).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

from starlette.requests import Request
from starlette.websockets import WebSocket

from ltx_media import media_available, probe_audio_duration, trim_audio_to_temp

REPO_ROOT = Path(__file__).resolve().parent
log = logging.getLogger("web_ui")

KNOWN_MODELS = [
    {"id": "auto", "label": "Auto (RAM-based)", "repo": "auto"},
    {"id": "dgrauet/ltx-2.3-mlx", "label": "MLX bf16 (full quality)", "repo": "dgrauet/ltx-2.3-mlx"},
    {"id": "dgrauet/ltx-2.3-mlx-q8", "label": "MLX int8 (balanced)", "repo": "dgrauet/ltx-2.3-mlx-q8"},
    {"id": "dgrauet/ltx-2.3-mlx-q4", "label": "MLX int4 (smallest)", "repo": "dgrauet/ltx-2.3-mlx-q4"},
]

RESOLUTION_PRESETS = [
    {"id": "704x480", "width": 704, "height": 480, "label": "704 × 480"},
    {"id": "1024x576", "width": 1024, "height": 576, "label": "1024 × 576 (16:9)"},
    {"id": "576x1024", "width": 576, "height": 1024, "label": "576 × 1024 (9:16)"},
    {"id": "1280x720", "width": 1280, "height": 720, "label": "1280 × 720 (HD)"},
]

GENERATION_MODES = [
    {"id": "generate", "label": "Text to video"},
    {"id": "i2v", "label": "Image to video (i2v)"},
    {"id": "a2v", "label": "Audio to video (a2v)"},
    {"id": "v2v", "label": "V2V (reference ± optional LoRA)"},
    {"id": "retake", "label": "Retake (edit region)"},
    {"id": "extend", "label": "Extend video"},
    {"id": "keyframe", "label": "Keyframe interpolation"},
    {"id": "lipdub", "label": "LipDub (experimental)"},
    {"id": "ic_lora", "label": "IC-LoRA (HDR / Union Control)"},
    {"id": "face_swap", "label": "Face swap (LTX 2.3)"},
    {"id": "id_lora", "label": "ID-LoRA (face + voice)"},
]

CHAIN_METHODS = [
    {
        "id": "autocontinue",
        "label": "Autocontinue (last frame → i2v)",
        "description": "Extract last frame and feed as start image for clip 2+",
    },
    {
        "id": "native_extend",
        "label": "Extend video (ltx-2-mlx)",
        "description": "Clip 1 generate/i2v; clip 2+ native extend_from_video on prior MP4",
    },
]

PIPELINE_PROFILES = [
    {"id": "distilled", "label": "Distilled (fast, default)"},
    {"id": "two_stage", "label": "Two-stage (dev + upscale)"},
    {"id": "hq", "label": "HQ (res_2s + CFG)"},
    {"id": "one_stage", "label": "One-stage (full-res CFG)"},
]

CLIP_MULTIPLIER_MAX = 10
IC_LORA_PRESET_ID = "ic_lora_hdr"
IC_LORA_MOTION_PRESET_ID = "ic_lora_union_motion"
IC_LORA_DEFAULT_SPEC = (
    "https://huggingface.co/buckets/audiohacking/LTX-2.3-22b-IC-LoRA-HDR-bucket/"
    "resolve/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"
)
IC_LORA_UNION_MOTION_SPEC = (
    "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/"
    "resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
)
IC_LORA_DEFAULT_SCALE = 1.0
IC_LORA_BUILTIN_SPECS = frozenset({IC_LORA_DEFAULT_SPEC, IC_LORA_UNION_MOTION_SPEC})
FACE_SWAP_PRESET_ID = "face_swap_head"
FACE_SWAP_DEFAULT_SPEC = (
    "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video/"
    "resolve/main/ltx-2.3/head_swap_v3_rank_adaptive_fro_098.safetensors"
)
FACE_SWAP_DEFAULT_SCALE = 0.98
ID_LORA_CELEBVHQ_PRESET_ID = "id_lora_celebvhq"
ID_LORA_TALKVID_PRESET_ID = "id_lora_talkvid"
ID_LORA_CELEBVHQ_SPEC = (
    "https://huggingface.co/AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K/"
    "resolve/main/lora_weights.safetensors"
)
ID_LORA_TALKVID_SPEC = (
    "https://huggingface.co/AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K/"
    "resolve/main/lora_weights.safetensors"
)
ID_LORA_DEFAULT_PRESET_ID = ID_LORA_CELEBVHQ_PRESET_ID
ID_LORA_DEFAULT_SCALE = 1.0
ID_LORA_PROMPT_PLACEHOLDER = (
    "[VISUAL]: A close-up of a person speaking to camera. "
    "[SPEECH]: Hello, nice to meet you. "
    "[SOUNDS]: Clear speech, quiet room."
)
LIPDUB_PRESET_ID = "lipdub_ic_lora"
LIPDUB_OFFICIAL_REPO = "Lightricks/LTX-2.3-22b-IC-LoRA-LipDub"
LIPDUB_OFFICIAL_FILENAME = "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"
LIPDUB_OFFICIAL_GATED_SPEC = (
    f"https://huggingface.co/{LIPDUB_OFFICIAL_REPO}/resolve/main/{LIPDUB_OFFICIAL_FILENAME}"
)
LIPDUB_PUBLIC_BUCKET_SPEC = (
    "https://huggingface.co/buckets/audiohacking/LTX-2.3-22b-IC-LoRA-LipDub-bucket/"
    f"resolve/{LIPDUB_OFFICIAL_FILENAME}"
)
LIPDUB_DEFAULT_SPEC = LIPDUB_PUBLIC_BUCKET_SPEC
ENV_LIPDUB_LORA = "LTX_WS_LIPDUB_LORA"
LIPDUB_DEFAULT_SCALE = 1.0
DEFAULT_OUTPUT_DIR = REPO_ROOT / "web_outputs"


def _builtin_lipdub_spec() -> str:
    """Built-in LipDub LoRA URL/path: ``LTX_WS_LIPDUB_LORA`` env override, else public bucket."""
    return os.environ.get(ENV_LIPDUB_LORA, "").strip() or LIPDUB_DEFAULT_SPEC


def _pose_control_available() -> bool:
    try:
        from ltx_ic_lora_preprocess import pose_control_available

        return pose_control_available()
    except ImportError:
        return False


DEFAULT_UPLOAD_DIR = REPO_ROOT / "web_uploads"
INDEX_FILE = "index.json"
SETTINGS_FILE = "settings.json"
FPS = 24
PROGRESS_KEEPALIVE_INTERVAL_S = 1.0


def snap_frames(raw: int) -> int:
    k = max(0, round((int(raw) - 1) / 8))
    return 8 * k + 1


def duration_to_frames(seconds: float) -> int:
    return snap_frames(int(seconds * FPS))


def _duration_preset(preset_id: str, seconds: float, num_frames: int) -> dict[str, Any]:
    """One UI duration option with explicit 8k+1 frame count (LTX VAE temporal compression)."""
    nf = int(num_frames)
    return {
        "id": preset_id,
        "seconds": float(seconds),
        "num_frames": nf,
        "label": f"~{seconds:g} s ({nf} frames @ {FPS} fps)",
    }


# 8k+1 frame counts @ 24 fps (see ltx-2-mlx / AGENTS.md frame reference).
DURATION_PRESETS = [
    _duration_preset("2s", 2.0, 49),
    _duration_preset("4s", 4.0, 97),
    _duration_preset("5s", 5.0, 121),
    _duration_preset("6s", 6.0, 145),
    _duration_preset("8s", 8.0, 193),
    _duration_preset("10s", 10.0, 241),
    _duration_preset("15s", 15.0, 361),
    _duration_preset("20s", 20.0, 481),
    _duration_preset("24s", 24.0, 577),
]


def _label_for_lora_spec(spec: str) -> str:
    name = spec.rsplit("/", 1)[-1] if "/" in spec else spec
    if name.endswith(".safetensors"):
        name = name[:-12]
    return name or spec


def _read_custom_loras(output_dir: Path) -> list[dict[str, Any]]:
    from ltx_mlx_backend import _normalize_lora_spec

    raw = read_web_settings(output_dir).get("custom_loras")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    dirty = False
    for item in raw:
        if not isinstance(item, dict):
            continue
        spec = _normalize_lora_spec(str(item.get("spec") or "").strip())
        if not spec:
            continue
        original = str(item.get("spec") or "").strip()
        if spec != original:
            dirty = True
        lid = str(item.get("id") or "").strip() or f"custom_{uuid.uuid4().hex[:8]}"
        try:
            scale = float(item.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        label = str(item.get("label") or "").strip() or _label_for_lora_spec(spec)
        out.append({"id": lid, "label": label, "spec": spec, "scale": scale, "custom": True})
    if dirty:
        # Persist repaired URLs (Path()-mangled https:/… → https://…).
        _write_custom_loras(
            output_dir,
            [{"id": e["id"], "label": e["label"], "spec": e["spec"], "scale": e["scale"]} for e in out],
        )
    return out


def _write_custom_loras(output_dir: Path, entries: list[dict[str, Any]]) -> None:
    data = read_web_settings(output_dir)
    data["custom_loras"] = entries
    write_web_settings(output_dir, data)


def _read_hidden_lora_ids(output_dir: Path) -> set[str]:
    raw = read_web_settings(output_dir).get("hidden_lora_preset_ids")
    if not isinstance(raw, list):
        return set()
    return {str(x).strip() for x in raw if str(x).strip()}


def _persist_hidden_lora_ids(output_dir: Path, ids: set[str]) -> None:
    data = read_web_settings(output_dir)
    data["hidden_lora_preset_ids"] = sorted(ids)
    write_web_settings(output_dir, data)


def _frames_dir(output_dir: Path) -> Path:
    return output_dir / "frames"


def _read_frame_library(output_dir: Path) -> list[dict[str, Any]]:
    raw = read_web_settings(output_dir).get("frame_library")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        fid = str(item.get("id") or "").strip() or f"frame_{uuid.uuid4().hex[:8]}"
        label = str(item.get("label") or "").strip() or "Frame"
        filename = str(item.get("filename") or Path(path).name)
        entry: dict[str, Any] = {
            "id": fid,
            "label": label,
            "path": path,
            "filename": filename,
            "created_at": str(item.get("created_at") or datetime.now().isoformat()),
        }
        for key in ("width", "height", "source_clip_id", "time_s"):
            if item.get(key) is not None:
                entry[key] = item[key]
        out.append(entry)
    return out


def _write_frame_library(output_dir: Path, entries: list[dict[str, Any]]) -> None:
    data = read_web_settings(output_dir)
    data["frame_library"] = entries
    write_web_settings(output_dir, data)


def _frame_for_api(entry: dict[str, Any]) -> dict[str, Any]:
    filename = str(entry.get("filename") or Path(str(entry.get("path") or "")).name)
    out = dict(entry)
    out["image_url"] = f"/api/frames/files/{filename}"
    return out


def _remove_lora_preset_entry(output_dir: Path, preset_id: str) -> None:
    """Drop a custom LoRA entry or hide a built-in/env preset from the Web UI."""
    lid = (preset_id or "").strip()
    if not lid or lid == "none":
        raise ValueError("cannot remove the none preset")
    if lid.startswith("custom_"):
        kept = [e for e in _read_custom_loras(output_dir) if e["id"] != lid]
        if len(kept) == len(_read_custom_loras(output_dir)):
            raise LookupError("LoRA preset not found")
        _write_custom_loras(
            output_dir,
            [
                {"id": e["id"], "label": e["label"], "spec": e["spec"], "scale": e["scale"]}
                for e in kept
            ],
        )
        return
    hidden = _read_hidden_lora_ids(output_dir)
    if lid in hidden:
        raise LookupError("LoRA preset not found")
    hidden.add(lid)
    _persist_hidden_lora_ids(output_dir, hidden)


def _lora_catalog(output_dir: Path | None = None) -> tuple[list[dict[str, Any]], str]:
    """
    LoRA presets for the Web UI (default from LTX_WS_DEFAULT_LORA / server defaults).
    Returns (presets including a None entry, default_preset_id).
    """
    from server import (
        DEFAULT_GLOBAL_LORA_PATH,
        DEFAULT_GLOBAL_LORA_SCALE,
        DEFAULT_LORA_URL,
        ENV_DEFAULT_LORA,
        ENV_DEFAULT_LORA_SCALE,
        _default_loras_from_env,
    )

    def _label_for_spec(spec: str) -> str:
        return _label_for_lora_spec(spec)

    seen: set[str] = set()
    presets: list[dict[str, Any]] = [
        {"id": "none", "label": "None (no LoRA)", "spec": "", "scale": 0.0},
    ]
    default_id = "none"

    def _add(id_: str, label: str, spec: str, scale: float, is_default: bool = False) -> None:
        nonlocal default_id
        key = f"{spec}:{scale}"
        if not spec or key in seen:
            return
        seen.add(key)
        presets.append({"id": id_, "label": label, "spec": spec, "scale": scale})
        if is_default:
            default_id = id_

    default_path = os.environ.get(ENV_DEFAULT_LORA, DEFAULT_GLOBAL_LORA_PATH).strip()
    if not default_path:
        default_path = DEFAULT_LORA_URL
    scale_raw = os.environ.get(ENV_DEFAULT_LORA_SCALE, str(DEFAULT_GLOBAL_LORA_SCALE)).strip()
    try:
        default_scale = float(scale_raw)
    except ValueError:
        default_scale = DEFAULT_GLOBAL_LORA_SCALE

    if default_path:
        label = (
            "OmniNFT RL LoRA (default)"
            if default_path == DEFAULT_LORA_URL
            else f"Default — {_label_for_spec(default_path)}"
        )
        _add(
            "default",
            label,
            default_path,
            default_scale,
            is_default=True,
        )

    for i, (path, scale) in enumerate(_default_loras_from_env()):
        if path == default_path and scale == default_scale:
            continue
        _add(f"env_{i}", f"Env LoRA — {_label_for_spec(path)}", path, scale)

    _add(
        "ic_lora_hdr",
        "IC-LoRA HDR",
        IC_LORA_DEFAULT_SPEC,
        1.0,
    )
    _add(
        "ic_lora_union_motion",
        "IC-LoRA Union Control (motion transfer)",
        IC_LORA_UNION_MOTION_SPEC,
        1.0,
    )
    _add(
        FACE_SWAP_PRESET_ID,
        "Face swap — head swap LoRA (LTX 2.3)",
        FACE_SWAP_DEFAULT_SPEC,
        FACE_SWAP_DEFAULT_SCALE,
    )
    _add(
        ID_LORA_CELEBVHQ_PRESET_ID,
        "ID-LoRA — CelebV-HQ (face + voice)",
        ID_LORA_CELEBVHQ_SPEC,
        ID_LORA_DEFAULT_SCALE,
    )
    _add(
        ID_LORA_TALKVID_PRESET_ID,
        "ID-LoRA — TalkVid (face + voice)",
        ID_LORA_TALKVID_SPEC,
        ID_LORA_DEFAULT_SCALE,
    )
    lipdub_spec = _builtin_lipdub_spec()
    _add(
        LIPDUB_PRESET_ID,
        "LipDub — IC-LoRA (LTX 2.3)",
        lipdub_spec,
        LIPDUB_DEFAULT_SCALE,
    )

    # Customs last, keyed by unique id so they always appear in the menu even when
    # the same URL/scale already exists as a builtin or earlier custom entry.
    if output_dir is not None:
        existing_ids = {str(p.get("id") or "") for p in presets}
        for entry in _read_custom_loras(output_dir):
            eid = str(entry.get("id") or "").strip()
            spec = str(entry.get("spec") or "").strip()
            if not eid or not spec or eid in existing_ids:
                continue
            try:
                scale = float(entry.get("scale", 1.0))
            except (TypeError, ValueError):
                scale = 1.0
            label = str(entry.get("label") or "").strip() or _label_for_spec(spec)
            presets.append(
                {
                    "id": eid,
                    "label": label,
                    "spec": spec,
                    "scale": scale,
                    "custom": True,
                }
            )
            existing_ids.add(eid)

    if output_dir is not None:
        hidden = _read_hidden_lora_ids(output_dir)
        if hidden:
            presets = [presets[0]] + [p for p in presets[1:] if p.get("id") not in hidden]
            if default_id in hidden:
                default_id = "none"

    return presets, default_id


def _ensure_lora_downloaded(spec: str) -> dict[str, Any]:
    from ltx_mlx_backend import _lora_cached_path, _normalize_lora_spec, _resolve_lora_path

    normalized = _normalize_lora_spec(spec)
    cached_path = _lora_cached_path(normalized)
    if cached_path is not None:
        return {
            "ok": True,
            "spec": normalized,
            "path": str(cached_path),
            "cached": True,
        }
    path, _ = _resolve_lora_path(normalized)
    return {"ok": True, "spec": normalized, "path": path, "cached": False}


_lora_ensure_locks: dict[str, asyncio.Lock] = {}


_RUN_BODIES: dict[str, dict[str, Any]] = {}


def resolve_web_dist() -> Path:
    return REPO_ROOT / "web" / "dist"


def web_dist_stale() -> bool:
    """True when built assets are missing or older than web/src sources."""
    dist = resolve_web_dist()
    if not dist.is_dir():
        return True
    assets = dist / "assets"
    js_files = list(assets.glob("index-*.js")) if assets.is_dir() else []
    if not js_files:
        return True
    newest_js = max(js_files, key=lambda path: path.stat().st_mtime)
    src_root = REPO_ROOT / "web" / "src"
    if not src_root.is_dir():
        return False
    try:
        newest_src = max(
            path.stat().st_mtime for path in src_root.rglob("*") if path.is_file()
        )
    except ValueError:
        return False
    return newest_src > newest_js.stat().st_mtime


def ensure_web_dist_built(*, auto_build: bool = True) -> bool:
    """Rebuild web/dist when missing/stale so git pulls show UI changes.

    ``web/dist`` is gitignored, so pulling source alone leaves a stale UI.
    Returns True when a usable dist exists after this call.
    """
    dist = resolve_web_dist()
    if dist.is_dir() and not web_dist_stale():
        return True
    if not auto_build:
        return dist.is_dir() and not web_dist_stale()

    web_dir = REPO_ROOT / "web"
    pkg = web_dir / "package.json"
    if not pkg.is_file():
        log.warning("Web UI package.json missing at %s — cannot auto-build", pkg)
        return False

    npm = shutil.which("npm")
    if not npm:
        log.warning(
            "web/dist is stale/missing and npm was not found — run: cd web && npm run build"
        )
        return False

    cmd = [npm, "run", "build"]
    if not (web_dir / "node_modules").is_dir():
        cmd = [npm, "install", "--no-fund", "--no-audit"]
        log.info("Installing Web UI deps (node_modules missing)…")
        try:
            subprocess.run(cmd, cwd=str(web_dir), check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            log.warning("Web UI npm install failed: %s", exc)
            return False
        cmd = [npm, "run", "build"]

    log.info("Building Web UI (web/dist missing or older than web/src)…")
    try:
        subprocess.run(cmd, cwd=str(web_dir), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        log.warning("Web UI build failed: %s — run: cd web && npm run build", exc)
        return False
    ok = dist.is_dir() and not web_dist_stale()
    if ok:
        log.info("Web UI build ready at %s", dist)
    return ok


def resolve_favicon_path() -> Path | None:
    """Locate favicon for embedded Web UI (built dist, then source public/)."""
    for candidate in (
        resolve_web_dist() / "favicon.ico",
        REPO_ROOT / "web" / "public" / "favicon.ico",
        REPO_ROOT / "web" / "favicon.ico",
    ):
        if candidate.is_file():
            return candidate
    return None


def _upload_extension(kind: str, filename: str | None) -> str:
    """Pick a safe suffix for an uploaded file."""
    ext = Path(filename or "").suffix.lower()
    allowed: dict[str, set[str]] = {
        "image": {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"},
        "audio": {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm"},
        "video": {".mp4", ".mov", ".webm", ".mkv", ".avi"},
    }
    if ext and ext in allowed.get(kind, set()):
        return ext
    defaults = {"image": ".jpg", "audio": ".mp3", "video": ".mp4"}
    return defaults.get(kind, ".bin")


async def _save_upload_file(
    request: Request,
    upload_dir: Path,
    *,
    kind: str = "image",
) -> dict[str, str]:
    """Persist multipart upload; avoids FastAPI UploadFile annotations (PEP 563 ForwardRef)."""
    form = await request.form()
    upload_file = form.get("file")
    if upload_file is None:
        raise ValueError("file is required")
    read = getattr(upload_file, "read", None)
    if read is None:
        raise ValueError("file is required")
    filename = getattr(upload_file, "filename", None) or "upload.bin"
    ext = _upload_extension(kind, filename)
    uid = str(uuid.uuid4())
    dest = upload_dir / f"{uid}{ext}"
    content = await read()
    dest.write_bytes(content)
    return {"path": str(dest), "filename": filename, "kind": kind}


def local_hostname() -> str:
    """Short machine hostname (like ``hostname -s``)."""
    try:
        name = socket.gethostname().strip().split(".")[0]
        if name:
            return name
    except OSError:
        pass
    return "localhost"


def public_host(bind_host: str) -> str:
    host = (bind_host or "").strip()
    if host in ("0.0.0.0", "::", "[::]"):
        return local_hostname()
    return host


def urls_from_request(request: Any) -> tuple[str, str]:
    """Build ws/http URLs from the incoming HTTP request (browser host)."""
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or ""
    ).split(",")[0].strip()
    proto = (
        request.headers.get("x-forwarded-proto") or "http"
    ).split(",")[0].strip().lower()
    if not host:
        return "", ""
    ws_proto = "wss" if proto == "https" else "ws"
    return f"{ws_proto}://{host}/ws", f"{proto}://{host}/"


def build_server_urls(bind_host: str, port: int) -> tuple[str, str]:
    host = public_host(bind_host)
    ws_url = f"ws://{host}:{port}/ws"
    http_url = f"http://{host}:{port}/"
    return ws_url, http_url


def bind_all_http_hint(port: int, bind_host: str = "0.0.0.0") -> str:
    return f"http://{public_host(bind_host)}:{port}/"


def num_frames_to_extend_latent(
    num_frames: int | None,
    *,
    duration_seconds: float | None = None,
    video_path: Path | str | None = None,
) -> int:
    """Latent frame count for native_extend (~one segment of new footage)."""
    from videofentanyl import resolve_extend_latent_frames

    return resolve_extend_latent_frames(
        video_path=video_path,
        num_frames=num_frames,
        duration_seconds=duration_seconds,
        fps=float(FPS),
    )


def _clip_settings_from_body(body: dict[str, Any]) -> dict[str, Any]:
    duration_s = float(body.get("duration_seconds") or 5.0)
    clip_count = int(body.get("clip_count") or 1)
    audiocontinue = bool(body.get("audiocontinue", False))
    autocontinue = bool(body.get("autocontinue", False)) or clip_count > 1 or audiocontinue
    autoconcat = bool(body.get("autoconcat", False)) or clip_count > 1 or audiocontinue
    return {
        "num_frames": body.get("num_frames") or duration_to_frames(duration_s),
        "width": body.get("width"),
        "height": body.get("height"),
        "seed": body.get("seed"),
        "num_steps": body.get("num_steps"),
        "duration_seconds": duration_s,
        "clip_count": clip_count,
        "autocontinue": autocontinue,
        "autoconcat": autoconcat,
        "audiocontinue": audiocontinue,
    }


def scan_local_models() -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    models_dir = REPO_ROOT / "models"
    if not models_dir.is_dir():
        return found
    for child in sorted(models_dir.iterdir()):
        if child.is_dir():
            found.append(
                {
                    "id": str(child),
                    "label": f"Local: {child.name}",
                    "repo": str(child),
                }
            )
    return found


def _all_model_ids(local: list[dict[str, str]] | None = None) -> set[str]:
    local = local or scan_local_models()
    return {m["id"] for m in KNOWN_MODELS + local}


def default_model_preference(
    local: list[dict[str, str]] | None = None,
    cli_default: str = "auto",
) -> str:
    """Prefer the first local weights directory when present, else CLI/env default."""
    local = local or scan_local_models()
    if local:
        return local[0]["id"]
    return (cli_default or "auto").strip() or "auto"


def resolve_model_preference(
    saved: str,
    local: list[dict[str, str]] | None = None,
    cli_default: str = "auto",
) -> str:
    local = local or scan_local_models()
    saved = (saved or "").strip()
    ids = _all_model_ids(local)
    if saved and saved in ids:
        return saved
    return default_model_preference(local, cli_default)


def read_web_settings(output_dir: Path) -> dict[str, Any]:
    path = output_dir / SETTINGS_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def write_web_settings(output_dir: Path, data: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SETTINGS_FILE
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _upload_paths_from_body(body: dict[str, Any], upload_dir: Path) -> list[Path]:
    """Collect upload-dir files referenced by a generate request body."""
    paths: list[Path] = []
    try:
        upload_root = upload_dir.resolve()
    except OSError:
        return paths
    for key in ("image_path", "audio_path", "video_path", "end_image_path", "conditioning_video_path"):
        raw = body.get(key)
        if not raw:
            continue
        try:
            candidate = Path(str(raw)).resolve()
        except (OSError, ValueError):
            continue
        if not candidate.is_file():
            continue
        try:
            if candidate.is_relative_to(upload_root):
                paths.append(candidate)
        except AttributeError:
            if str(candidate).startswith(str(upload_root)):
                paths.append(candidate)
    for item in body.get("video_conditioning") or []:
        if not isinstance(item, list) or not item:
            continue
        raw = item[0]
        if not raw:
            continue
        try:
            candidate = Path(str(raw)).resolve()
        except (OSError, ValueError):
            continue
        if not candidate.is_file():
            continue
        try:
            if candidate.is_relative_to(upload_root):
                paths.append(candidate)
        except AttributeError:
            if str(candidate).startswith(str(upload_root)):
                paths.append(candidate)
    return paths


def _delete_upload_paths(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not delete upload %s: %s", path, exc)


def _cleanup_run_uploads(state: AppState, gen_body: dict[str, Any]) -> None:
    """Delete upload-dir files from a finished run (session clear uses :meth:`clear_upload_dir`)."""
    paths = _upload_paths_from_body(gen_body, state.upload_dir)
    if paths:
        log.info("Web UI: cleaning %d upload file(s)", len(paths))
        _delete_upload_paths(paths)


def clear_upload_dir(upload_dir: Path) -> int:
    """Remove all files in the Web UI upload directory."""
    deleted = 0
    if not upload_dir.is_dir():
        return deleted
    for path in upload_dir.iterdir():
        if not path.is_file():
            continue
        try:
            path.unlink(missing_ok=True)
            deleted += 1
        except OSError as exc:
            log.warning("Could not delete upload %s: %s", path, exc)
    return deleted


def _validate_request_media_paths(body: dict[str, Any]) -> None:
    """Fail fast when the client references uploads that are no longer on disk."""
    from fastapi import HTTPException

    for key, label in (
        ("audio_path", "Audio"),
        ("image_path", "Image"),
        ("end_image_path", "End image"),
        ("video_path", "Video"),
        ("conditioning_video_path", "Conditioning video"),
    ):
        raw = body.get(key)
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            raise HTTPException(
                400,
                f"{label} file not found: {raw}. Re-upload the file and try again.",
            )
    for item in body.get("video_conditioning") or []:
        if not isinstance(item, list) or not item:
            continue
        raw = item[0]
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            raise HTTPException(
                400,
                f"Conditioning video file not found: {raw}. Re-upload the file and try again.",
            )


def _clip_audio_duration_seconds(body: dict[str, Any]) -> float:
    duration_s = body.get("duration_seconds")
    if duration_s is not None:
        return max(0.25, float(duration_s))
    num_frames = body.get("num_frames")
    if num_frames is not None:
        return max(0.25, int(num_frames) / float(FPS))
    return 5.0


def _apply_audio_start_offset(body: dict[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    """Crop a2v audio from ``audio_start_seconds`` before generation."""
    temps: list[Path] = []
    try:
        start = float(body.get("audio_start_seconds") or 0)
    except (TypeError, ValueError):
        start = 0.0
    if start <= 0:
        return body, temps
    audio_path = str(body.get("audio_path") or "").strip()
    if not audio_path:
        return body, temps
    ui_mode = (body.get("mode") or "generate").strip().lower()
    if ui_mode != "a2v":
        return body, temps
    if not media_available():
        raise ValueError("Audio start offset requires PyAV — install with: pip install av")

    client_duration = body.get("audio_source_duration_seconds")
    try:
        client_duration_s = float(client_duration) if client_duration is not None else None
        if client_duration_s is not None and client_duration_s <= 0:
            client_duration_s = None
    except (TypeError, ValueError):
        client_duration_s = None

    clip_need = _clip_audio_duration_seconds(body)
    source_duration = client_duration_s if client_duration_s is not None else probe_audio_duration(audio_path)
    if source_duration is not None and source_duration - start + 0.05 < clip_need:
        raise ValueError(
            f"Only {max(0.0, source_duration - start):.1f}s remains after {start:.1f}s offset, "
            f"but this clip needs ~{clip_need:.1f}s — lower the start offset or "
            "shorten the clip"
        )

    out_file, temp_dir = trim_audio_to_temp(audio_path, start)
    temps.append(temp_dir)
    trimmed_duration = probe_audio_duration(out_file)
    if trimmed_duration is None or trimmed_duration <= 0:
        raise ValueError("Audio crop produced empty output — check the source file")
    if trimmed_duration + 0.05 < clip_need:
        raise ValueError(
            f"Cropped audio is only {trimmed_duration:.1f}s long, "
            f"but this clip needs ~{clip_need:.1f}s — lower the start offset or "
            "shorten the clip"
        )

    new_body = dict(body)
    new_body["audio_path"] = str(out_file)
    new_body["audio_start_seconds"] = 0
    log.info(
        "Web UI: audio cropped from %.2fs — %.1fs segment at %s",
        start,
        trimmed_duration,
        out_file,
    )
    return new_body, temps


@dataclass
class _RunTempFiles:
    """Tracks per-run scratch dirs/files; always cleaned in ``cleanup()``."""

    dirs: list[Path] = field(default_factory=list)

    def add_dir(self, path: Path | None) -> None:
        if path is not None:
            self.dirs.append(path)

    def add_dirs(self, paths: list[Path]) -> None:
        for path in paths:
            self.add_dir(path)

    def cleanup(self) -> None:
        seen: set[Path] = set()
        for raw in self.dirs:
            path = Path(raw)
            if path in seen:
                continue
            seen.add(path)
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.is_file():
                    path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Could not remove run temp %s: %s", path, exc)
        self.dirs.clear()


def _finish_run_teardown(state: AppState, temps: _RunTempFiles) -> None:
    temps.cleanup()
    state.set_active_run(None)
    vs = state.video_server
    if vs is not None:
        gen = getattr(vs, "generator", None)
        if gen is not None:
            if hasattr(gen, "clear_cancel"):
                gen.clear_cancel()
            cleanup = getattr(gen, "cleanup_after_generation", None)
            if callable(cleanup):
                cleanup()


def resolve_source_video_path(state: AppState, body: dict[str, Any]) -> str | None:
    """Resolve ``source_clip_id`` (library) or ``video_path`` (upload) to a local MP4 path."""
    clip_id = str(body.get("source_clip_id") or "").strip()
    if clip_id:
        clip = state.clips.get(clip_id)
        if not clip:
            raise ValueError(f"Source clip not found: {clip_id}")
        if clip.status != RunStatus.DONE.value:
            raise ValueError(f"Source clip is not ready (status={clip.status})")
        if not clip.filename:
            raise ValueError("Source clip has no video file")
        path = state.output_dir / clip.filename
        if not path.is_file():
            raise ValueError(f"Source clip file missing on disk: {clip.filename}")
        return str(path.resolve())

    raw = body.get("video_path")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_file():
        raise ValueError(f"Video file not found: {raw}")
    return str(path.resolve())


def _validate_source_video_request(state: AppState, body: dict[str, Any], ui_mode: str) -> None:
    if ui_mode not in ("retake", "extend", "lipdub", "face_swap"):
        return
    if not body.get("video_path") and not body.get("source_clip_id"):
        raise HTTPException(400, f"{ui_mode} mode requires a source video (upload or library clip)")
    try:
        resolved = resolve_source_video_path(state, body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not resolved:
        raise HTTPException(400, f"{ui_mode} mode requires a source video (upload or library clip)")


def _chain_clip_count(run: RunRecord) -> int:
    return len(run.prompts) or len(run.clip_ids)


def _should_stream_clip_video(run: RunRecord, clip_index: int, total_clips: int) -> bool:
    """Whether the client should receive/stream this clip's MP4 during a chain run."""
    if total_clips <= 1:
        return True
    if run.autoconcat:
        return False
    if run.autocontinue:
        return clip_index == total_clips - 1
    return True


def _build_clip_done_event(
    run: RunRecord,
    clip_id: str,
    clip: ClipRecord,
    index: int,
    total_clips: int,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "clip_done",
        "clip_id": clip_id,
        "index": index,
        "total_clips": total_clips,
        "autoconcat": run.autoconcat,
        "autocontinue": run.autocontinue,
    }
    if _should_stream_clip_video(run, index, total_clips):
        event["video_url"] = clip.video_url
        event["bytes"] = clip.bytes
    return event


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ClipRecord:
    id: str
    prompt: str
    label: str
    video_url: str
    filename: str
    chain_id: str
    clip_index: int
    mode: str
    status: str
    created_at: str
    elapsed_s: Optional[float] = None
    bytes: Optional[int] = None
    error: Optional[str] = None
    num_frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    num_steps: Optional[int] = None
    duration_seconds: Optional[float] = None
    clip_count: Optional[int] = None
    autocontinue: Optional[bool] = None
    autoconcat: Optional[bool] = None
    audiocontinue: Optional[bool] = None


@dataclass
class RunRecord:
    id: str
    status: str
    prompts: list[str]
    chain_id: str
    clip_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    error: Optional[str] = None
    autocontinue: bool = False
    autoconcat: bool = False
    audiocontinue: bool = False
    chain_method: str = "autocontinue"
    merged_url: Optional[str] = None
    merged_clip_id: Optional[str] = None


class AppState:
    def __init__(
        self,
        server_url: str,
        output_dir: Path,
        upload_dir: Path,
        preferred_model: str,
        *,
        embedded: bool = False,
        http_url: str = "",
        active_model: str = "",
        runtime_defaults: dict[str, Any] | None = None,
        server_process: Optional[subprocess.Popen] = None,
        video_server: Any = None,
    ):
        self.server_url = server_url
        self.http_url = http_url
        self.output_dir = output_dir
        self.upload_dir = upload_dir
        from ltx_paths import configure_scratch_root

        configure_scratch_root(output_dir / ".scratch")
        self.preferred_model = preferred_model
        self.active_model = active_model or preferred_model
        self.embedded = embedded
        self.runtime_defaults = runtime_defaults or {}
        self.server_process = server_process
        self.video_server = video_server
        self._cli_model_default = preferred_model
        self.runs: dict[str, RunRecord] = {}
        self.clips: dict[str, ClipRecord] = {}
        self.event_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._pending: asyncio.Queue[str] = asyncio.Queue()
        self._submit_lock = asyncio.Lock()
        self._worker_started = False
        self._worker_task: asyncio.Task[None] | None = None
        self._cancelled_runs: set[str] = set()
        self._active_run_id: str | None = None
        self._sigint_count: int = 0
        self._sigint_last_ts: float = 0.0
        self._uvicorn_server: Any | None = None

    def is_generation_active(self) -> bool:
        if self._active_run_id is not None:
            return True
        vs = self.video_server
        if vs is not None:
            sched = getattr(vs, "scheduler", None)
            if sched is not None and sched.running_generation_id:
                return True
        return False

    def is_pipeline_idle(self) -> bool:
        """True when no run is active and nothing is waiting in the worker queue."""
        if self._active_run_id is not None:
            return False
        if self._pending.qsize() > 0:
            return False
        vs = self.video_server
        if vs is not None:
            sched = getattr(vs, "scheduler", None)
            if sched is not None:
                if sched.running_generation_id:
                    return False
                gen_lock = getattr(sched, "_gen_lock", None)
                if gen_lock is not None and gen_lock.locked():
                    return False
        return True

    async def enqueue_generation_run(self, run_id: str) -> bool:
        """Queue a run for the worker. Returns True if the pipeline was idle (immediate start)."""
        _reconcile_generation_scheduler(self)
        async with self._submit_lock:
            idle = self.is_pipeline_idle()
            run = self.runs.get(run_id)
            if run is not None:
                run.status = (
                    RunStatus.RUNNING.value if idle else RunStatus.QUEUED.value
                )
            await self._pending.put(run_id)
            return idle

    def request_shutdown(self) -> None:
        uv = self._uvicorn_server
        if uv is not None:
            uv.should_exit = True

    def _force_exit(self) -> None:
        vs = self.video_server
        if vs is not None:
            gen = getattr(vs, "generator", None)
            if gen is not None and hasattr(gen, "shutdown"):
                gen.shutdown(wait=False)
        os._exit(130)

    def on_console_interrupt(self) -> None:
        """Idle: graceful shutdown. During MLX work: cancel, then force-quit on repeat."""
        if not self.is_generation_active():
            log.info("Shutting down…")
            self.request_shutdown()
            return

        now = time.monotonic()
        if now - self._sigint_last_ts > 2.0:
            self._sigint_count = 0
        self._sigint_last_ts = now
        self._sigint_count += 1

        self._signal_generator_cancel()
        if self._active_run_id:
            self._cancelled_runs.add(self._active_run_id)

        if self._sigint_count == 1:
            log.warning(
                "Interrupt received — cancelling generation "
                "(press Ctrl+C again within 2s to force quit)"
            )
        else:
            log.warning("Force quit")
            self._force_exit()

    def set_active_run(self, run_id: str | None) -> None:
        self._active_run_id = run_id

    def is_run_cancelled(self, run_id: str) -> bool:
        return run_id in self._cancelled_runs

    def clear_run_cancelled(self, run_id: str) -> None:
        self._cancelled_runs.discard(run_id)

    def _signal_generator_cancel(self) -> None:
        vs = self.video_server
        if vs is None:
            return
        gen = getattr(vs, "generator", None)
        if gen is not None and hasattr(gen, "request_cancel"):
            gen.request_cancel()

    def request_cancel_run(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if not run:
            return False
        if run.status in (
            RunStatus.DONE.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
        ):
            return False
        self._cancelled_runs.add(run_id)
        self._signal_generator_cancel()
        return True

    def cancel_active_generation(self) -> None:
        self._signal_generator_cancel()
        if self._active_run_id:
            self._cancelled_runs.add(self._active_run_id)

    def apply_saved_settings(self) -> None:
        """Load persisted UI settings; default model preference favors local weights."""
        local = scan_local_models()
        saved = str(read_web_settings(self.output_dir).get("preferred_model") or "").strip()
        self.preferred_model = resolve_model_preference(
            saved,
            local,
            self._cli_model_default,
        )

    def persist_preferred_model(self, model: str) -> None:
        model = (model or "auto").strip() or "auto"
        local = scan_local_models()
        ids = _all_model_ids(local)
        if model in ids:
            self.preferred_model = model
        else:
            self.preferred_model = resolve_model_preference(
                model, local, self._cli_model_default
            )
        data = read_web_settings(self.output_dir)
        data["preferred_model"] = self.preferred_model
        write_web_settings(self.output_dir, data)

    def preferred_lora_preset_ids(self) -> list[str]:
        data = read_web_settings(self.output_dir)
        raw = data.get("preferred_lora_preset_ids")
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip() and str(x).strip() != "none"]
        legacy = str(data.get("preferred_lora_preset_id") or "").strip()
        if legacy and legacy != "none":
            return [legacy]
        return []

    def persist_preferred_loras(self, preset_ids: list[str]) -> list[str]:
        clean: list[str] = []
        seen: set[str] = set()
        for raw in preset_ids:
            pid = str(raw or "").strip()
            if not pid or pid == "none" or pid in seen:
                continue
            seen.add(pid)
            clean.append(pid)
        data = read_web_settings(self.output_dir)
        data["preferred_lora_preset_ids"] = clean
        data.pop("preferred_lora_preset_id", None)
        write_web_settings(self.output_dir, data)
        return clean

    def ensure_worker(self) -> None:
        """Start background generation worker (idempotent; restarts if it died)."""
        task = self._worker_task
        if self._worker_started and task is not None and not task.done():
            return
        if task is not None and task.done():
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                exc = None
            if exc is not None:
                log.error("Generation worker exited unexpectedly: %s", exc)
            else:
                log.warning("Generation worker stopped; restarting queue processor")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.apply_saved_settings()
        self.load_index()
        self._worker_task = asyncio.create_task(_worker_loop(self))
        self._worker_started = True

    def load_index(self) -> None:
        path = self.output_dir / INDEX_FILE
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for c in data.get("clips", []):
                self.clips[c["id"]] = ClipRecord(
                    **{k: v for k, v in c.items() if k in ClipRecord.__dataclass_fields__}
                )
            for r in data.get("runs", []):
                self.runs[r["id"]] = RunRecord(
                    **{k: v for k, v in r.items() if k in RunRecord.__dataclass_fields__}
                )
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    def save_index(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / INDEX_FILE
        data = {
            "clips": [asdict(c) for c in self.clips.values()],
            "runs": [asdict(r) for r in self.runs.values()],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def clip_url(self, filename: str) -> str:
        return f"/api/videos/{filename}"

    def delete_clip_record(self, clip_id: str) -> bool:
        clip = self.clips.get(clip_id)
        if not clip:
            return False
        path = self.output_dir / clip.filename
        if path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                log.warning("Could not delete clip file %s: %s", path, exc)
        del self.clips[clip_id]
        for run in list(self.runs.values()):
            if clip_id in run.clip_ids:
                run.clip_ids = [cid for cid in run.clip_ids if cid != clip_id]
        return True

    def delete_chain(self, chain_id: str) -> int:
        removed = 0
        for clip_id, clip in list(self.clips.items()):
            if clip.chain_id == chain_id:
                if self.delete_clip_record(clip_id):
                    removed += 1
        for run_id, run in list(self.runs.items()):
            if run.chain_id == chain_id:
                del self.runs[run_id]
        self.save_index()
        return removed

    def clear_session(self) -> dict[str, int]:
        """Remove all generated outputs and index entries (Web UI session reset)."""
        deleted_files = 0
        seen: set[Path] = set()
        for clip in list(self.clips.values()):
            if clip.filename:
                path = self.output_dir / clip.filename
                if path.is_file() and path not in seen:
                    try:
                        path.unlink()
                        deleted_files += 1
                        seen.add(path)
                    except OSError as exc:
                        log.warning("Could not delete clip file %s: %s", path, exc)
        for path in self.output_dir.glob("*.mp4"):
            if path.is_file() and path not in seen:
                try:
                    path.unlink()
                    deleted_files += 1
                except OSError as exc:
                    log.warning("Could not delete output %s: %s", path, exc)
        clip_count = len(self.clips)
        self.clips.clear()
        self.runs.clear()
        self.event_queues.clear()
        upload_deleted = clear_upload_dir(self.upload_dir)
        self.save_index()
        return {
            "deleted_clips": clip_count,
            "deleted_files": deleted_files,
            "deleted_uploads": upload_deleted,
        }

    async def emit(self, run_id: str, event: dict[str, Any]) -> None:
        q = self.event_queues.get(run_id)
        if q:
            await q.put(event)


def _clip_for_api(state: AppState, clip: ClipRecord) -> dict[str, Any]:
    """Serialize a clip; omit video_url when the output file is no longer on disk."""
    data = asdict(clip)
    filename = str(data.get("filename") or "").strip()
    if filename:
        if (state.output_dir / filename).is_file():
            if not data.get("video_url"):
                data["video_url"] = state.clip_url(filename)
        else:
            data["video_url"] = ""
    return data


def _ensure_web_deps() -> None:
    import importlib
    import subprocess

    for pkg, mod in (
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("python-multipart", "multipart"),
        ("starlette", "starlette"),
    ):
        try:
            __import__(mod)
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                stdout=subprocess.DEVNULL,
            )
            importlib.invalidate_caches()


def _import_videofentanyl():
    from videofentanyl import (
        GenerationParams,
        Job,
        JobStatus,
        VideoSession,
        extract_last_frame,
        load_image_payload,
        load_media_payload,
        sanitize_filename,
        try_autoconcat_clips,
        try_finalize_native_extend_chain,
    )
    return (
        GenerationParams,
        Job,
        JobStatus,
        VideoSession,
        extract_last_frame,
        load_image_payload,
        load_media_payload,
        sanitize_filename,
        try_autoconcat_clips,
        try_finalize_native_extend_chain,
    )


class ProgressVideoSession:
    """VideoSession wrapper that forwards protocol events to the WebUI."""

    def __init__(self, job, mode: str, verbose: bool, on_event: Any, VideoSession):
        self._session = VideoSession(job, mode=mode, verbose=verbose)
        self._on_event = on_event
        self.job = job

    async def run(self, idle_timeout: float | None) -> bool:
        orig_json = self._session._handle_json
        orig_binary = self._session._handle_binary

        async def wrapped_json(raw: str) -> None:
            try:
                msg = json.loads(raw)
                await self._on_event({"type": "protocol", "event": msg})
            except json.JSONDecodeError:
                pass
            await orig_json(raw)

        def wrapped_binary(data: bytes) -> None:
            orig_binary(data)
            kb = sum(len(c) for c in self._session._chunks) / 1024
            asyncio.create_task(
                self._on_event(
                    {
                        "type": "download_progress",
                        "chunks": len(self._session._chunks),
                        "kb": round(kb, 1),
                    }
                )
            )

        self._session._handle_json = wrapped_json
        self._session._handle_binary = wrapped_binary
        return await self._session.run(idle_timeout)


def _set_server_override(url: str) -> None:
    import videofentanyl as vf

    vf._SERVER_OVERRIDE = url


def _run_ws_url(state: AppState) -> str:
    """Loopback WS for embedded runs — same as ``videofentanyl --server``."""
    if state.embedded and state.video_server is not None:
        return f"ws://127.0.0.1:{state.video_server.port}/ws"
    return state.server_url


async def healthcheck_ws(url: str) -> bool:
    import websockets

    try:
        async with websockets.connect(
            url,
            open_timeout=3.0,
            close_timeout=2.0,
            max_size=1024,
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "session_init_v2",
                        "preset_id": "simple_custom_prompt",
                        "curated_prompts": [],
                        "single_clip_mode": True,
                    }
                )
            )
            return True
    except Exception:
        return False


def _resolve_ic_lora_video_conditioning(
    state: AppState,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Normalize IC-LoRA / V2V motion-reference fields into ``video_conditioning``."""
    body = dict(body)
    if (body.get("mode") or "generate").strip().lower() not in ("ic_lora", "v2v"):
        return body
    if body.get("video_conditioning"):
        return body
    try:
        scale = float(body.get("conditioning_video_scale", 1.0))
    except (TypeError, ValueError):
        scale = 1.0
    clip_id = str(body.get("conditioning_clip_id") or "").strip()
    if clip_id:
        clip = state.clips.get(clip_id)
        if not clip:
            raise HTTPException(400, f"Conditioning clip not found: {clip_id}")
        if clip.status != RunStatus.DONE.value:
            raise HTTPException(400, f"Conditioning clip is not ready (status={clip.status})")
        if not clip.filename:
            raise HTTPException(400, "Conditioning clip has no video file")
        path = state.output_dir / clip.filename
        if not path.is_file():
            raise HTTPException(400, f"Conditioning clip file missing on disk: {clip.filename}")
        body["video_conditioning"] = [[str(path.resolve()), scale]]
        return body
    raw = body.get("conditioning_video_path")
    if raw:
        body["video_conditioning"] = [[str(raw), scale]]
    return body


def _ic_lora_has_motion_reference(body: dict[str, Any]) -> bool:
    if body.get("video_conditioning"):
        return True
    if body.get("conditioning_video_path") or body.get("conditioning_clip_id"):
        return True
    return False


def _ic_lora_specs_list(body: dict[str, Any]) -> list[str]:
    specs: list[str] = []
    for item in body.get("lora_specs") or []:
        if isinstance(item, (list, tuple)) and item:
            spec = str(item[0]).strip()
            if spec:
                specs.append(spec)
    return specs


def _ic_lora_is_union_request(body: dict[str, Any]) -> bool:
    """True when the user explicitly selected Union Control (not HDR)."""
    return IC_LORA_UNION_MOTION_SPEC in _ic_lora_specs_list(body)


def _ic_lora_primary_spec(*, selected_specs: list[str] | None = None) -> str:
    """Choose built-in primary LoRA.

    Upstream ``hdr-ic-lora`` defaults to the HDR adapter. Image and/or video are
    optional inputs on that path — they must **not** auto-switch to Union Control.
    Union is opt-in via the LoRA picker (or an explicit Union spec in the request).
    """
    selected = list(selected_specs or [])
    if IC_LORA_UNION_MOTION_SPEC in selected:
        return IC_LORA_UNION_MOTION_SPEC
    if IC_LORA_DEFAULT_SPEC in selected:
        return IC_LORA_DEFAULT_SPEC
    return IC_LORA_DEFAULT_SPEC


def _apply_ic_lora_defaults(body: dict[str, Any]) -> dict[str, Any]:
    """Ensure a sensible primary IC-LoRA; keep extra user-selected LoRAs.

  - Custom-only LoRA list (e.g. CrossView Prompt) → leave as-is (no HDR/Union inject).
  - Explicit Union Control selection → keep Union.
  - Otherwise → HDR IC-LoRA (pure T2V, V2V, I2V, or I2V+V2V — image/video optional).
    """
    if (body.get("mode") or "generate").strip().lower() != "ic_lora":
        return body
    new_body = dict(body)

    parsed: list[list[Any]] = []
    seen_specs: set[str] = set()
    for item in new_body.get("lora_specs") or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        spec = str(item[0]).strip()
        if not spec or spec in seen_specs:
            continue
        try:
            scale = float(item[1])
        except (TypeError, ValueError):
            scale = IC_LORA_DEFAULT_SCALE
        parsed.append([spec, scale])
        seen_specs.add(spec)

    has_builtin = any(spec in IC_LORA_BUILTIN_SPECS for spec, _ in parsed)
    custom_only = bool(parsed) and not has_builtin
    if custom_only:
        # User chose a non-builtin IC-LoRA (CrossView, community adapters, …).
        new_body["lora_specs"] = parsed
        return new_body

    primary = _ic_lora_primary_spec(selected_specs=[spec for spec, _ in parsed])
    merged: list[list[Any]] = [[primary, IC_LORA_DEFAULT_SCALE]]
    seen = {primary}
    for spec, scale in parsed:
        if spec in seen:
            continue
        if spec in IC_LORA_BUILTIN_SPECS and spec != primary:
            continue
        merged.append([spec, scale])
        seen.add(spec)
    new_body["lora_specs"] = merged
    return new_body


def _api_mode(mode: str) -> str:
    """Map Web UI mode to generation_mode (i2v → generate + initial_image; a2v stays a2v)."""
    m = (mode or "generate").strip().lower()
    if m == "i2v":
        return "generate"
    # V2V + LoRA uses the same ICLoraPipeline path as IC-LoRA, without HDR/Union defaults.
    if m == "v2v":
        return "ic_lora"
    return m


def _resolve_seed(raw: Any) -> int:
    """LTX uses seed < 0 (typically -1) for random; None → random."""
    if raw is None:
        return -1
    return int(raw)


def _optional_float(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _local_file_ref(path: str | None) -> str | None:
    """Return an absolute path when ``path`` is a readable local file."""
    if not path:
        return None
    p = Path(str(path).strip())
    if p.is_file():
        return str(p.resolve())
    return None


def _build_params_from_request(body: dict[str, Any], *, state: AppState | None = None) -> Any:
    (
        GenerationParams,
        *_,
    ) = _import_videofentanyl()
    ui_mode = (body.get("mode") or "generate").strip().lower()
    mode = _api_mode(ui_mode)
    image_path = body.get("image_path") if ui_mode in (
        "i2v",
        "a2v",
        "generate",
        "keyframe",
        "ic_lora",
        "v2v",
        "lipdub",
        "face_swap",
        "id_lora",
    ) else None
    end_image_path = body.get("end_image_path") if ui_mode == "keyframe" else None
    audio_path = body.get("audio_path") if ui_mode in ("a2v", "lipdub", "id_lora") else None
    video_path: str | None = None
    if ui_mode in ("retake", "extend", "lipdub", "face_swap"):
        if state is not None:
            video_path = resolve_source_video_path(state, body)
        else:
            video_path = body.get("video_path")
    load_image_payload, load_media_payload = _import_videofentanyl()[5:7]

    image_payload = (
        _local_file_ref(image_path) or load_image_payload(image_path)
        if image_path
        else None
    )
    end_image_payload = (
        _local_file_ref(end_image_path) or load_image_payload(end_image_path)
        if end_image_path
        else None
    )
    audio_payload = (
        _local_file_ref(audio_path) or load_media_payload(audio_path, kind="audio")
        if audio_path
        else None
    )
    video_payload = (
        _local_file_ref(video_path) or load_media_payload(video_path, kind="video")
        if video_path
        else None
    )

    lora_specs: list[tuple[str, float]] = []
    for item in body.get("lora_specs") or []:
        if isinstance(item, list) and len(item) == 2:
            lora_specs.append((str(item[0]), float(item[1])))

    video_conditioning_specs: list[tuple[dict, float]] = []
    for item in body.get("video_conditioning") or []:
        if isinstance(item, list) and len(item) == 2:
            payload = load_media_payload(str(item[0]).strip(), kind="video")
            video_conditioning_specs.append((payload, float(item[1])))

    duration_s = body.get("duration_seconds")
    num_frames = body.get("num_frames")
    if duration_s is not None:
        num_frames = duration_to_frames(float(duration_s))
    elif num_frames is not None:
        num_frames = snap_frames(int(num_frames))

    audio_start_seconds = None
    raw_start = body.get("audio_start_seconds")
    if raw_start not in (None, ""):
        try:
            parsed_start = float(raw_start)
            if parsed_start > 0:
                audio_start_seconds = parsed_start
        except (TypeError, ValueError):
            audio_start_seconds = None

    return GenerationParams(
        prompt=str(body.get("prompt") or "").strip(),
        preset_id="simple_custom_prompt",
        enhancement_enabled=False,
        single_clip_mode=True,
        initial_image=image_payload,
        end_image=end_image_payload,
        seed=_resolve_seed(body.get("seed")),
        num_frames=num_frames,
        height=body.get("height"),
        width=body.get("width"),
        num_steps=body.get("num_steps"),
        generation_mode=mode if ui_mode != "keyframe" else "keyframe",
        audio_input=audio_payload,
        source_video=video_payload,
        retake_start=body.get("retake_start"),
        retake_end=body.get("retake_end"),
        extend_frames=body.get("extend_frames"),
        extend_direction=body.get("extend_direction"),
        lora_specs=lora_specs,
        video_conditioning_specs=video_conditioning_specs,
        enhance_prompt=bool(body.get("enhance_prompt", False)),
        pipeline_profile=str(body.get("pipeline_profile") or "distilled"),
        cfg_scale=body.get("cfg_scale"),
        stg_scale=body.get("stg_scale"),
        stage2_steps=body.get("stage2_steps"),
        no_regen_audio=bool(body.get("no_regen_audio", False)),
        reference_strength=_optional_float(body.get("reference_strength")),
        audio_start_seconds=audio_start_seconds,
        negative_prompt=str(body.get("negative_prompt") or "").strip(),
        skip_stage_2=bool(body.get("skip_stage_2", False)),
        upsample_only=bool(body.get("upsample_only", False)),
        modality_scale=_optional_float(body.get("modality_scale")),
    )


def _cleanup_temp_video(path: str | None) -> None:
    if not path:
        return
    try:
        p = Path(path)
        if p.is_file():
            p.unlink(missing_ok=True)
        parent = p.parent
        if parent.is_dir() and parent.name.startswith(
            (
                "fv_",
                "fvserver_work_",
                "web_audio_trim_",
                "videofentanyl_audio_",
                "ltx_audio_trim_",
            )
        ):
            shutil.rmtree(parent, ignore_errors=True)
    except OSError:
        pass


async def _emit_protocol(on_event: Any, payload: dict[str, Any]) -> None:
    await on_event({"type": "protocol", "event": payload})


def _reconcile_generation_scheduler(state: AppState) -> None:
    vs = state.video_server
    if vs is None:
        return
    sched = getattr(vs, "scheduler", None)
    if sched is not None and hasattr(sched, "reconcile"):
        sched.reconcile()


def _model_progress_payload(video_server: Any) -> dict[str, Any]:
    mp = video_server.generator.model_progress_for_ws()
    return {"model_progress": mp} if mp else {}


async def _emit_generation_progress(
    video_server: Any,
    on_event: Any,
    t_start: float,
    generation_id: str,
) -> None:
    elapsed_s = round(time.time() - t_start, 1)
    extra = _model_progress_payload(video_server)
    mp = extra.get("model_progress")
    payload: dict[str, Any] = {
        "type": "generation_progress",
        "elapsed_s": elapsed_s,
        "phase": mp.get("stage", "generating") if mp else "generating",
        "generation_id": generation_id,
        **extra,
    }
    await on_event(payload)
    await _emit_protocol(
        on_event,
        {
            "type": "generation_keepalive",
            "elapsed_s": elapsed_s,
            "phase": payload["phase"],
            "generation_id": generation_id,
            **extra,
        },
    )


async def _generation_progress_loop(
    video_server: Any,
    on_event: Any,
    t_start: float,
    generation_id: str,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        await _emit_generation_progress(video_server, on_event, t_start, generation_id)
        try:
            await asyncio.wait_for(stop.wait(), timeout=PROGRESS_KEEPALIVE_INTERVAL_S)
        except asyncio.TimeoutError:
            continue


async def _run_clip_inprocess(
    video_server: Any,
    job: Any,
    on_event: Any,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> bool:
    """Run one clip via the embedded VideoServer (no WebSocket round-trip)."""
    from ltx_mlx_backend import GenerationCancelledError
    from videofentanyl import JobStatus

    cancel_check = should_cancel or (lambda: False)
    params = job.params
    vs = video_server
    t0 = time.time()
    gen = getattr(vs, "generator", None)
    if gen is not None and hasattr(gen, "clear_cancel"):
        gen.clear_cancel()

    async def notify(**kwargs: Any) -> None:
        await _emit_protocol(on_event, kwargs)

    job.started_at = time.time()
    video_path: str | None = None
    try:
        async with vs.scheduler.generation_slot(notify) as generation_id:
            await _emit_protocol(
                on_event,
                {
                    "type": "gpu_assigned",
                    "gpu_id": "mlx:0",
                    "session_timeout": 7200,
                    "generation_id": generation_id,
                },
            )
            await _emit_protocol(
                on_event,
                {
                    "type": "ltx2_stream_start",
                    "total_segments": 1,
                    "stream_mode": "single",
                },
            )
            await _emit_protocol(
                on_event,
                {
                    "type": "ltx2_segment_start",
                    "segment_idx": 0,
                    "total_segments": 1,
                },
            )

            progress_stop = asyncio.Event()
            progress_task = asyncio.create_task(
                _generation_progress_loop(vs, on_event, t0, generation_id, progress_stop)
            )
            try:
                gen_task = asyncio.create_task(
                    vs.generator.generate(
                        prompt=params.prompt,
                        image_data=params.initial_image,
                        audio_data=params.audio_input,
                        source_video_data=params.source_video,
                        seed=_resolve_seed(params.seed),
                        num_frames=params.num_frames,
                        height=params.height,
                        width=params.width,
                        mode=params.generation_mode,
                        num_steps=params.num_steps,
                        retake_start=params.retake_start,
                        retake_end=params.retake_end,
                        extend_frames=params.extend_frames,
                        extend_direction=params.extend_direction or "after",
                        lora_specs=list(params.lora_specs) if params.lora_specs else None,
                        video_conditioning_specs=(
                            list(params.video_conditioning_specs)
                            if params.video_conditioning_specs
                            else None
                        ),
                        job_id=generation_id,
                        a2v_visual_i2v_continue=bool(
                            getattr(params, "a2v_visual_i2v_continue", False)
                        ),
                        end_image_data=getattr(params, "end_image", None),
                        enhance_prompt=bool(getattr(params, "enhance_prompt", False)),
                        pipeline_profile=str(
                            getattr(params, "pipeline_profile", None) or "distilled"
                        ),
                        cfg_scale=getattr(params, "cfg_scale", None),
                        stg_scale=getattr(params, "stg_scale", None),
                        stage2_steps=getattr(params, "stage2_steps", None),
                        no_regen_audio=bool(getattr(params, "no_regen_audio", False)),
                        reference_strength=getattr(params, "reference_strength", None),
                        audio_start_seconds=getattr(params, "audio_start_seconds", None),
                        negative_prompt=str(getattr(params, "negative_prompt", "") or ""),
                        skip_stage_2=bool(getattr(params, "skip_stage_2", False)),
                        upsample_only=bool(getattr(params, "upsample_only", False)),
                        modality_scale=getattr(params, "modality_scale", None),
                    )
                )
                while not gen_task.done():
                    if cancel_check() and gen is not None and hasattr(gen, "request_cancel"):
                        gen.request_cancel()
                    await asyncio.sleep(0.2)
                video_path = await gen_task
            finally:
                progress_stop.set()
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_path, job.output_path)
            hdr_src = Path(video_path).with_suffix(".hdr.npz")
            if hdr_src.is_file():
                hdr_dest = job.output_path.with_suffix(".hdr.npz")
                try:
                    shutil.copy2(hdr_src, hdr_dest)
                    log.info("Saved HDR tensor next to clip: %s", hdr_dest)
                except OSError as exc:
                    log.warning("Could not save HDR .npz sidecar: %s", exc)
            job.file_bytes = job.output_path.stat().st_size
            job.chunk_count = 1
            job.ttff_ms = (time.time() - t0) * 1000
            job.gen_latency_ms = job.ttff_ms
            job.e2e_latency_ms = job.ttff_ms

            await on_event(
                {
                    "type": "download_progress",
                    "chunks": 1,
                    "kb": round(job.file_bytes / 1024, 1),
                }
            )
            await _emit_protocol(
                on_event,
                {
                    "type": "ltx2_segment_complete",
                    "segment_idx": 0,
                    "total_segments": 1,
                },
            )
            await _emit_protocol(on_event, {"type": "ltx2_stream_complete"})
            await _emit_protocol(
                on_event,
                {
                    "type": "latency",
                    "generation_ms": int(job.gen_latency_ms or 0),
                    "e2e_ms": int(job.e2e_latency_ms or 0),
                },
            )
            job.finished_at = time.time()
            job.status = JobStatus.DONE
            return True
    except GenerationCancelledError:
        job.finished_at = time.time()
        job.error = "Cancelled"
        job.status = JobStatus.FAILED
        await _emit_protocol(
            on_event,
            {
                "type": "error",
                "error_code": "cancelled",
                "message": "Generation cancelled",
            },
        )
        return False
    except Exception as exc:
        job.finished_at = time.time()
        job.error = str(exc)
        job.status = JobStatus.FAILED
        await _emit_protocol(
            on_event,
            {
                "type": "error",
                "error_code": "generation_failed",
                "message": str(exc),
            },
        )
        return False
    finally:
        _cleanup_temp_video(video_path)


def _audiocontinue_segment_seconds(body: dict[str, Any]) -> float:
    nf = body.get("num_frames")
    if nf is None:
        duration_s = float(body.get("duration_seconds") or 5.0)
        nf = duration_to_frames(duration_s)
    return max(0.25, int(nf) / float(FPS))


def _prepare_audiocontinue_segments(
    body: dict[str, Any],
    num_clips: int,
) -> tuple[list[dict[str, Any]], Path | None]:
    """Split one audio track into per-clip segments (CLI ``--audiocontinue``)."""
    from videofentanyl import load_media_payload, split_audio_for_jobs

    audio_path = str(body.get("audio_path") or "").strip()
    if not audio_path:
        raise ValueError("audiocontinue requires audio_path")
    segment_seconds = _audiocontinue_segment_seconds(body)
    segs, temp_dir = split_audio_for_jobs(
        audio_path,
        segment_seconds=segment_seconds,
        required_segments=num_clips,
    )
    payloads = [load_media_payload(str(p), kind="audio") for p in segs]
    log.info(
        "Web UI: audiocontinue — %d segment(s), ~%.2fs each",
        len(payloads),
        segment_seconds,
    )
    return payloads, temp_dir


def _apply_audiocontinue_audio(
    params: Any,
    index: int,
    audio_segments: list[dict[str, Any]] | None,
) -> None:
    if audio_segments is not None:
        params.audio_input = audio_segments[index]


async def _fail_run_early(
    state: AppState,
    run_id: str,
    message: str,
) -> None:
    run = state.runs.get(run_id)
    if not run:
        return
    run.status = RunStatus.FAILED.value
    run.error = message
    for clip_id in run.clip_ids:
        clip = state.clips.get(clip_id)
        if clip and clip.status != RunStatus.DONE.value:
            clip.status = RunStatus.FAILED.value
            clip.error = message
    state.save_index()
    await state.emit(run_id, {"type": "error", "message": message})


async def _abort_run_cancelled(state: AppState, run_id: str) -> None:
    run = state.runs.get(run_id)
    if not run:
        return
    if run.status == RunStatus.CANCELLED.value:
        return
    run.status = RunStatus.CANCELLED.value
    run.error = "Cancelled by user"
    for clip_id in run.clip_ids:
        clip = state.clips.get(clip_id)
        if clip and clip.status not in (RunStatus.DONE.value,):
            clip.status = RunStatus.CANCELLED.value
    state.save_index()
    await state.emit(
        run_id,
        {
            "type": "run_cancelled",
            "run_id": run_id,
            "message": "Generation cancelled",
        },
    )
    state.clear_run_cancelled(run_id)


def _is_cancelled_clip_failure(
    ok: bool,
    job: Any,
    state: AppState,
    run_id: str,
) -> bool:
    if ok:
        return False
    return state.is_run_cancelled(run_id) or getattr(job, "error", "") == "Cancelled"


def _clip_request_body(
    gen_body: dict[str, Any],
    prompt: str,
    index: int,
    chaining_enabled: bool,
    chain_method: str,
) -> dict[str, Any]:
    """Per-clip request body; start image upload applies only to the first clip when chaining."""
    body = dict(gen_body)
    body["prompt"] = prompt
    if index > 0 and chaining_enabled:
        body.pop("image_path", None)
        body.pop("end_image_path", None)
        if chain_method == "native_extend":
            body.pop("video_path", None)
    return body


def _apply_chain_continuation(
    params: Any,
    i: int,
    chain_method: str,
    chaining_enabled: bool,
    initial_image: Any,
    extract_last_frame: Any,
    load_media_payload: Any,
    prev_path: Path,
    prev_filename: str,
    gen_body: dict[str, Any],
) -> None:
    if i == 0 and initial_image:
        params.initial_image = initial_image
        params.seed = int(time.time_ns() % (2**31 - 1)) or 1
    elif i > 0 and chaining_enabled:
        if chain_method == "native_extend":
            params.initial_image = None
            params.generation_mode = "extend"
            if prev_path.exists():
                params.source_video = load_media_payload(str(prev_path), kind="video")
                if gen_body.get("extend_frames") is not None:
                    params.extend_frames = int(gen_body["extend_frames"])
                else:
                    params.extend_frames = num_frames_to_extend_latent(
                        gen_body.get("num_frames"),
                        duration_seconds=gen_body.get("duration_seconds"),
                        video_path=prev_path,
                    )
                params.extend_direction = str(gen_body.get("extend_direction") or "after")
                params.seed = int(time.time_ns() % (2**31 - 1)) or 1
                from videofentanyl import count_video_frames

                src_frames = count_video_frames(prev_path)
                seg_frames = duration_to_frames(float(gen_body.get("duration_seconds") or 5.0))
                log.info(
                    "Web UI: native_extend clip %d ← video %s "
                    "(source=%sf, segment=%sf, extend_frames=%s)",
                    i + 1,
                    prev_filename,
                    src_frames if src_frames is not None else "?",
                    seg_frames,
                    params.extend_frames,
                )
            else:
                log.warning(
                    "Web UI: native_extend failed — missing prior clip %s",
                    prev_path,
                )
        else:
            if prev_path.exists():
                frame = extract_last_frame(prev_path)
                if frame:
                    params.initial_image = frame
                    params.seed = int(time.time_ns() % (2**31 - 1)) or 1
                    log.info(
                        "Web UI: autocontinue clip %d ← last frame of %s",
                        i + 1,
                        prev_filename,
                    )
                else:
                    log.warning(
                        "Web UI: autocontinue failed — no frame from %s",
                        prev_path,
                    )
            else:
                log.warning(
                    "Web UI: autocontinue failed — missing prior clip %s",
                    prev_path,
                )


def _apply_autocontinue_frame(
    params: Any,
    i: int,
    autocontinue: bool,
    initial_image: Any,
    extract_last_frame: Any,
    prev_path: Path,
    prev_filename: str,
) -> None:
    """Backward-compatible wrapper; prefer _apply_chain_continuation."""
    _apply_chain_continuation(
        params,
        i,
        "autocontinue",
        autocontinue,
        initial_image,
        extract_last_frame,
        None,
        prev_path,
        prev_filename,
        {},
    )


async def _finish_autoconcat(
    state: AppState,
    run: RunRecord,
    run_id: str,
    jobs: list[Any],
    prefix: str,
    prompts: list[str],
    gen_body: dict[str, Any],
    try_autoconcat_clips: Any,
    try_finalize_native_extend_chain: Any,
) -> None:
    if not run.autoconcat or len(jobs) < 2:
        return
    chain_method = getattr(run, "chain_method", None) or gen_body.get("chain_method") or "autocontinue"
    if chain_method == "native_extend":
        try_finalize_native_extend_chain(jobs, prefix, "mp4", verbose=False)
    else:
        autocompact = bool(gen_body.get("autocompact", False))
        try_autoconcat_clips(jobs, prefix, "mp4", verbose=False, compact=autocompact)
    merged_files = sorted(state.output_dir.glob(f"{prefix}_merged_*.mp4"))
    if not merged_files:
        return
    merged_path = merged_files[-1]
    merged_name = merged_path.name
    run.merged_url = state.clip_url(merged_name)
    # Fragment files were removed by autoconcat; drop their index entries.
    for clip_id in list(run.clip_ids):
        state.clips.pop(clip_id, None)
    merged_id = str(uuid.uuid4())
    max_idx = max(
        (
            c.clip_index
            for c in list(state.clips.values())
            if c.chain_id == run.chain_id
        ),
        default=-1,
    )
    for c in list(state.clips.values()):
        if c.chain_id == run.chain_id and c.label in ("CURRENT", "MERGED"):
            c.label = "EDIT"
    state.clips[merged_id] = ClipRecord(
        id=merged_id,
        prompt=f"{prompts[0]} (×{len(jobs)} merged)",
        label="MERGED",
        video_url=run.merged_url,
        filename=merged_name,
        chain_id=run.chain_id,
        clip_index=max_idx + 1,
        mode=str(gen_body.get("mode") or "generate"),
        status=RunStatus.DONE.value,
        created_at=datetime.now().isoformat(),
        bytes=merged_path.stat().st_size,
        **_clip_settings_from_body(gen_body),
    )
    run.merged_clip_id = merged_id
    run.clip_ids = [merged_id]
    state.save_index()
    await state.emit(
        run_id,
        {
            "type": "merged",
            "video_url": run.merged_url,
            "clip_id": merged_id,
            "filename": merged_name,
            "chain_id": run.chain_id,
        },
    )


async def _execute_run(state: AppState, run_id: str) -> None:
    if state.embedded and state.video_server is not None:
        await _execute_run_embedded(state, run_id)
        return
    await _execute_run_via_ws(state, run_id)


async def _execute_run_embedded(state: AppState, run_id: str) -> None:
    log.info("Web UI: executing run %s (in-process)", run_id)
    if state.is_run_cancelled(run_id):
        await _abort_run_cancelled(state, run_id)
        return

    (
        _GenerationParams,
        Job,
        JobStatus,
        _VideoSession,
        extract_last_frame,
        _load_image,
        load_media_payload,
        sanitize_filename,
        try_autoconcat_clips,
        try_finalize_native_extend_chain,
    ) = _import_videofentanyl()

    run = state.runs[run_id]
    state.set_active_run(run_id)
    run.status = RunStatus.RUNNING.value
    run_t0 = time.time()
    await state.emit(
        run_id,
        {
            "type": "run_started",
            "run_id": run_id,
            "autoconcat": run.autoconcat,
            "audiocontinue": run.audiocontinue,
            "autocontinue": run.autocontinue,
            "chain_method": getattr(run, "chain_method", "autocontinue"),
            "clip_count": len(run.prompts),
        },
    )

    jobs: list[Job] = []
    temps = _RunTempFiles()
    gen_body = dict(_RUN_BODIES.get(run_id, {}))
    prompts = run.prompts
    total_clips = len(prompts)
    chaining_enabled = run.autocontinue
    chain_method = getattr(run, "chain_method", None) or gen_body.get("chain_method") or "autocontinue"
    prefix = sanitize_filename(prompts[0]) or "clip"

    audio_segments: list[dict[str, Any]] | None = None

    try:
        try:
            gen_body, trim_dirs = _apply_audio_start_offset(gen_body)
            temps.add_dirs(trim_dirs)
        except Exception as exc:
            log.exception("Web UI: audio start offset failed")
            await _fail_run_early(state, run_id, str(exc))
            return

        if gen_body.get("audiocontinue"):
            try:
                audio_segments, temp_audio_dir = _prepare_audiocontinue_segments(
                    gen_body, len(prompts)
                )
                temps.add_dir(temp_audio_dir)
            except Exception as exc:
                log.exception("Web UI: audiocontinue setup failed")
                await _fail_run_early(state, run_id, str(exc))
                return

        continue_from = gen_body.get("continue_from")
        initial_image = None
        if continue_from:
            parent = state.clips.get(continue_from)
            if parent and parent.filename:
                parent_path = state.output_dir / parent.filename
                if parent_path.exists():
                    initial_image = extract_last_frame(parent_path)

        for i, prompt in enumerate(prompts):
            if state.is_run_cancelled(run_id):
                await _abort_run_cancelled(state, run_id)
                return

            clip_id = run.clip_ids[i]
            clip = state.clips[clip_id]
            clip.status = RunStatus.RUNNING.value
            await state.emit(
                run_id,
                {
                    "type": "clip_started",
                    "clip_id": clip_id,
                    "index": i,
                    "total_clips": len(prompts),
                },
            )

            body = _clip_request_body(gen_body, prompt, i, chaining_enabled, chain_method)
            params = _build_params_from_request(body, state=state)
            _apply_audiocontinue_audio(params, i, audio_segments)
            if i == 0 and initial_image:
                _apply_chain_continuation(
                    params,
                    i,
                    chain_method,
                    True,
                    initial_image,
                    extract_last_frame,
                    load_media_payload,
                    Path(),
                    "",
                    gen_body,
                )
            elif i > 0 and chaining_enabled:
                params.initial_image = None
                prev_clip = state.clips[run.clip_ids[i - 1]]
                prev_path = state.output_dir / prev_clip.filename
                _apply_chain_continuation(
                    params,
                    i,
                    chain_method,
                    True,
                    None,
                    extract_last_frame,
                    load_media_payload,
                    prev_path,
                    prev_clip.filename,
                    gen_body,
                )
                if (
                    params.generation_mode == "a2v"
                    and params.initial_image is not None
                    and params.audio_input is not None
                ):
                    params.a2v_visual_i2v_continue = True

            out_name = clip.filename
            job = Job(
                id=i + 1,
                params=params,
                output_path=state.output_dir / out_name,
                max_attempts=1,
            )
            jobs.append(job)

            async def on_event(event: dict[str, Any], _clip_id: str = clip_id) -> None:
                event["clip_id"] = _clip_id
                await state.emit(run_id, event)

            ok = await _run_clip_inprocess(
                state.video_server,
                job,
                on_event,
                should_cancel=lambda rid=run_id: state.is_run_cancelled(rid),
            )

            if _is_cancelled_clip_failure(ok, job, state, run_id):
                await _abort_run_cancelled(state, run_id)
                return
            if ok:
                clip.status = RunStatus.DONE.value
                clip.elapsed_s = round(job.elapsed, 2)
                clip.bytes = job.file_bytes
                clip.video_url = (
                    state.clip_url(out_name)
                    if _should_stream_clip_video(run, i, total_clips)
                    else ""
                )
                for c in list(state.clips.values()):
                    if c.chain_id == run.chain_id and c.label == "CURRENT":
                        c.label = "EDIT"
                clip.label = "CURRENT"
                log.info(
                    "Web UI clip e2e: run=%s clip=%d/%d status=ok elapsed=%.3fs "
                    "gen_ms=%s e2e_ms=%s output_kb=%s",
                    run_id,
                    i + 1,
                    total_clips,
                    job.elapsed,
                    int(job.gen_latency_ms) if job.gen_latency_ms is not None else "-",
                    int(job.e2e_latency_ms) if job.e2e_latency_ms is not None else "-",
                    (job.file_bytes or 0) // 1024,
                )
                await state.emit(
                    run_id,
                    _build_clip_done_event(run, clip_id, clip, i, total_clips),
                )
            else:
                clip.status = RunStatus.FAILED.value
                clip.error = job.error or "Generation failed"
                run.status = RunStatus.FAILED.value
                run.error = clip.error
                state.save_index()
                log.info(
                    "Web UI clip e2e: run=%s clip=%d/%d status=failed elapsed=%.3fs error=%r",
                    run_id,
                    i + 1,
                    total_clips,
                    job.elapsed,
                    clip.error,
                )
                await state.emit(
                    run_id,
                    {"type": "clip_failed", "clip_id": clip_id, "error": clip.error},
                )
                log.info(
                    "Web UI run e2e grandtotal: run=%s chain=%s clips=%d/%d status=failed "
                    "wall=%.3fs (%dms)",
                    run_id,
                    run.chain_id,
                    i + 1,
                    total_clips,
                    time.time() - run_t0,
                    int((time.time() - run_t0) * 1000),
                )
                return

        await _finish_autoconcat(
            state,
            run,
            run_id,
            jobs,
            prefix,
            prompts,
            gen_body,
            try_autoconcat_clips,
            try_finalize_native_extend_chain,
        )

        run.status = RunStatus.DONE.value
        state.save_index()
        wall_s = time.time() - run_t0
        log.info(
            "Web UI run e2e grandtotal: run=%s chain=%s clips=%d/%d status=ok "
            "wall=%.3fs (%dms) clip_elapsed_s=%s",
            run_id,
            run.chain_id,
            total_clips,
            total_clips,
            wall_s,
            int(wall_s * 1000),
            [round(j.elapsed, 2) for j in jobs],
        )
        await state.emit(
            run_id,
            {"type": "run_done", "run_id": run_id, "chain_id": run.chain_id},
        )
    finally:
        _finish_run_teardown(state, temps)


async def _execute_run_via_ws(state: AppState, run_id: str) -> None:
    log.info("Web UI: executing run %s", run_id)
    if state.is_run_cancelled(run_id):
        await _abort_run_cancelled(state, run_id)
        return

    (
        _GenerationParams,
        Job,
        JobStatus,
        VideoSession,
        extract_last_frame,
        _load_image,
        load_media_payload,
        sanitize_filename,
        try_autoconcat_clips,
        try_finalize_native_extend_chain,
    ) = _import_videofentanyl()

    run = state.runs[run_id]
    state.set_active_run(run_id)
    run.status = RunStatus.RUNNING.value
    run_t0 = time.time()
    await state.emit(
        run_id,
        {
            "type": "run_started",
            "run_id": run_id,
            "autoconcat": run.autoconcat,
            "audiocontinue": run.audiocontinue,
            "autocontinue": run.autocontinue,
            "chain_method": getattr(run, "chain_method", "autocontinue"),
            "clip_count": len(run.prompts),
        },
    )

    _set_server_override(_run_ws_url(state))
    jobs: list[Job] = []
    temps = _RunTempFiles()
    gen_body = dict(_RUN_BODIES.get(run_id, {}))
    prompts = run.prompts
    total_clips = len(prompts)
    chaining_enabled = run.autocontinue
    chain_method = getattr(run, "chain_method", None) or gen_body.get("chain_method") or "autocontinue"
    prefix = sanitize_filename(prompts[0]) or "clip"

    audio_segments: list[dict[str, Any]] | None = None

    try:
        try:
            gen_body, trim_dirs = _apply_audio_start_offset(gen_body)
            temps.add_dirs(trim_dirs)
        except Exception as exc:
            log.exception("Web UI: audio start offset failed")
            await _fail_run_early(state, run_id, str(exc))
            return

        if gen_body.get("audiocontinue"):
            try:
                audio_segments, temp_audio_dir = _prepare_audiocontinue_segments(
                    gen_body, len(prompts)
                )
                temps.add_dir(temp_audio_dir)
            except Exception as exc:
                log.exception("Web UI: audiocontinue setup failed")
                await _fail_run_early(state, run_id, str(exc))
                return

        continue_from = gen_body.get("continue_from")
        initial_image = None
        if continue_from:
            parent = state.clips.get(continue_from)
            if parent and parent.filename:
                parent_path = state.output_dir / parent.filename
                if parent_path.exists():
                    initial_image = extract_last_frame(parent_path)

        for i, prompt in enumerate(prompts):
            if state.is_run_cancelled(run_id):
                await _abort_run_cancelled(state, run_id)
                return

            clip_id = run.clip_ids[i]
            clip = state.clips[clip_id]
            clip.status = RunStatus.RUNNING.value
            await state.emit(
                run_id,
                {
                    "type": "clip_started",
                    "clip_id": clip_id,
                    "index": i,
                    "total_clips": len(prompts),
                },
            )

            body = _clip_request_body(gen_body, prompt, i, chaining_enabled, chain_method)
            params = _build_params_from_request(body, state=state)
            _apply_audiocontinue_audio(params, i, audio_segments)
            if i == 0 and initial_image:
                _apply_chain_continuation(
                    params,
                    i,
                    chain_method,
                    True,
                    initial_image,
                    extract_last_frame,
                    load_media_payload,
                    Path(),
                    "",
                    gen_body,
                )
            elif i > 0 and chaining_enabled:
                params.initial_image = None
                prev_clip = state.clips[run.clip_ids[i - 1]]
                prev_path = state.output_dir / prev_clip.filename
                _apply_chain_continuation(
                    params,
                    i,
                    chain_method,
                    True,
                    None,
                    extract_last_frame,
                    load_media_payload,
                    prev_path,
                    prev_clip.filename,
                    gen_body,
                )
                if (
                    params.generation_mode == "a2v"
                    and params.initial_image is not None
                    and params.audio_input is not None
                ):
                    params.a2v_visual_i2v_continue = True

            out_name = clip.filename
            job = Job(
                id=i + 1,
                params=params,
                output_path=state.output_dir / out_name,
                max_attempts=1,
            )
            jobs.append(job)

            async def on_event(event: dict[str, Any], _clip_id: str = clip_id) -> None:
                event["clip_id"] = _clip_id
                await state.emit(run_id, event)

            session = ProgressVideoSession(
                job, "ltx", False, on_event, VideoSession
            )
            ok = await session.run(idle_timeout=None)

            if state.is_run_cancelled(run_id):
                await _abort_run_cancelled(state, run_id)
                return
            if ok:
                clip.status = RunStatus.DONE.value
                clip.elapsed_s = round(job.elapsed, 2)
                clip.bytes = job.file_bytes
                clip.video_url = (
                    state.clip_url(out_name)
                    if _should_stream_clip_video(run, i, total_clips)
                    else ""
                )
                for c in list(state.clips.values()):
                    if c.chain_id == run.chain_id and c.label == "CURRENT":
                        c.label = "EDIT"
                clip.label = "CURRENT"
                log.info(
                    "Web UI clip e2e: run=%s clip=%d/%d status=ok elapsed=%.3fs "
                    "gen_ms=%s e2e_ms=%s output_kb=%s",
                    run_id,
                    i + 1,
                    total_clips,
                    job.elapsed,
                    int(job.gen_latency_ms) if job.gen_latency_ms is not None else "-",
                    int(job.e2e_latency_ms) if job.e2e_latency_ms is not None else "-",
                    (job.file_bytes or 0) // 1024,
                )
                await state.emit(
                    run_id,
                    _build_clip_done_event(run, clip_id, clip, i, total_clips),
                )
            else:
                clip.status = RunStatus.FAILED.value
                clip.error = job.error or "Generation failed"
                run.status = RunStatus.FAILED.value
                run.error = clip.error
                state.save_index()
                log.info(
                    "Web UI clip e2e: run=%s clip=%d/%d status=failed elapsed=%.3fs error=%r",
                    run_id,
                    i + 1,
                    total_clips,
                    job.elapsed,
                    clip.error,
                )
                await state.emit(
                    run_id,
                    {"type": "clip_failed", "clip_id": clip_id, "error": clip.error},
                )
                log.info(
                    "Web UI run e2e grandtotal: run=%s chain=%s clips=%d/%d status=failed "
                    "wall=%.3fs (%dms)",
                    run_id,
                    run.chain_id,
                    i + 1,
                    total_clips,
                    time.time() - run_t0,
                    int((time.time() - run_t0) * 1000),
                )
                return

        await _finish_autoconcat(
            state,
            run,
            run_id,
            jobs,
            prefix,
            prompts,
            gen_body,
            try_autoconcat_clips,
            try_finalize_native_extend_chain,
        )

        run.status = RunStatus.DONE.value
        state.save_index()
        wall_s = time.time() - run_t0
        log.info(
            "Web UI run e2e grandtotal: run=%s chain=%s clips=%d/%d status=ok "
            "wall=%.3fs (%dms) clip_elapsed_s=%s",
            run_id,
            run.chain_id,
            total_clips,
            total_clips,
            wall_s,
            int(wall_s * 1000),
            [round(j.elapsed, 2) for j in jobs],
        )
        await state.emit(
            run_id,
            {"type": "run_done", "run_id": run_id, "chain_id": run.chain_id},
        )
    finally:
        _finish_run_teardown(state, temps)


async def _worker_loop(state: AppState) -> None:
    while True:
        run_id = await state._pending.get()
        try:
            try:
                await _execute_run(state, run_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("Web UI: run %s failed", run_id)
                run = state.runs.get(run_id)
                if run and run.status != RunStatus.CANCELLED.value:
                    run.status = RunStatus.FAILED.value
                    run.error = str(exc)
                    state.save_index()
                    await state.emit(run_id, {"type": "error", "message": str(exc)})
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            log.exception("Web UI: generation worker error on run %s", run_id)
            run = state.runs.get(run_id)
            if run and run.status not in (
                RunStatus.CANCELLED.value,
                RunStatus.DONE.value,
            ):
                run.status = RunStatus.FAILED.value
                run.error = str(exc)
                state.save_index()
                try:
                    await state.emit(run_id, {"type": "error", "message": str(exc)})
                except Exception:
                    log.exception("Web UI: failed to emit run error for %s", run_id)
        finally:
            _reconcile_generation_scheduler(state)
            try:
                await state.emit(
                    run_id,
                    {
                        "type": "run_complete",
                        "run_id": run_id,
                        "chain_id": state.runs.get(run_id).chain_id if state.runs.get(run_id) else None,
                    },
                )
            except Exception:
                log.exception("Web UI: failed to emit run_complete for %s", run_id)
            _RUN_BODIES.pop(run_id, None)


def create_app(
    state: AppState,
    mount_static: bool = True,
    ws_handler: Callable[..., Any] | None = None,
) -> Any:
    _ensure_web_deps()
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.ensure_worker()
        loop = asyncio.get_running_loop()

        def _on_interrupt() -> None:
            state.on_console_interrupt()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _on_interrupt)
            except (NotImplementedError, RuntimeError):
                pass
        yield
        vs = state.video_server
        if vs is not None:
            gen = getattr(vs, "generator", None)
            if gen is not None and hasattr(gen, "shutdown"):
                gen.shutdown(wait=True)
        if state.server_process:
            state.server_process.terminate()

    app = FastAPI(title="ltx-ws WebUI", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _defaults() -> dict[str, Any]:
        if state.runtime_defaults:
            return dict(state.runtime_defaults)
        return {
            "num_frames": duration_to_frames(5.0),
            "width": 704,
            "height": 480,
            "num_steps": 8,
            "fps": FPS,
        }

    async def _is_connected(request: Request | None = None) -> bool:
        if state.embedded:
            return True
        url = state.server_url
        if request is not None:
            ws_url, _ = urls_from_request(request)
            if ws_url:
                url = ws_url
        return await healthcheck_ws(url)

    @app.get("/api/health")
    async def api_health(request: Request):
        ws_url, http_url = urls_from_request(request)
        if ws_url:
            state.server_url = ws_url
        if http_url:
            state.http_url = http_url
        ok = await _is_connected(request)
        return {"ok": ok, "server_url": state.server_url, "web_url": state.http_url}

    @app.get("/api/config")
    async def api_config(request: Request):
        ws_url, http_url = urls_from_request(request)
        if ws_url:
            state.server_url = ws_url
        if http_url:
            state.http_url = http_url
        local = scan_local_models()
        models = local + KNOWN_MODELS
        ok = await _is_connected(request)
        lora_presets, default_lora_preset_id = _lora_catalog(state.output_dir)
        preferred_lora_ids = state.preferred_lora_preset_ids()
        if not preferred_lora_ids and default_lora_preset_id != "none":
            preferred_lora_ids = [default_lora_preset_id]
        default_model = default_model_preference(local, "auto")
        model_note = (
            "MLX weights only (dgrauet/ltx-2.3-mlx*). "
            "Restart server.py with --model when changing model."
        )
        if state.embedded:
            model_note = (
                f"Active model: {state.active_model}. "
                "Restart server.py with --model <repo> to change weights."
            )
        return {
            "server_connected": ok,
            "embedded": state.embedded,
            "server_url": state.server_url,
            "web_url": state.http_url,
            "active_model": state.active_model,
            "preferred_model": state.preferred_model,
            "default_model": default_model,
            "models": models,
            "resolution_presets": RESOLUTION_PRESETS,
            "duration_presets": DURATION_PRESETS,
            "generation_modes": GENERATION_MODES,
            "chain_methods": CHAIN_METHODS,
            "pipeline_profiles": PIPELINE_PROFILES,
            "clip_multiplier_max": CLIP_MULTIPLIER_MAX,
            "defaults": _defaults(),
            "model_note": model_note,
            "lora_presets": lora_presets,
            "default_lora_preset_id": default_lora_preset_id,
            "preferred_lora_preset_ids": preferred_lora_ids,
            "ic_lora_preset_id": IC_LORA_PRESET_ID,
            "ic_lora_motion_preset_id": IC_LORA_MOTION_PRESET_ID,
            "ic_lora_default_spec": IC_LORA_DEFAULT_SPEC,
            "ic_lora_union_motion_spec": IC_LORA_UNION_MOTION_SPEC,
            "face_swap_preset_id": FACE_SWAP_PRESET_ID,
            "face_swap_default_spec": FACE_SWAP_DEFAULT_SPEC,
            "id_lora_preset_id": ID_LORA_DEFAULT_PRESET_ID,
            "id_lora_celebvhq_preset_id": ID_LORA_CELEBVHQ_PRESET_ID,
            "id_lora_talkvid_preset_id": ID_LORA_TALKVID_PRESET_ID,
            "id_lora_default_spec": ID_LORA_CELEBVHQ_SPEC,
            "id_lora_talkvid_spec": ID_LORA_TALKVID_SPEC,
            "id_lora_prompt_placeholder": ID_LORA_PROMPT_PLACEHOLDER,
            "lipdub_preset_id": LIPDUB_PRESET_ID,
            "lipdub_default_spec": _builtin_lipdub_spec(),
            "lipdub_official_gated_spec": LIPDUB_OFFICIAL_GATED_SPEC,
            "lipdub_official_hf_url": f"https://huggingface.co/{LIPDUB_OFFICIAL_REPO}",
            "lipdub_env_var": ENV_LIPDUB_LORA,
            "pose_control_available": _pose_control_available(),
            "pyav_available": media_available(),
            "audio_trim_available": media_available(),
        }

    @app.post("/api/config/loras")
    async def set_preferred_loras(body: dict[str, Any]):
        raw = body.get("preset_ids") or body.get("preferred_lora_preset_ids") or []
        if not isinstance(raw, list):
            raise HTTPException(400, "preset_ids must be a list")
        catalog, _ = _lora_catalog(state.output_dir)
        valid = {p["id"] for p in catalog if p.get("id") and p["id"] != "none"}
        ids = state.persist_preferred_loras([str(x) for x in raw if str(x) in valid])
        return {"ok": True, "preferred_lora_preset_ids": ids}

    @app.post("/api/loras/custom")
    async def add_custom_lora(body: dict[str, Any]):
        from ltx_mlx_backend import _normalize_lora_spec

        spec = _normalize_lora_spec(str(body.get("spec") or body.get("url") or "").strip())
        if not spec:
            raise HTTPException(400, "spec or url is required")
        if not (
            spec.startswith(("http://", "https://"))
            or (spec.endswith(".safetensors") and Path(spec).expanduser().is_file())
            or Path(spec).expanduser().is_file()
        ):
            raise HTTPException(
                400,
                "spec must be an http(s) URL (e.g. Hugging Face resolve link) or local .safetensors path",
            )
        try:
            scale = float(body.get("scale", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "scale must be a number")
        label = str(body.get("label") or "").strip() or _label_for_lora_spec(spec)
        entries = [
            {
                "id": e["id"],
                "label": e["label"],
                "spec": e["spec"],
                "scale": e["scale"],
            }
            for e in _read_custom_loras(state.output_dir)
        ]
        # Reuse an existing custom with the same normalized URL so the menu does not
        # accumulate orphan ids that never appear in the catalog.
        existing = next((e for e in entries if str(e.get("spec") or "") == spec), None)
        reused = existing is not None
        if existing is not None:
            lid = str(existing["id"])
            existing["label"] = label
            existing["scale"] = scale
        else:
            lid = f"custom_{uuid.uuid4().hex[:8]}"
            entries.append({"id": lid, "label": label, "spec": spec, "scale": scale})
        _write_custom_loras(state.output_dir, entries)

        # If this id was previously hidden from the menu, un-hide it.
        hidden = _read_hidden_lora_ids(state.output_dir)
        if lid in hidden:
            hidden.discard(lid)
            _persist_hidden_lora_ids(state.output_dir, hidden)

        lora_presets, default_lora_preset_id = _lora_catalog(state.output_dir)
        preset = next((p for p in lora_presets if p.get("id") == lid), None)
        if preset is None:
            raise HTTPException(
                500,
                f"Custom LoRA was saved but did not appear in the catalog (id={lid})",
            )

        preferred = state.preferred_lora_preset_ids()
        if lid not in preferred:
            preferred.append(lid)
            state.persist_preferred_loras(preferred)

        return {
            "ok": True,
            "id": lid,
            "reused": reused,
            "preset": preset,
            "lora_presets": lora_presets,
            "default_lora_preset_id": default_lora_preset_id,
            "preferred_lora_preset_ids": state.preferred_lora_preset_ids(),
        }

    @app.delete("/api/loras/preset/{lora_id}")
    async def remove_lora_preset(lora_id: str):
        lid = (lora_id or "").strip()
        if not lid or lid == "none":
            raise HTTPException(400, "cannot remove the none preset")
        try:
            _remove_lora_preset_entry(state.output_dir, lid)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        state.persist_preferred_loras(
            [x for x in state.preferred_lora_preset_ids() if x != lid]
        )
        lora_presets, default_lora_preset_id = _lora_catalog(state.output_dir)
        return {
            "ok": True,
            "deleted": lid,
            "lora_presets": lora_presets,
            "default_lora_preset_id": default_lora_preset_id,
            "preferred_lora_preset_ids": state.preferred_lora_preset_ids(),
        }

    @app.delete("/api/loras/custom/{lora_id}")
    async def delete_custom_lora(lora_id: str):
        """Backward-compatible alias for custom LoRA removal."""
        return await remove_lora_preset(lora_id)

    @app.post("/api/loras/ensure")
    async def ensure_lora(body: dict[str, Any]):
        from ltx_mlx_backend import _normalize_lora_spec

        spec = _normalize_lora_spec(str(body.get("spec") or "").strip())
        if not spec:
            raise HTTPException(400, "spec is required")
        log.info("LoRA ensure requested: %s", spec[:160])
        lock = _lora_ensure_locks.setdefault(spec, asyncio.Lock())
        try:
            async with lock:
                result = await asyncio.to_thread(_ensure_lora_downloaded, spec)
        except Exception as exc:
            from ltx_mlx_backend import format_lora_download_error

            log.warning("LoRA ensure failed for %s: %s", spec, exc)
            raise HTTPException(500, format_lora_download_error(exc, spec)) from exc
        log.info(
            "LoRA ensure complete: %s (cached=%s)",
            spec[:160],
            result.get("cached"),
        )
        return result

    @app.post("/api/config/model")
    async def set_model(body: dict[str, Any]):
        model = str(body.get("model") or "auto").strip()
        state.persist_preferred_model(model)
        if state.embedded:
            return {
                "preferred_model": state.preferred_model,
                "active_model": state.active_model,
                "server_restarted": False,
                "server_connected": True,
                "note": "Restart server.py with --model to load different weights.",
            }
        spawned = False
        if state.server_process:
            state.server_process.terminate()
            state.server_process = None
        if body.get("restart_server"):
            cmd = [
                sys.executable,
                str(REPO_ROOT / "server.py"),
                "--model",
                state.preferred_model,
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
            ]
            state.server_process = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            spawned = True
            for _ in range(60):
                await asyncio.sleep(2)
                if await healthcheck_ws(state.server_url):
                    break
        return {
            "preferred_model": state.preferred_model,
            "server_restarted": spawned,
            "server_connected": await _is_connected(),
        }

    @app.get("/api/clips")
    async def list_clips(chain_id: Optional[str] = None):
        clips = list(state.clips.values())
        if chain_id:
            clips = [c for c in clips if c.chain_id == chain_id]
        clips.sort(key=lambda c: c.created_at)
        return {"clips": [_clip_for_api(state, c) for c in clips]}

    @app.post("/api/session/clear")
    async def clear_session():
        summary = state.clear_session()
        return {"ok": True, **summary}

    @app.delete("/api/clips/{clip_id}")
    async def delete_clip(clip_id: str):
        if not state.delete_clip_record(clip_id):
            raise HTTPException(404, "Clip not found")
        state.save_index()
        return {"ok": True, "deleted": clip_id}

    @app.delete("/api/chains/{chain_id}")
    async def delete_chain(chain_id: str):
        count = state.delete_chain(chain_id)
        if count == 0:
            raise HTTPException(404, "Chain not found")
        return {"ok": True, "deleted": count, "chain_id": chain_id}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        run = state.runs.get(run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        return asdict(run)

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str):
        if run_id not in state.runs:
            raise HTTPException(404, "Run not found")
        run = state.runs[run_id]
        if run.status == RunStatus.CANCELLED.value:
            return {"ok": True, "status": "cancelled"}
        if not state.request_cancel_run(run_id):
            raise HTTPException(
                409,
                f"Cannot cancel run in state {run.status}",
            )
        return {"ok": True, "status": "cancelling"}

    @app.post("/api/generate")
    async def generate(body: dict[str, Any]):
        state.ensure_worker()
        prompt = str(body.get("prompt") or "").strip()
        prompts = body.get("prompts") or []
        if prompt:
            prompts = [prompt] + [p for p in prompts if p.strip()]
        prompts = [p.strip() for p in prompts if p and str(p).strip()]
        if not prompts:
            raise HTTPException(400, "prompt is required")

        ui_mode = (body.get("mode") or "generate").strip().lower()
        continue_from = body.get("continue_from")
        clip_count = int(body.get("clip_count") or 1)
        clip_count = max(1, min(CLIP_MULTIPLIER_MAX, clip_count))
        # Multi-clip runs are one self-contained autocontinue chain (README --count N).
        if clip_count > 1:
            continue_from = None
            chain_id = str(uuid.uuid4())
        else:
            chain_id = body.get("chain_id") or str(uuid.uuid4())

        if ui_mode == "i2v" and not body.get("image_path") and not continue_from:
            raise HTTPException(400, "i2v mode requires an image upload")
        if ui_mode == "a2v" and not body.get("audio_path"):
            raise HTTPException(400, "a2v mode requires an audio upload")
        _validate_source_video_request(state, body, ui_mode)
        body = _resolve_ic_lora_video_conditioning(state, body)
        if ui_mode == "v2v":
            if not body.get("video_conditioning"):
                raise HTTPException(400, "v2v mode requires a reference video")
            # LoRA is optional: empty = pure reference-video conditioning (motion transfer).
            for lora_item in body.get("lora_specs") or []:
                if not isinstance(lora_item, (list, tuple)) or not lora_item:
                    continue
                spec = str(lora_item[0])
                try:
                    await asyncio.to_thread(_ensure_lora_downloaded, spec)
                except Exception as exc:
                    raise HTTPException(
                        400,
                        f"Could not download V2V LoRA weights ({spec}): {exc}",
                    ) from exc
        if ui_mode == "ic_lora":
            body = _apply_ic_lora_defaults(body)
            is_union = _ic_lora_is_union_request(body)
            if is_union and not body.get("video_conditioning"):
                raise HTTPException(
                    400,
                    "Union Control requires a reference video for pose / motion transfer. "
                    "For HDR text-to-video or image-to-video, select the HDR IC-LoRA preset "
                    "(video optional).",
                )
            # HDR path (upstream hdr-ic-lora): video and image are independently optional
            # → T2V / V2V / I2V / I2V+V2V.
            for lora_item in body.get("lora_specs") or []:
                if not isinstance(lora_item, (list, tuple)) or not lora_item:
                    continue
                spec = str(lora_item[0])
                try:
                    await asyncio.to_thread(_ensure_lora_downloaded, spec)
                except Exception as exc:
                    raise HTTPException(
                        400,
                        f"Could not download IC-LoRA weights ({spec}): {exc}",
                    ) from exc
        if ui_mode == "keyframe" and (not body.get("image_path") or not body.get("end_image_path")):
            raise HTTPException(400, "keyframe mode requires start and end image uploads")
        if ui_mode == "retake":
            try:
                rs = int(body.get("retake_start", 0))
                re_ = int(body.get("retake_end", rs + 1))
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, "retake_start and retake_end must be integers") from exc
            if re_ <= rs:
                raise HTTPException(
                    400,
                    "retake requires retake_end > retake_start (end frame is exclusive)",
                )
        if ui_mode == "lipdub":
            lora_items = body.get("lora_specs") or []
            if len(lora_items) != 1:
                raise HTTPException(400, "lipdub requires exactly one LoRA preset (LipDub IC-LoRA)")
            _validate_source_video_request(state, body, ui_mode)
            if not body.get("audio_path") and media_available():
                try:
                    ref_path = resolve_source_video_path(state, body)
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
                if ref_path:
                    from ltx_media import probe_video_info

                    info = probe_video_info(ref_path)
                    if not info.has_audio:
                        raise HTTPException(
                            400,
                            "lipdub requires voice-tone audio: upload an audio file or "
                            "use a reference video with an audio track",
                        )
            for lora_item in lora_items:
                if not isinstance(lora_item, (list, tuple)) or not lora_item:
                    continue
                spec = str(lora_item[0])
                try:
                    await asyncio.to_thread(_ensure_lora_downloaded, spec)
                except Exception as exc:
                    from ltx_mlx_backend import format_lora_download_error

                    raise HTTPException(
                        400,
                        format_lora_download_error(exc, spec),
                    ) from exc
        if ui_mode == "face_swap":
            if not body.get("image_path"):
                raise HTTPException(400, "face_swap requires a face identity image")
            lora_items = body.get("lora_specs") or []
            if len(lora_items) != 1:
                raise HTTPException(400, "face_swap requires exactly one LoRA preset (head swap)")
            _validate_source_video_request(state, body, ui_mode)
            for lora_item in lora_items:
                if not isinstance(lora_item, (list, tuple)) or not lora_item:
                    continue
                spec = str(lora_item[0])
                try:
                    await asyncio.to_thread(_ensure_lora_downloaded, spec)
                except Exception as exc:
                    raise HTTPException(
                        400,
                        f"Could not download face-swap LoRA weights ({spec}): {exc}",
                    ) from exc
        if ui_mode == "id_lora":
            if not body.get("image_path"):
                raise HTTPException(400, "id_lora requires a first-frame reference image")
            if not body.get("audio_path"):
                raise HTTPException(
                    400,
                    "id_lora requires reference audio (~5s identity sample; not the final soundtrack)",
                )
            lora_items = body.get("lora_specs") or []
            if not lora_items:
                body["lora_specs"] = [[ID_LORA_CELEBVHQ_SPEC, ID_LORA_DEFAULT_SCALE]]
                lora_items = body["lora_specs"]
            if len(lora_items) != 1:
                raise HTTPException(400, "id_lora requires exactly one ID-LoRA preset")
            for lora_item in lora_items:
                if not isinstance(lora_item, (list, tuple)) or not lora_item:
                    continue
                spec = str(lora_item[0])
                try:
                    await asyncio.to_thread(_ensure_lora_downloaded, spec)
                except Exception as exc:
                    raise HTTPException(
                        400,
                        f"Could not download ID-LoRA weights ({spec}): {exc}",
                    ) from exc

        _validate_request_media_paths(body)
        try:
            audio_start = float(body.get("audio_start_seconds") or 0)
        except (TypeError, ValueError):
            audio_start = 0.0
        if audio_start < 0:
            raise HTTPException(400, "audio_start_seconds must be >= 0")
        if audio_start > 0 and not media_available():
            raise HTTPException(
                400,
                "audio_start_seconds requires PyAV — install with: pip install av",
            )

        if clip_count > 1 and len(prompts) == 1:
            prompts = [prompts[0]] * clip_count

        audiocontinue = bool(body.get("audiocontinue", False))
        if audiocontinue:
            if ui_mode != "a2v":
                raise HTTPException(400, "audiocontinue only supports a2v mode")
            if not body.get("audio_path"):
                raise HTTPException(400, "audiocontinue requires an audio upload")
            if not media_available():
                raise HTTPException(
                    400,
                    "audiocontinue requires PyAV — install with: pip install av",
                )

        chain_method = str(body.get("chain_method") or "autocontinue").strip().lower()
        if chain_method not in ("autocontinue", "native_extend"):
            chain_method = "autocontinue"
        if chain_method == "native_extend":
            if audiocontinue:
                raise HTTPException(400, "native_extend is incompatible with audiocontinue")
            if ui_mode not in ("generate", "i2v"):
                raise HTTPException(
                    400,
                    "native_extend chaining only supports generate / i2v base modes",
                )

        mode = ui_mode
        run_id = str(uuid.uuid4())
        autocontinue = bool(body.get("autocontinue", False)) or clip_count > 1 or audiocontinue
        autoconcat = bool(body.get("autoconcat", False)) or clip_count > 1 or audiocontinue
        if autoconcat:
            autocontinue = True

        body = dict(body)
        body["chain_method"] = chain_method
        if audiocontinue:
            body["autocompact"] = True
        if clip_count > 1:
            body.pop("continue_from", None)
            body["chain_id"] = chain_id

        existing = [c for c in state.clips.values() if c.chain_id == chain_id]
        base_index = len(existing)

        clip_ids: list[str] = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        from videofentanyl import sanitize_filename as _sanitize

        for i, p in enumerate(prompts):
            clip_id = str(uuid.uuid4())
            slug = _sanitize(p) or "clip"
            filename = f"web_{slug}_{ts}_{i}.mp4"
            label = (
                "ORIGINAL"
                if base_index == 0 and i == 0 and not continue_from
                else "CURRENT"
            )
            clip = ClipRecord(
                id=clip_id,
                prompt=p,
                label=label,
                video_url="",
                filename=filename,
                chain_id=chain_id,
                clip_index=base_index + i,
                mode=mode,
                status=RunStatus.QUEUED.value,
                created_at=datetime.now().isoformat(),
                **_clip_settings_from_body(body),
            )
            state.clips[clip_id] = clip
            clip_ids.append(clip_id)

        run = RunRecord(
            id=run_id,
            status=RunStatus.QUEUED.value,
            prompts=prompts,
            chain_id=chain_id,
            clip_ids=clip_ids,
            created_at=datetime.now().isoformat(),
            autocontinue=autocontinue or (continue_from is not None),
            autoconcat=autoconcat,
            audiocontinue=audiocontinue,
            chain_method=chain_method,
        )
        state.runs[run_id] = run
        _RUN_BODIES[run_id] = body
        state.save_index()

        state.event_queues[run_id] = asyncio.Queue()
        started_immediately = await state.enqueue_generation_run(run_id)
        state.save_index()
        seg_frames = duration_to_frames(float(body.get("duration_seconds") or 5.0))
        if started_immediately:
            log.info(
                "Web UI: executing run %s  chain=%s  clips=%d  mode=%s  chain_method=%s  "
                "duration=%ss (%d frames)  audiocontinue=%s  audio_start=%ss",
                run_id,
                chain_id,
                len(clip_ids),
                mode,
                chain_method,
                body.get("duration_seconds", 5.0),
                seg_frames,
                audiocontinue,
                body.get("audio_start_seconds") or 0,
            )
        else:
            log.info(
                "Web UI: queued run %s  chain=%s  clips=%d  mode=%s  chain_method=%s  "
                "duration=%ss (%d frames)  audiocontinue=%s  audio_start=%ss",
                run_id,
                chain_id,
                len(clip_ids),
                mode,
                chain_method,
                body.get("duration_seconds", 5.0),
                seg_frames,
                audiocontinue,
                body.get("audio_start_seconds") or 0,
            )

        return {
            "run_id": run_id,
            "chain_id": chain_id,
            "clip_ids": clip_ids,
            "status": state.runs[run_id].status,
            "started_immediately": started_immediately,
        }

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str):
        if run_id not in state.runs:
            raise HTTPException(404, "Run not found")
        if run_id not in state.event_queues:
            state.event_queues[run_id] = asyncio.Queue()

        async def stream() -> AsyncIterator[str]:
            q = state.event_queues[run_id]
            run = state.runs[run_id]
            if run.status in (RunStatus.DONE.value, RunStatus.FAILED.value):
                if run.status == RunStatus.DONE.value and run.merged_clip_id:
                    merged = state.clips.get(run.merged_clip_id)
                    if merged and merged.video_url:
                        payload = {
                            "type": "merged",
                            "video_url": merged.video_url,
                            "clip_id": merged.id,
                            "filename": merged.filename,
                            "chain_id": merged.chain_id,
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                elif run.status == RunStatus.DONE.value:
                    for clip_id in run.clip_ids:
                        clip = state.clips.get(clip_id)
                        if clip and clip.status == RunStatus.DONE.value and clip.video_url:
                            payload = {
                                "type": "clip_done",
                                "clip_id": clip.id,
                                "video_url": clip.video_url,
                                "bytes": clip.bytes,
                            }
                            yield f"data: {json.dumps(payload)}\n\n"
                yield f"data: {json.dumps({'type': 'run_complete', 'run_id': run_id, 'chain_id': run.chain_id})}\n\n"
                return
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=120.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("run_complete", "run_done", "error"):
                        if event.get("type") in ("run_complete", "run_done"):
                            break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/upload")
    async def upload(request: Request, kind: str = "image"):
        kind = (kind or "image").strip().lower()
        if kind not in ("image", "audio", "video"):
            raise HTTPException(400, f"unsupported upload kind: {kind}")
        try:
            return await _save_upload_file(request, state.upload_dir, kind=kind)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("Upload failed for kind=%s", kind)
            raise HTTPException(500, f"Upload failed: {exc}") from exc

    @app.get("/api/frames")
    async def list_frames():
        entries = _read_frame_library(state.output_dir)
        entries.sort(key=lambda e: e.get("created_at") or "", reverse=True)
        return {"frames": [_frame_for_api(e) for e in entries]}

    @app.post("/api/frames")
    async def save_frame(request: Request):
        form = await request.form()
        upload_file = form.get("file")
        if upload_file is None:
            raise HTTPException(400, "file is required")
        read = getattr(upload_file, "read", None)
        if read is None:
            raise HTTPException(400, "file is required")
        content = await read()
        if not content:
            raise HTTPException(400, "empty frame file")

        label = str(form.get("label") or "").strip()
        source_clip_id = str(form.get("source_clip_id") or "").strip() or None
        time_raw = form.get("time_s")
        time_s: float | None = None
        if time_raw is not None and str(time_raw).strip():
            try:
                time_s = float(time_raw)
            except (TypeError, ValueError):
                raise HTTPException(400, "time_s must be a number") from None

        frames_root = _frames_dir(state.output_dir)
        frames_root.mkdir(parents=True, exist_ok=True)
        fid = f"frame_{uuid.uuid4().hex[:8]}"
        filename = f"{fid}.png"
        dest = frames_root / filename
        dest.write_bytes(content)

        if not label:
            label = f"Frame @ {time_s:.1f}s" if time_s is not None else "Saved frame"

        entry = {
            "id": fid,
            "label": label,
            "path": str(dest.resolve()),
            "filename": filename,
            "created_at": datetime.now().isoformat(),
        }
        if source_clip_id:
            entry["source_clip_id"] = source_clip_id
        if time_s is not None:
            entry["time_s"] = round(time_s, 3)

        entries = _read_frame_library(state.output_dir)
        entries.append(entry)
        _write_frame_library(state.output_dir, entries)
        return {"ok": True, "frame": _frame_for_api(entry)}

    @app.delete("/api/frames/{frame_id}")
    async def delete_frame(frame_id: str):
        fid = (frame_id or "").strip()
        if not fid:
            raise HTTPException(400, "frame id is required")
        entries = _read_frame_library(state.output_dir)
        kept: list[dict[str, Any]] = []
        removed: dict[str, Any] | None = None
        for entry in entries:
            if entry.get("id") == fid:
                removed = entry
            else:
                kept.append(entry)
        if removed is None:
            raise HTTPException(404, "Frame not found")
        path = Path(str(removed.get("path") or ""))
        if path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                log.warning("Could not delete frame file %s: %s", path, exc)
        _write_frame_library(state.output_dir, kept)
        return {"ok": True, "deleted": fid, "frames": [_frame_for_api(e) for e in kept]}

    @app.get("/api/frames/files/{filename}")
    async def serve_frame_file(filename: str):
        safe = Path(filename).name
        if safe != filename:
            raise HTTPException(400, "invalid filename")
        path = _frames_dir(state.output_dir) / safe
        if not path.is_file():
            raise HTTPException(404, "Frame not found")
        return FileResponse(path, media_type="image/png", filename=safe)

    @app.get("/api/videos/{filename}")
    async def serve_video(filename: str):
        path = state.output_dir / filename
        if not path.is_file():
            raise HTTPException(404, "Video not found")
        return FileResponse(path, media_type="video/mp4", filename=filename)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        path = resolve_favicon_path()
        if path is None:
            raise HTTPException(404, "Favicon not found")
        return FileResponse(path, media_type="image/x-icon")

    if ws_handler is not None:
        @app.websocket("/ws")
        async def websocket_inference(ws: WebSocket) -> None:
            # Must accept in this route — delegating accept() to ws_handler alone
            # yields HTTP 403 on the WebSocket upgrade (Starlette/FastAPI requirement).
            await ws.accept()
            await ws_handler(ws)

    if mount_static and resolve_web_dist().is_dir():
        app.mount("/", StaticFiles(directory=str(resolve_web_dist()), html=True), name="static")

    return app


def build_combined_application(
    ws_handler: Callable[..., Any],
    state: AppState,
) -> Any:
    """Single FastAPI app: WebSocket /ws + HTTP API + static UI (lifespan runs)."""
    return create_app(state, mount_static=True, ws_handler=ws_handler)


async def run_uvicorn(app: Any, host: str, port: int, state: AppState | None = None) -> None:
    _ensure_web_deps()
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    if state is not None:
        state._uvicorn_server = server
    await server.serve()


def run_standalone() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ltx-ws WebUI (standalone)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5299)
    parser.add_argument(
        "--server-url",
        default=os.environ.get(
            "LTX_WS_URL", build_server_urls("127.0.0.1", 8765)[0]
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument(
        "--model",
        default=os.environ.get("LTX_WS_MODEL", "auto"),
    )
    parser.add_argument("--spawn-server", action="store_true")
    args = parser.parse_args()

    ws_url = args.server_url
    http_url = f"http://{public_host(args.host)}:{args.port}/"
    state = AppState(
        server_url=ws_url,
        output_dir=args.output_dir.resolve(),
        upload_dir=args.upload_dir.resolve(),
        preferred_model=args.model,
        http_url=http_url,
    )
    state.apply_saved_settings()
    server_proc = None
    if args.spawn_server:
        server_proc = subprocess.Popen(
            [
                sys.executable,
                str(REPO_ROOT / "server.py"),
                "--model",
                state.preferred_model,
                "--web-ui",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
            ],
            cwd=str(REPO_ROOT),
        )
    state.server_process = server_proc
    app = create_app(state)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
