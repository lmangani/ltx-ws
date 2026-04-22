#!/usr/bin/env python3
"""
server.py — Local LTX WebSocket server (MLX)
============================================
Runs a local WebSocket server compatible with the **videofentanyl** client
(same JSON + binary MP4 framing as the historical FastVideo-hosted API),
using **ltx-2-mlx** ([dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx))
on Apple Silicon.

USAGE
─────
  # Install MLX packages first (see README), then:
  python server.py

  python videofentanyl.py --server ws://localhost:8765/ws \\
      --prompt "a fox running through snow"

  python server.py --port 9000 --num-frames 65 --height 480 --width 704

PROTOCOL — local LTX (same message shapes as legacy 1080p WS)
──────────────────────────────────────────────────────────────
  ← session_init_v2    (client handshake; store session state; may include
                        ``initial_image`` / ``initialImage`` for i2v or autocontinue)
  ← simple_generate    (client triggers generation; optional ``seed``, ``num_frames``,
                        ``height``, ``width`` override server defaults; same image keys
                        as session_init for start / continuation frames)
  → queue_status       (position in queue while waiting; active_generation_id)
  → gpu_assigned       (device ready; includes generation_id for this job)
  → ltx2_stream_start
  → ltx2_segment_start
  → generation_keepalive (periodic JSON while the model runs; ``model_progress`` reserved)
  ← generation_status  (optional client ping while generating)
  → generation_status_ack (phase / elapsed_s / generation_id / optional model_progress)
  → [binary chunks]    (raw MP4 video data)
  → ltx2_segment_complete
  → ltx2_stream_complete
  → latency            (timing info)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import mimetypes
import os
import re
import shutil
import sys
import subprocess
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

NotifyJson = Callable[..., Awaitable[None]]

# ── Dependency bootstrap ───────────────────────────────────────────────────────

def _ensure(pkg: str, import_as: str | None = None):
    """Import a package, auto-installing it if missing."""
    import importlib
    name = import_as or pkg
    try:
        return __import__(name)
    except ImportError:
        print(f"  '{pkg}' not found — installing…", flush=True)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                stdout=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            print(f"Error: could not install '{pkg}'. "
                  f"Please install it manually:\n  pip install {pkg}")
            sys.exit(1)
        importlib.invalidate_caches()
        return __import__(name)

websockets = _ensure("websockets")

from ltx_mlx_backend import (
    LocalVideoGenerator,
    looks_like_hf_repo_id,
    preview_mlx_weights_source,
)

# ── Constants / defaults ───────────────────────────────────────────────────────

DEFAULT_HOST           = "0.0.0.0"
DEFAULT_PORT           = 8765
# Full bf16 HF repo (also used when RAM auto-select picks the largest variant).
DEFAULT_MODEL          = "dgrauet/ltx-2.3-mlx"
MODEL_AUTO             = "auto"  # --model default: pick HF repo from installed RAM
MODEL_MLX_BF16       = DEFAULT_MODEL
MODEL_MLX_Q8         = "dgrauet/ltx-2.3-mlx-q8"
MODEL_MLX_Q4         = "dgrauet/ltx-2.3-mlx-q4"
# Unified-memory Macs: hw.memsize is the relevant pool for MLX (no separate VRAM).
RAM_GB_BF16_MIN      = 64.0
RAM_GB_Q8_MIN        = 32.0
RAM_GB_Q4_MIN        = 16.0
DEFAULT_GLOBAL_LORA_PATH = (
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/"
    "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"
)
DEFAULT_GLOBAL_LORA_SCALE = 1.0
DEFAULT_NUM_FRAMES     = 97    # ~4 s @ 24 fps; LTX requires (8k+1) frames: 9, 17, 25, … 97, 105, …
DEFAULT_HEIGHT         = 480
DEFAULT_WIDTH          = 704   # MLX LTX default width; must be multiple of 32
DEFAULT_FPS            = 24
DEFAULT_INFER_STEPS    = 8     # distilled one-stage default in ltx-2-mlx
DEFAULT_CHUNK_SIZE     = 64 * 1024  # 64 KB per WebSocket binary message
GENERATION_KEEPALIVE_INTERVAL_S = 15.0

# LTX2 frame requirement: num_frames must satisfy (frames - 1) % 8 == 0
# e.g. 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129
LTX2_SPATIAL_ALIGN = 32

ENV_DEFAULT_LORA = "LTX_WS_DEFAULT_LORA"
ENV_DEFAULT_LORA_SCALE = "LTX_WS_DEFAULT_LORA_SCALE"
ENV_DEFAULT_LORAS = "LTX_WS_DEFAULT_LORAS"  # comma-separated: path:scale,path:scale
ENV_ENABLE_LORA = "LTX_WS_ENABLE_LORA"
ENV_DEFAULT_MODEL = "LTX_WS_MODEL"  # if set, used as --model default when omitted (use "auto" or a repo id)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fvserver")


def _default_loras_from_env() -> list[tuple[str, float]]:
    multi = (os.environ.get(ENV_DEFAULT_LORAS) or "").strip()
    if multi:
        specs: list[tuple[str, float]] = []
        for part in multi.split(","):
            entry = part.strip()
            if not entry:
                continue
            if ":" in entry:
                path, s = entry.rsplit(":", 1)
                try:
                    scale = float(s.strip())
                except ValueError:
                    continue
            else:
                path = entry
                scale = 1.0
            path = path.strip()
            if path:
                specs.append((path, scale))
        if specs:
            return specs

    path = os.environ.get(ENV_DEFAULT_LORA, DEFAULT_GLOBAL_LORA_PATH).strip()
    if not path:
        return []
    scale_raw = os.environ.get(ENV_DEFAULT_LORA_SCALE, str(DEFAULT_GLOBAL_LORA_SCALE)).strip()
    try:
        scale = float(scale_raw)
    except ValueError:
        scale = DEFAULT_GLOBAL_LORA_SCALE
    return [(path, scale)]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on", "y")


def _parser_default_model_arg() -> str:
    """Default for ``--model`` when the flag is omitted: ``$LTX_WS_MODEL`` or ``auto``."""
    v = (os.environ.get(ENV_DEFAULT_MODEL) or "").strip()
    return v if v else MODEL_AUTO


def _physical_memory_bytes() -> int | None:
    """Best-effort total physical RAM (bytes).

    Apple Silicon uses unified memory for MLX (no separate VRAM bucket to query).
    """
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return int(out)
        except (subprocess.CalledProcessError, ValueError, OSError):
            return None
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        return int(parts[1]) * 1024  # kB → bytes
        except (OSError, ValueError, IndexError):
            return None
    return None


def _select_hf_repo_for_ram(ram_bytes: int | None) -> tuple[str, str]:
    """Return ``(HF repo id, short log line)`` for RAM-based auto model."""
    if ram_bytes is None:
        return (
            MODEL_MLX_BF16,
            f"RAM unknown — defaulting to {MODEL_MLX_BF16} (pass explicit --model to override)",
        )
    ram_gb = ram_bytes / (1024**3)
    if ram_gb >= RAM_GB_BF16_MIN:
        return MODEL_MLX_BF16, f"~{ram_gb:.1f} GiB RAM → bf16 ({MODEL_MLX_BF16})"
    if ram_gb >= RAM_GB_Q8_MIN:
        return MODEL_MLX_Q8, f"~{ram_gb:.1f} GiB RAM → int8 ({MODEL_MLX_Q8})"
    if ram_gb >= RAM_GB_Q4_MIN:
        return MODEL_MLX_Q4, f"~{ram_gb:.1f} GiB RAM → int4 ({MODEL_MLX_Q4})"
    return (
        MODEL_MLX_Q4,
        f"~{ram_gb:.1f} GiB RAM (below {RAM_GB_Q4_MIN:.0f} GiB guidance) → int4 ({MODEL_MLX_Q4}); "
        "expect memory pressure",
    )


def _resolve_model_cli_value(raw: str) -> tuple[str, str | None]:
    """Expand ``auto`` to an HF repo; explicit repo ids and local paths pass through.

    Returns ``(resolved_model, auto_note)`` where ``auto_note`` is set only when
    ``raw`` was ``auto`` (case-insensitive).
    """
    s = (raw or "").strip()
    if not s:
        return raw, None
    if s.lower() != MODEL_AUTO.lower():
        return s, None
    mem = _physical_memory_bytes()
    repo, note = _select_hf_repo_for_ram(mem)
    return repo, note


def _spill_slug(prompt: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^\w\s-]+", "", prompt.lower().strip())[:maxlen]
    s = re.sub(r"[\s_]+", "_", s).strip("_")
    return s or "clip"


def _largest_mp4_under(root: Path) -> Path | None:
    """Best-effort: newest non-empty ``*.mp4`` under ``root`` (recursive)."""
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


def _cleanup_temp_video(path: str | None) -> None:
    """Remove a temp MP4 and its parent directory (best-effort)."""
    if not path:
        return
    try:
        p = Path(path)
        p.unlink(missing_ok=True)
        try:
            p.parent.rmdir()
        except OSError:
            pass
    except OSError:
        pass


# ── Image helpers ──────────────────────────────────────────────────────────────

def _resolve_initial_image_payload(msg: dict, session: dict) -> dict | str | None:
    """
    Pick the first non-empty image payload from the generate message, then session.

    Returns a ``dict`` (data URL / base64) or a ``str`` (filesystem path or ``http(s)`` URL)
    for ``GenerationRequest`` / ``InputConfig(image_path=…)``, matching FastVideo I2V usage.
    """
    keys = (
        "initial_image",
        "initialImage",
        "continuation_frame",
        "continuationFrame",
        "start_image",
        "startImage",
        # FastVideo-style / OpenAPI-style keys used by other clients
        "input_reference",
        "inputReference",
        "reference_image",
        "referenceImage",
        "inputs",
    )
    for source in (msg, session):
        for key in keys:
            raw = source.get(key)
            if key == "inputs" and isinstance(raw, dict):
                nested = raw.get("image_path") or raw.get("imagePath")
                if isinstance(nested, str) and nested.strip():
                    p = nested.strip()
                    if os.path.isfile(p) or p.startswith(("http://", "https://")):
                        return p
                    return {"data_url": nested, "mime_type": "image/jpeg"}
                if isinstance(nested, dict) and nested:
                    return nested
            if isinstance(raw, dict) and raw:
                return raw
    return None


def _resolve_audio_payload(msg: dict, session: dict) -> dict | str | None:
    keys = (
        "audio_input",
        "audioInput",
        "input_audio",
        "inputAudio",
        "audio_path",
        "audioPath",
        "audio",
        "inputs",
    )
    for source in (msg, session):
        for key in keys:
            raw = source.get(key)
            if key == "inputs" and isinstance(raw, dict):
                nested = raw.get("audio_path") or raw.get("audioPath") or raw.get("audio")
                if isinstance(nested, (str, dict)) and nested:
                    return nested
            if isinstance(raw, (str, dict)) and raw:
                return raw
    return None


def _resolve_source_video_payload(msg: dict, session: dict) -> dict | str | None:
    keys = (
        "source_video",
        "sourceVideo",
        "video_path",
        "videoPath",
        "video",
        "input_video",
        "inputVideo",
        "inputs",
    )
    for source in (msg, session):
        for key in keys:
            raw = source.get(key)
            if key == "inputs" and isinstance(raw, dict):
                nested = raw.get("video_path") or raw.get("videoPath") or raw.get("video")
                if isinstance(nested, (str, dict)) and nested:
                    return nested
            if isinstance(raw, (str, dict)) and raw:
                return raw
    return None


def _resolve_lora_specs(msg: dict, session: dict) -> list[tuple[str, float]]:
    raw = msg.get("lora_paths") or session.get("lora_paths") or []
    out: list[tuple[str, float]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            path = str(item[0]).strip()
            try:
                scale = float(item[1])
            except (TypeError, ValueError):
                continue
            if path:
                out.append((path, scale))
        elif isinstance(item, str):
            path = item.strip()
            if path:
                out.append((path, 1.0))
    return out


def _resolve_video_conditioning_specs(msg: dict, session: dict) -> list[tuple[dict | str, float]]:
    raw = msg.get("video_conditioning") or session.get("video_conditioning") or []
    out: list[tuple[dict | str, float]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            src = item[0]
            if not isinstance(src, (dict, str)):
                continue
            try:
                scale = float(item[1])
            except (TypeError, ValueError):
                continue
            if src:
                out.append((src, scale))
        elif isinstance(item, (dict, str)) and item:
            out.append((item, 1.0))
    return out



# ── Single-flight generation queue (in-memory, self-cleaning) ───────────────────

class GenerationScheduler:
    """
    Ensures only one ``LocalVideoGenerator.generate`` runs at a time.

    Uses an asyncio lock plus a wait counter so clients blocked on the lock get a
    ``queue_status`` with their position and the UUID of the job currently on the
    device.  State is cleared in ``finally`` blocks — no persistence.
    """

    __slots__ = ("_gen_lock", "_meta", "_n_waiters", "_running_id")

    def __init__(self) -> None:
        self._gen_lock = asyncio.Lock()
        self._meta = asyncio.Lock()
        self._n_waiters = 0
        self._running_id: str | None = None

    @property
    def running_generation_id(self) -> str | None:
        return self._running_id

    @contextlib.asynccontextmanager
    async def generation_slot(self, notify: NotifyJson):
        """
        ``notify`` is awaited as ``await notify(type=..., ...)`` for ``queue_status``.

        Yields a fresh ``generation_id`` (UUID string) after the exclusive lock is
        acquired.
        """
        async with self._meta:
            self._n_waiters += 1
            ahead = self._n_waiters - 1
        held = False
        try:
            if ahead > 0:
                async with self._meta:
                    active = self._running_id
                await notify(
                    type="queue_status",
                    position=ahead,
                    available_gpus=0,
                    total_gpus=1,
                    active_generation_id=active,
                )
            await self._gen_lock.acquire()
            held = True
            gid = str(uuid.uuid4())
            async with self._meta:
                self._running_id = gid
            try:
                yield gid
            finally:
                async with self._meta:
                    self._running_id = None
                if held:
                    self._gen_lock.release()
                    held = False
        finally:
            async with self._meta:
                self._n_waiters -= 1


# ── Per-connection request handler ─────────────────────────────────────────────

class RequestHandler:
    """
    Handles a single WebSocket connection.  Receives session_init_v2 and
    simple_generate messages, runs generation, and streams binary video back.
    """

    def __init__(
        self,
        ws:         Any,
        generator:  LocalVideoGenerator,
        verbose:    bool,
        chunk_size: int,
        scheduler:  GenerationScheduler,
        spill_dir:  Path,
    ) -> None:
        self.ws         = ws
        self.generator  = generator
        self.verbose    = verbose
        self.chunk_size = chunk_size
        self.scheduler  = scheduler
        self.spill_dir   = spill_dir
        self._session:  dict = {}
        self._t0        = time.time()
        self._bg_disconnect_task: asyncio.Task | None = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ts(self) -> str:
        return f"[{time.time() - self._t0:6.2f}s]"

    def _dbg(self, msg: str) -> None:
        if self.verbose:
            print(f"    {self._ts()} {msg}", flush=True)

    async def _send_json(self, **kwargs) -> None:
        await self.ws.send(json.dumps(kwargs))

    def _ws_model_progress_payload(self) -> dict[str, Any]:
        mp = self.generator.model_progress_for_ws()
        return {"model_progress": mp} if mp else {}

    def _spill_copy(self, video_path: str, generation_id: str, prompt: str) -> None:
        """If the client is gone, persist the finished MP4 under ``spill_dir``."""
        try:
            src = Path(video_path)
            if not src.is_file():
                return
            self.spill_dir.mkdir(parents=True, exist_ok=True)
            slug = _spill_slug(prompt)
            dest = self.spill_dir / f"{generation_id}_{slug}.mp4"
            shutil.copy2(src, dest)
            log.info(
                "  ◆ spill-saved (client disconnected) → %s",
                dest,
            )
        except OSError as exc:
            log.error("  ✗ spill copy failed: %s", exc)

    async def _drain_disconnected_job(
        self,
        gen_task: asyncio.Task,
        generation_id: str,
        prompt: str,
    ) -> None:
        """After the client leaves mid-flight, wait for MLX then spill (runs in background)."""
        try:
            video_path = await gen_task
        except asyncio.CancelledError:
            log.warning(
                "  ◆ MLX wait for job %s cancelled (server shutting down?); "
                "worker thread may still be finishing",
                generation_id[:8],
            )
            raise
        except Exception as gen_exc:
            log.error(
                "  ✗ background generation %s after client disconnect: %s",
                generation_id[:8],
                gen_exc,
            )
            return
        try:
            self._spill_copy(video_path, generation_id, prompt)
        finally:
            _cleanup_temp_video(video_path)

    async def _handle_client_msg_while_generating(
        self,
        frame: Any,
        generation_id: str,
        t_start: float,
    ) -> None:
        if isinstance(frame, bytes):
            return
        try:
            msg = json.loads(frame)
        except (json.JSONDecodeError, TypeError, ValueError):
            return
        t = msg.get("type", "")
        if t == "generation_status":
            try:
                await self._send_json(
                    type="generation_status_ack",
                    generation_id=generation_id,
                    phase="generating",
                    elapsed_s=round(time.time() - t_start, 1),
                    **self._ws_model_progress_payload(),
                )
            except Exception:
                pass

    # ── main loop ─────────────────────────────────────────────────────────────

    async def handle(self) -> None:
        async for frame in self.ws:
            if isinstance(frame, bytes):
                continue  # clients should not send binary frames
            try:
                msg = json.loads(frame)
            except (json.JSONDecodeError, ValueError):
                continue

            t = msg.get("type", "")
            self._dbg(f"← {t}")

            if t == "session_init_v2":
                self._session = msg
            elif t == "simple_generate":
                await self._handle_generate(msg)
                return  # connection is done after one generation
            # all other client messages (ping, set_auto_extension, …) are ignored

    # ── generation ────────────────────────────────────────────────────────────

    async def _handle_generate(self, msg: dict) -> None:
        prompt = msg.get("prompt", "").strip()
        raw_neg = msg.get("negative_prompt")
        negative_prompt = raw_neg.strip() if isinstance(raw_neg, str) else ""

        if not prompt:
            await self._send_json(
                type="error",
                error_code="invalid_prompt",
                message="Empty prompt",
            )
            return

        # Resolve initial image: prefer simple_generate, then session_init_v2.
        initial_image: dict | str | None = _resolve_initial_image_payload(msg, self._session)
        audio_input = _resolve_audio_payload(msg, self._session)
        source_video = _resolve_source_video_payload(msg, self._session)
        lora_specs = _resolve_lora_specs(msg, self._session)
        video_conditioning_specs = _resolve_video_conditioning_specs(msg, self._session)
        total_loras = len(lora_specs) + self.generator.default_lora_count()

        def _msg_int(name: str, default: int) -> int:
            raw = msg.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        gen_seed = _msg_int("seed", 1024)
        gen_num_frames = _msg_int("num_frames", self.generator.num_frames)
        gen_height = _msg_int("height", self.generator.height)
        gen_width = _msg_int("width", self.generator.width)
        gen_num_steps = _msg_int("num_steps", self.generator.inference_steps)
        gen_retake_start = _msg_int("retake_start", 1)
        gen_retake_end = _msg_int("retake_end", gen_retake_start)
        gen_extend_frames = _msg_int("extend_frames", 2)
        gen_extend_direction = str(msg.get("extend_direction") or "after").strip().lower()

        mode = str(msg.get("mode") or msg.get("generation_mode") or "").strip().lower()
        if not mode:
            if source_video:
                mode = "extend" if ("extend" in str(msg.get("operation", "")).lower()) else "retake"
            elif audio_input:
                mode = "a2v"
            elif lora_specs and video_conditioning_specs:
                mode = "ic_lora"
            else:
                mode = "generate"

        async def _notify_queue(**kwargs: Any) -> None:
            await self._send_json(**kwargs)

        async with self.scheduler.generation_slot(_notify_queue) as generation_id:
            t_start = time.time()
            upscale_tail = ""
            if self.generator.spatial_upscale:
                upscale_tail = (
                    f"  [upscale: stage1 @ {gen_width // 2}×{gen_height // 2}px "
                    f"(½ res) → output {gen_width}×{gen_height}px; "
                    f"server stage1_floor={self.generator.stage1_steps} "
                    f"(client num_steps={gen_num_steps} is one-stage only)]"
                )
            log.info(
                "  ▶ generation %s  mode=%s  prompt=%r  image=%s  audio=%s  video=%s "
                "loras=%s (req=%s default=%s) vcond=%s seed=%s  %s×%s  frames=%s  steps=%s%s",
                generation_id,
                mode,
                prompt[:72],
                "yes" if initial_image else "no",
                "yes" if audio_input else "no",
                "yes" if source_video else "no",
                total_loras,
                len(lora_specs),
                self.generator.default_lora_count(),
                len(video_conditioning_specs),
                gen_seed,
                gen_height,
                gen_width,
                gen_num_frames,
                gen_num_steps,
                upscale_tail,
            )

            # ── gpu_assigned ──────────────────────────────────────────────────
            await self._send_json(
                type="gpu_assigned",
                gpu_id="mlx:0",
                session_timeout=7200,
                generation_id=generation_id,
            )
            self._dbg("→ gpu_assigned")

            # ── stream_start ──────────────────────────────────────────────────
            await self._send_json(
                type="ltx2_stream_start",
                total_segments=1,
                stream_mode="single",
            )
            self._dbg("→ ltx2_stream_start")

            # ── segment_start ─────────────────────────────────────────────────
            await self._send_json(
                type="ltx2_segment_start",
                segment_idx=0,
                total_segments=1,
            )
            self._dbg("→ ltx2_segment_start (segment 0/1)")

            # ── generate (keepalive + concurrent client JSON e.g. generation_status)
            video_path: str | None = None

            async def _keepalive() -> None:
                while True:
                    await asyncio.sleep(GENERATION_KEEPALIVE_INTERVAL_S)
                    try:
                        await self._send_json(
                            type="generation_keepalive",
                            elapsed_s=round(time.time() - t_start, 1),
                            phase="generating",
                            generation_id=generation_id,
                            **self._ws_model_progress_payload(),
                        )
                    except Exception:
                        break

            ka_task = asyncio.create_task(_keepalive())
            try:
                gen_task = asyncio.create_task(
                    self.generator.generate(
                        prompt=prompt,
                        image_data=initial_image,
                        audio_data=audio_input,
                        source_video_data=source_video,
                        negative_prompt=negative_prompt,
                        seed=gen_seed,
                        num_frames=gen_num_frames,
                        height=gen_height,
                        width=gen_width,
                        mode=mode,
                        num_steps=gen_num_steps,
                        retake_start=gen_retake_start,
                        retake_end=gen_retake_end,
                        extend_frames=gen_extend_frames,
                        extend_direction=gen_extend_direction,
                        lora_specs=lora_specs,
                        video_conditioning_specs=video_conditioning_specs,
                        job_id=generation_id,
                    )
                )
                try:
                    while not gen_task.done():
                        recv_t = asyncio.create_task(self.ws.recv())
                        done, _ = await asyncio.wait(
                            {gen_task, recv_t},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if gen_task in done:
                            recv_t.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await recv_t
                            break
                        try:
                            frame = recv_t.result()
                        except websockets.exceptions.ConnectionClosed:
                            log.warning(
                                "  ⚠ client disconnected during generation %s — "
                                "releasing queue slot; MLX continues in-process (cannot preempt "
                                "mid-step); spill-save runs when the job finishes",
                                generation_id[:8],
                            )
                            self._bg_disconnect_task = asyncio.create_task(
                                self._drain_disconnected_job(
                                    gen_task, generation_id, prompt
                                )
                            )
                            return
                        await self._handle_client_msg_while_generating(
                            frame, generation_id, t_start,
                        )
                    video_path = await gen_task
                except Exception as exc:
                    log.error("Generation %s failed: %s", generation_id, exc)
                    try:
                        await self._send_json(
                            type="error",
                            error_code="generation_failed",
                            message=str(exc),
                        )
                    except Exception:
                        pass
                    return
            finally:
                ka_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ka_task

            t_gen = time.time() - t_start

            # ── stream binary chunks (spill-save if client vanishes mid-transfer)
            try:
                video_bytes = Path(video_path).read_bytes()
                total_bytes = len(video_bytes)
                sent        = 0
                while sent < total_bytes:
                    end   = min(sent + self.chunk_size, total_bytes)
                    await self.ws.send(video_bytes[sent:end])
                    sent = end
                self._dbg(f"→ {total_bytes} bytes sent in chunks")

                await self._send_json(
                    type="ltx2_segment_complete",
                    segment_idx=0,
                    total_segments=1,
                )

                await self._send_json(type="ltx2_stream_complete")
                self._dbg("→ ltx2_stream_complete")

                e2e_ms = (time.time() - self._t0) * 1000
                gen_ms = t_gen * 1000
                await self._send_json(
                    type="latency",
                    generation_ms=int(gen_ms),
                    e2e_ms=int(e2e_ms),
                )

                log.info(
                    "  ✓ generation %s  sent %d KB in %.1fs  (gen=%.1fs)  prompt=%r",
                    generation_id,
                    total_bytes // 1024,
                    e2e_ms / 1000,
                    t_gen,
                    prompt[:72],
                )
            except websockets.exceptions.ConnectionClosed:
                log.warning(
                    "  ⚠ client disconnected while streaming %s — spill-saving MP4",
                    generation_id[:8],
                )
                self._spill_copy(video_path, generation_id, prompt)
                return
            finally:
                _cleanup_temp_video(video_path)


# ── Server ─────────────────────────────────────────────────────────────────────

class VideoServer:
    """
    Manages the WebSocket server.

    Multiple clients may connect at once; ``GenerationScheduler`` ensures only one
    ``generate`` runs at a time.  Session traffic is not serialized.
    """

    def __init__(
        self,
        host:       str,
        port:       int,
        generator:  LocalVideoGenerator,
        verbose:    bool,
        chunk_size: int,
        spill_dir:  Path,
    ) -> None:
        self.host        = host
        self.port        = port
        self.generator   = generator
        self.verbose     = verbose
        self.chunk_size  = chunk_size
        self.spill_dir    = spill_dir
        self.scheduler   = GenerationScheduler()

    # ── per-connection callback ───────────────────────────────────────────────

    async def _handle_client(self, ws) -> None:
        addr = getattr(ws, "remote_address", "?")
        log.info("  ┌ connect  %s", addr)
        try:
            handler = RequestHandler(
                ws        = ws,
                generator = self.generator,
                verbose   = self.verbose,
                chunk_size= self.chunk_size,
                scheduler = self.scheduler,
                spill_dir = self.spill_dir,
            )
            await handler.handle()

        except Exception as exc:
            log.error("  ✗ error  %s  %s", addr, exc)
        finally:
            log.info("  └ disconnect  %s", addr)

    # ── start ────────────────────────────────────────────────────────────────

    async def serve(self) -> None:
        url = f"ws://{self.host}:{self.port}/ws"
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            max_size=200 * 1024 * 1024,
            ping_interval=30,
            ping_timeout=None,
            close_timeout=5,
        ):
            log.info("WebSocket server listening on %s", url)
            await asyncio.Future()  # run until interrupted


# ── CLI ────────────────────────────────────────────────────────────────────────

def _align_ltx2_spatial(n: int, align: int = LTX2_SPATIAL_ALIGN) -> int:
    """Round height or width to the nearest multiple of ``align`` (minimum ``align``)."""
    if n < align:
        return align
    lower = (n // align) * align
    upper = lower + align
    return lower if (n - lower) <= (upper - n) else upper


def _nearest_valid_frames(n: int) -> int:
    """Round n to the nearest value satisfying (frames - 1) % 8 == 0, min 9.

    LTX2 uses a temporal compression factor of 8, so valid frame counts are
    9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, … (i.e. 8k+1).
    """
    if n < 9:
        return 9
    remainder = (n - 1) % 8
    if remainder == 0:
        return n
    lower = n - remainder
    upper = lower + 8
    return lower if (n - lower) <= (upper - n) else upper


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="server",
        description=(
            "Local LTX-2.3 WebSocket server (MLX via ltx-2-mlx).\n"
            "Same JSON/binary protocol as videofentanyl ``--server`` expects."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Start with defaults (ws://0.0.0.0:8765/ws)
  python server.py

  # Custom port, larger resolution
  python server.py --port 9000 --height 720 --width 1280

  # Then use the client
  python videofentanyl.py --server ws://localhost:8765/ws \\
      --prompt "a fox running through snow"
""",
    )

    net = p.add_argument_group("network")
    net.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"bind address (default: {DEFAULT_HOST})",
    )
    net.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"port (default: {DEFAULT_PORT})",
    )

    mdl = p.add_argument_group("model")
    mdl.add_argument(
        "--model",
        default=_parser_default_model_arg(),
        help=(
            "HuggingFace MLX weights repo, local directory, or "
            f"'{MODEL_AUTO}' (default when flag omitted: {MODEL_AUTO}, "
            f"unless ${ENV_DEFAULT_MODEL} is set). "
            f"'{MODEL_AUTO}' maps RAM → bf16 (≥{RAM_GB_BF16_MIN:.0f} GiB), "
            f"q8 (≥{RAM_GB_Q8_MIN:.0f} GiB), q4 otherwise; "
            "Apple Silicon uses unified memory (no separate VRAM). "
            "Any explicit repo id or path skips auto-selection."
        ),
    )
    mdl.add_argument(
        "--model-dir", default=None, dest="model_dir", metavar="DIR",
        help=(
            "When --model is a HuggingFace repo id, store snapshot_download here "
            "(default: ./models/<org>__<name>/ or $VIDEOFENTANYL_MODELS/…). "
            "When --model is a shorthand or non-path label, may point at an existing "
            "local weights directory (see README). "
            "Missing Hub files are fetched automatically."
        ),
    )
    mdl.add_argument(
        "--lora",
        action="append",
        nargs=2,
        metavar=("LORA", "SCALE"),
        default=None,
        help=(
            "Optional global LoRA applied to all requests; repeatable. "
            "Format: --lora <local_path_or_hf_repo_or_url> <scale>. "
            f"Default comes from {ENV_DEFAULT_LORA}/{ENV_DEFAULT_LORA_SCALE} "
            f"or {ENV_DEFAULT_LORAS}."
        ),
    )
    mdl.add_argument(
        "--enable-lora",
        action="store_true",
        default=_env_bool(ENV_ENABLE_LORA, False),
        help=(
            "Enable global LoRA loading. Disabled by default. "
            f"Can also be enabled via {ENV_ENABLE_LORA}=1"
        ),
    )

    vid = p.add_argument_group("video")
    vid.add_argument(
        "--num-frames", type=int, default=DEFAULT_NUM_FRAMES, dest="num_frames",
        help=f"frames to generate — rounded to (8k+1) (default: {DEFAULT_NUM_FRAMES})",
    )
    vid.add_argument(
        "--height", type=int, default=DEFAULT_HEIGHT,
        help=(
            f"output height px (default: {DEFAULT_HEIGHT}); "
            f"rounded to nearest multiple of {LTX2_SPATIAL_ALIGN} for LTX2"
        ),
    )
    vid.add_argument(
        "--width", type=int, default=DEFAULT_WIDTH,
        help=(
            f"output width px (default: {DEFAULT_WIDTH}); "
            f"rounded to nearest multiple of {LTX2_SPATIAL_ALIGN} for LTX2"
        ),
    )
    vid.add_argument(
        "--fps", type=int, default=DEFAULT_FPS,
        help=(
            f"nominal fps (default: {DEFAULT_FPS}); MLX mux may use fixed rate — "
            "reserved for future pipeline options"
        ),
    )

    perf = p.add_argument_group("performance (MLX)")
    perf.add_argument(
        "--infer-steps", type=int, default=DEFAULT_INFER_STEPS, dest="infer_steps",
        metavar="N",
        help=(
            f"denoising steps for one-stage distilled sampling "
            f"(default: {DEFAULT_INFER_STEPS}, minimum 1)."
        ),
    )
    perf.add_argument(
        "--mlx-low-memory",
        action="store_true",
        dest="mlx_low_memory",
        help="pass low_memory=True to ltx-2-mlx (staged loads; slower, less RAM)",
    )

    up = p.add_argument_group(
        "native spatial upscale (ltx-2-mlx TwoStagePipeline)",
    )
    up.add_argument(
        "--upscale",
        action="store_true",
        help=(
            "For mode=generate (T2V and I2V): run ltx-2-mlx two-stage inference — "
            "stage 1 at half the requested H×W with the dev transformer + CFG, then "
            "native 2× latent upsampling (spatial_upscaler_x2_v1_1), then stage-2 "
            "refine at full resolution with the bundled distilled LoRA. "
            "Requires transformer-dev, spatial upscaler, and distilled LoRA files in "
            "the model directory (e.g. dgrauet/ltx-2.3-mlx-q8). "
            "Per-request LoRAs are ignored on this path. Does not use pixel scalers."
        ),
    )
    up.add_argument(
        "--stage1-steps",
        type=int,
        default=None,
        dest="stage1_steps",
        metavar="N",
        help=(
            "minimum two-stage stage-1 denoise steps (dev + CFG at half res); "
            "defaults to --infer-steps. Per-job num_steps is max(client, this floor). "
            "ltx-2-mlx often uses ~30 for best stage-1 quality — expect slower wall time."
        ),
    )
    up.add_argument(
        "--stage2-steps",
        type=int,
        default=None,
        dest="stage2_steps",
        metavar="N",
        help=(
            "two-stage stage-2 denoise steps (omit for ltx-2-mlx default STAGE_2_SIGMAS)."
        ),
    )
    up.add_argument(
        "--two-stage-cfg-scale",
        type=float,
        default=3.0,
        dest="two_stage_cfg_scale",
        help="CFG scale for two-stage stage 1 (video branch; default: 3.0).",
    )
    up.add_argument(
        "--two-stage-stg-scale",
        type=float,
        default=0.0,
        dest="two_stage_stg_scale",
        help="STG scale for two-stage stage 1 (default: 0.0).",
    )

    misc = p.add_argument_group("misc")
    misc.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, dest="chunk_size",
        help=f"binary chunk size in bytes (default: {DEFAULT_CHUNK_SIZE})",
    )
    misc.add_argument(
        "--spill-dir",
        type=str,
        default="fvserver_completed",
        metavar="DIR",
        help=(
            "directory where finished MP4s are copied if the client disconnects "
            "during generation or download (default: ./fvserver_completed)"
        ),
    )
    misc.add_argument(
        "--verbose", "-v", action="store_true",
        help="verbose per-connection logging",
    )
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if not (args.model or "").strip():
        parser.error("--model cannot be empty")
    resolved_model, model_auto_note = _resolve_model_cli_value(args.model)
    args.model = resolved_model
    if model_auto_note:
        log.info("Model selection: %s", model_auto_note)

    if args.infer_steps < 1:
        parser.error("--infer-steps must be >= 1")
    stage1_floor = (
        args.stage1_steps if args.stage1_steps is not None else args.infer_steps
    )
    if args.upscale and stage1_floor < 1:
        parser.error(
            "when --upscale is set, stage-1 floor (--stage1-steps or --infer-steps) "
            "must be >= 1",
        )
    stage2_arg: int | None = args.stage2_steps if args.upscale else None
    default_loras: list[tuple[str, float]] = []
    if args.enable_lora:
        default_loras = _default_loras_from_env()
    if args.lora:
        if not args.enable_lora:
            parser.error("--lora requires --enable-lora")
        for item in args.lora:
            path = str(item[0]).strip()
            try:
                scale = float(item[1])
            except (TypeError, ValueError):
                parser.error(f"invalid --lora scale: {item[1]!r}")
            if not path:
                parser.error("--lora path/repo cannot be empty")
            default_loras.append((path, scale))

    # ── Validate / adjust num_frames ─────────────────────────────────────────
    valid_frames = _nearest_valid_frames(args.num_frames)
    if valid_frames != args.num_frames:
        print(
            f"  [warn] --num-frames {args.num_frames} adjusted to {valid_frames} "
            f"(LTX2 requires (8k+1) frames: 9, 17, 25, … 97, …)",
            flush=True,
        )
        args.num_frames = valid_frames

    ah = _align_ltx2_spatial(args.height)
    aw = _align_ltx2_spatial(args.width)
    if ah != args.height or aw != args.width:
        print(
            f"  [warn] --height/--width {args.height}×{args.width} adjusted to "
            f"{ah}×{aw} (LTX2 requires multiples of {LTX2_SPATIAL_ALIGN})",
            flush=True,
        )
        args.height, args.width = ah, aw

    # ── Banner (preview paths only; Hub download happens in generator.load()) ─
    _preview = preview_mlx_weights_source(args.model, args.model_dir)
    if looks_like_hf_repo_id(args.model):
        _model_source = f"HuggingFace → {_preview}  ({args.model})"
    elif Path(_preview).is_dir():
        _model_source = f"local  ({_preview})"
    else:
        _model_source = f"{args.model}  (expected: {_preview})"
    print(f"\n{'═' * 60}")
    print(f"  LTX local server (MLX · ltx-2-mlx)")
    print(f"  Model    : {_model_source}")
    if model_auto_note:
        print(f"  RAM pick : {model_auto_note}")
    print(f"  Runtime  : Apple Silicon / MLX")
    print(f"  Endpoint : ws://{args.host}:{args.port}/ws")
    print(f"  Video    : {args.num_frames} frames @ "
          f"{args.height}×{args.width}  {args.fps} fps")
    print(f"  Denoise  : {args.infer_steps} steps  (use --infer-steps to tune)")
    if args.enable_lora and default_loras:
        print(f"  LoRA     : {len(default_loras)} global spec(s) (applied to all requests)")
    elif args.enable_lora:
        print("  LoRA     : enabled (no defaults configured)")
    else:
        print("  LoRA     : disabled (use --enable-lora)")
    print(f"  low_mem  : {args.mlx_low_memory}")
    if args.upscale:
        s2 = "default" if stage2_arg is None else str(stage2_arg)
        print(
            f"  upscale  : native 2× (TwoStage)  stage1_floor={stage1_floor}  "
            f"(per-job max(client num_steps, floor); dev+CFG is slower per step than distilled)  "
            f"stage2={s2}  cfg={args.two_stage_cfg_scale}  stg={args.two_stage_stg_scale}"
        )
    else:
        print("  upscale  : off (single-stage distilled unless client uses other modes)")
    print(f"{'═' * 60}\n")

    spill_dir = Path(args.spill_dir).expanduser().resolve()
    spill_dir.mkdir(parents=True, exist_ok=True)
    log.info("Disconnect spill directory: %s", spill_dir)

    # ── Resolve model/pipeline registry before WebSocket bind ─
    generator = LocalVideoGenerator(
        model               = args.model,
        num_frames          = args.num_frames,
        height              = args.height,
        width               = args.width,
        fps                 = float(args.fps),
        model_dir           = args.model_dir,
        inference_steps     = args.infer_steps,
        default_lora_specs  = default_loras,
        spill_dir           = spill_dir,
        low_memory          = args.mlx_low_memory,
        spatial_upscale     = args.upscale,
        stage1_steps        = stage1_floor,
        stage2_steps        = stage2_arg,
        two_stage_cfg_scale = args.two_stage_cfg_scale,
        two_stage_stg_scale = args.two_stage_stg_scale,
    )
    log.info("Loading weights before accepting connections …")
    generator.load()
    if default_loras:
        log.info("Resolving global LoRA(s) before accepting connections …")
        generator.ensure_default_loras_ready()
    log.info("Server ready — model path resolved; first used pipeline loads on demand.")

    # ── Start server ──────────────────────────────────────────────────────────
    server = VideoServer(
        host        = args.host,
        port        = args.port,
        generator   = generator,
        verbose     = args.verbose,
        chunk_size  = args.chunk_size,
        spill_dir   = spill_dir,
    )
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
    finally:
        log.info("Joining MLX worker thread (wait for any in-flight denoise to finish)…")
        generator.shutdown_mlx_executor(wait=True)


if __name__ == "__main__":
    main()
