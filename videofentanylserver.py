#!/usr/bin/env python3
"""
videofentanylserver.py — Local FastVideo/LTX2 WebSocket Server
==============================================================
Runs a local WebSocket server that implements the same protocol as
wss://1080p.fastvideo.org/ws, generating videos with FastVideo's
LTX2-Distilled model on Apple MPS (or CPU as fallback).

USAGE
─────
  # Start the server (default: ws://0.0.0.0:8765/ws)
  python videofentanylserver.py

  # Then run the client pointing at the local server
  python videofentanyl.py --server ws://localhost:8765/ws \\
      --prompt "a fox running through snow"

  # Custom device settings
  python videofentanylserver.py --port 9000 --num-frames 65 --height 480 --width 832

PROTOCOL — FastVideo (1080p) — server side
─────────────────────────────────────────────
  ← session_init_v2    (client handshake; store session state; may include
                        ``initial_image`` / ``initialImage`` for i2v or autocontinue)
  ← simple_generate    (client triggers generation; optional ``seed``, ``num_frames``,
                        ``height``, ``width`` override server defaults; same image keys
                        as session_init for start / continuation frames)
  → queue_status       (position in queue while waiting; active_generation_id)
  → gpu_assigned       (device ready; includes generation_id for this job)
  → ltx2_stream_start
  → ltx2_segment_start
  → generation_keepalive (periodic JSON while the model runs; elapsed_s / phase /
    generation_id / optional model_progress from worker denoise steps)
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
import base64
import functools
import json
import logging
import mimetypes
import os
import queue
import re
import shutil
import sys
import subprocess
import tempfile
import threading
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

# ── Constants / defaults ───────────────────────────────────────────────────────

DEFAULT_HOST           = "0.0.0.0"
DEFAULT_PORT           = 8765
DEFAULT_MODEL          = "FastVideo/LTX2-Distilled-Diffusers"
DEFAULT_NUM_GPUS       = 1
DEFAULT_NUM_FRAMES     = 97    # ~4 s @ 24 fps; LTX requires (8k+1) frames: 9, 17, 25, … 97, 105, …
DEFAULT_HEIGHT         = 480
# LTX2 latent prep requires height and width divisible by 32 (848 is invalid).
DEFAULT_WIDTH          = 832
DEFAULT_FPS            = 24
DEFAULT_GUIDANCE_SCALE = 1.0   # LTX2-Distilled uses cfg=1.0
# Distilled LTX2 is meant for few-step sampling; FastVideo's schema default (50)
# is oriented at CUDA servers and is far too slow on Apple MPS (~tens of s/step).
DEFAULT_INFER_STEPS    = 12
DEFAULT_CHUNK_SIZE     = 64 * 1024  # 64 KB per WebSocket binary message
# JSON keepalives during long MPS runs so clients can use an idle timeout instead
# of a short wall-clock cap (see videofentanyl.py).
GENERATION_KEEPALIVE_INTERVAL_S = 15.0
# Must match ``FV_PROGRESS_JSON`` lines emitted by patched FastVideo (e.g. ltx2_denoising).
FV_PROGRESS_LOG_TAG = "FV_PROGRESS_JSON"

# LTX2 frame requirement: num_frames must satisfy (frames - 1) % 8 == 0
# e.g. 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129
LTX2_SPATIAL_ALIGN = 32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fvserver")


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


def _make_fv_mp_queue() -> Any:
    """Multiprocessing queue compatible with FastVideo worker ``log_queue``."""
    try:
        from fastvideo.utils import get_mp_context

        return get_mp_context().Queue()
    except Exception:
        import multiprocessing as mp

        return mp.Queue()


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


def _apply_pytorch_mps_runtime_tuning() -> None:
    """Best-effort runtime knobs for Metal / MPS (no-op if torch or MPS missing)."""
    try:
        import torch
    except ImportError:
        return
    log.info("PyTorch %s", torch.__version__)
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        return
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # Prefer bf16 accumulation where the build supports it (reduces upcasts).
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and hasattr(mps, "allow_fp16_accumulation_in_bf16"):
        try:
            mps.allow_fp16_accumulation_in_bf16 = True  # type: ignore[attr-defined]
        except Exception:
            pass


# ── Image helpers ──────────────────────────────────────────────────────────────

def _resolve_initial_image_payload(msg: dict, session: dict) -> dict | None:
    """
    Pick the first non-empty image payload dict from the generate message, then session.

    Accepts several keys so autocontinue / custom clients match FastVideo-style
    ``inputs.image_path`` flows once decoded to a temp file for ``GenerationRequest``.
    """
    keys = (
        "initial_image",
        "initialImage",
        "continuation_frame",
        "continuationFrame",
        "start_image",
        "startImage",
    )
    for source in (msg, session):
        for key in keys:
            raw = source.get(key)
            if isinstance(raw, dict) and raw:
                return raw
    return None


def _decode_initial_image(image_data: dict) -> str:
    """
    Decode an initial_image payload dict (name/mime_type/data_url) to a temp
    file and return its path.  Caller is responsible for deleting the file.
    """
    data_url: str = image_data.get("data_url", "")
    if data_url.startswith("data:"):
        header, encoded = data_url.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
    else:
        mime    = image_data.get("mime_type", "image/jpeg")
        encoded = data_url

    ext = mimetypes.guess_extension(mime) or ".jpg"
    if ext == ".jpe":
        ext = ".jpg"

    fd, path = tempfile.mkstemp(suffix=ext, prefix="fvserver_img_")
    with os.fdopen(fd, "wb") as f:
        f.write(base64.b64decode(encoded))
    return path


# ── FastVideo wrapper ──────────────────────────────────────────────────────────

class LocalVideoGenerator:
    """
    Thin wrapper around fastvideo.VideoGenerator for single-device (MPS/CPU)
    inference.  The model is loaded once at startup and reused across requests.
    """

    def __init__(
        self,
        model:          str,
        num_gpus:       int,
        num_frames:     int,
        height:         int,
        width:          int,
        fps:            int,
        guidance_scale:           float,
        model_dir:                str | None = None,
        inference_steps:        int = DEFAULT_INFER_STEPS,
        enable_stage_verification: bool = False,
        enable_torch_compile:   bool = False,
        spill_dir:                Path | None = None,
        mac_ipc_safe_offload:   bool = False,
    ) -> None:
        self.model          = model
        self.num_gpus       = num_gpus
        self.num_frames     = num_frames
        self.height         = height
        self.width          = width
        self.fps            = fps
        self.guidance_scale = guidance_scale
        self.model_dir      = model_dir
        self.inference_steps = inference_steps
        self.enable_stage_verification = enable_stage_verification
        self.enable_torch_compile = enable_torch_compile
        self.spill_dir              = spill_dir
        self.mac_ipc_safe_offload   = mac_ipc_safe_offload
        self._generator             = None
        self._fv_log_queue          = _make_fv_mp_queue()
        self._progress_lock         = threading.Lock()
        self._model_progress: dict[str, Any] = {}
        self._log_drain_thread: threading.Thread | None = None
        self._log_drain_stop        = threading.Event()

    # ── worker log queue → model_progress for WebSocket keepalives ────────────

    def _reset_model_progress(self) -> None:
        with self._progress_lock:
            self._model_progress = {}

    def model_progress_for_ws(self) -> dict[str, Any] | None:
        """Latest structured progress from FastVideo workers (denoise steps, ETA)."""
        with self._progress_lock:
            if not self._model_progress.get("stage"):
                return None
            return dict(self._model_progress)

    def _merge_fv_progress_payload(self, payload: dict[str, Any]) -> None:
        stage = payload.get("stage")
        if not stage:
            return
        step = payload.get("step")
        total = payload.get("total")
        pct: float | None = None
        try:
            si, st = int(step), int(total)
            if st > 0:
                pct = round(100.0 * min(si, st) / st, 1)
        except (TypeError, ValueError):
            pass
        merged = {
            "stage": stage,
            "step": step,
            "total": total,
            "pct": pct,
            "elapsed_s": payload.get("elapsed_s"),
            "last_step_s": payload.get("last_step_s"),
            "avg_step_s": payload.get("avg_step_s"),
            "eta_s": payload.get("eta_s"),
        }
        with self._progress_lock:
            self._model_progress = merged

    def _fv_log_drain_loop(self) -> None:
        while not self._log_drain_stop.is_set():
            try:
                rec = self._fv_log_queue.get(timeout=0.35)
            except queue.Empty:
                continue
            except Exception:
                continue
            try:
                msg = rec.getMessage()
            except Exception:
                continue
            tag = FV_PROGRESS_LOG_TAG
            if tag not in msg:
                continue
            try:
                blob = msg.split(tag, 1)[1].strip()
                payload = json.loads(blob)
            except (json.JSONDecodeError, IndexError, TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                self._merge_fv_progress_payload(payload)

    def _ensure_fv_log_drain_thread(self) -> None:
        if self._log_drain_thread is not None and self._log_drain_thread.is_alive():
            return
        self._log_drain_stop.clear()
        self._log_drain_thread = threading.Thread(
            target=self._fv_log_drain_loop,
            name="fvserver-fv-log-drain",
            daemon=True,
        )
        self._log_drain_thread.start()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model weights.  Blocks until ready.  Safe to call more than once."""
        if self._generator is not None:
            return

        from fastvideo import VideoGenerator

        local_path = self._resolve_model_path()
        log.info("Loading model from %s …", local_path)

        # Older FastVideo releases may not include recent model IDs in the
        # pipeline registry.  When the registry returns None it falls back to
        # a generic PipelineConfig that lacks LTX2-specific components
        # (LTX2GemmaConfig text encoder, LTX2VAEConfig VAE, etc.), causing a
        # crash during model loading.  Explicitly supplying LTX2T2VConfig here
        # ensures the correct pipeline config is used regardless of which
        # FastVideo version is installed.  In newer releases where the model IS
        # registered, the passed instance simply overrides the registry result
        # with an identical config — a safe no-op.
        extra_kwargs: dict = {}
        _model_lower = (self.model or local_path).lower()
        if "ltx2" in _model_lower or "ltx-2" in _model_lower:
            _ltx2_config_cls = None
            _ltx2_import_err: ImportError | None = None

            # Try the direct submodule path first (FastVideo ≥ LTX2 era).
            try:
                from fastvideo.configs.pipelines.ltx2 import LTX2T2VConfig as _LTX2T2VConfig
                _ltx2_config_cls = _LTX2T2VConfig
            except ImportError as _err:
                _ltx2_import_err = _err
                # Fall back to the package __init__ re-export in case the
                # symbol is available there even though the direct path failed.
                try:
                    from fastvideo.configs.pipelines import LTX2T2VConfig as _LTX2T2VConfig2
                    _ltx2_config_cls = _LTX2T2VConfig2
                except ImportError:
                    pass  # both paths failed; handled below

            if _ltx2_config_cls is not None:
                extra_kwargs["pipeline_config"] = _ltx2_config_cls()
                log.info(
                    "Providing explicit LTX2T2VConfig to guard against "
                    "missing registry entry in installed FastVideo version."
                )
            else:
                # Neither import path worked.  Log the root cause so the user
                # can see exactly which module is missing, then abort with
                # clear upgrade instructions.
                log.warning(
                    "Could not import LTX2T2VConfig: %s", _ltx2_import_err,
                    exc_info=_ltx2_import_err,
                )
                try:
                    import fastvideo as _fv
                    installed_ver = getattr(_fv, "__version__", "unknown")
                except ImportError:
                    installed_ver = "not installed"
                _installer = Path(__file__).resolve().parent / "scripts" / "fastvideo_install"
                _install_hint = (
                    f"  python {_installer}\n"
                    "     (initializes submodule, applies macOS Triton workaround, "
                    "installs editable)\n"
                    "  # Standalone clone at DIR after git pull:\n"
                    f"  python {_installer} --no-submodule --path DIR\n\n"
                )
                raise RuntimeError(
                    f"Failed to load LTX2T2VConfig from the installed FastVideo "
                    f"({installed_ver}).  See the ImportError logged above for "
                    "the root cause.\n\n"
                    "The installed FastVideo is likely too old and does not yet "
                    "include LTX2 support.  Reinstall from a fresh checkout using "
                    "the helper script from this repo:\n\n"
                    f"{_install_hint}"
                    "To confirm the correct version is active afterwards:\n\n"
                    "  python -c \"import fastvideo; print(fastvideo.__version__)\"\n"
                    "  python -c \"from fastvideo.configs.pipelines.ltx2 import "
                    "LTX2T2VConfig; print('OK')\""
                ) from _ltx2_import_err

        _heavy = self.mac_ipc_safe_offload
        if _heavy:
            log.warning(
                "mac-ipc-safe-offload: DiT/VAE/text-encoder CPU offload enabled "
                "(much slower; only use if FastVideo workers fail with "
                "_share_filename_ / MPS over multiprocessing pipes)."
            )

        # log_queue here (not on ``generate()``): workers inherit it at spawn.
        # Passing a Queue through ``set_log_queue`` RPC pickles it and raises
        # "Queue objects should only be shared between processes through inheritance".
        self._generator = VideoGenerator.from_pretrained(
            local_path,
            num_gpus=self.num_gpus,
            log_queue=self._fv_log_queue,
            # Default: no heavy CPU offload (MPS throughput). Opt-in via
            # --mac-ipc-safe-offload if worker IPC requires CPU-resident tensors.
            dit_cpu_offload=_heavy,
            vae_cpu_offload=_heavy,
            text_encoder_cpu_offload=_heavy,
            image_encoder_cpu_offload=False,
            pin_cpu_memory=False,
            # Throughput: skip per-stage verification (FastVideo default is True).
            enable_stage_verification=self.enable_stage_verification,
            enable_torch_compile=self.enable_torch_compile,
            **extra_kwargs,
        )
        self._ensure_fv_log_drain_thread()
        log.info("Model loaded ✓")

    def _resolve_model_path(self) -> str:
        """
        Return a local filesystem path for the model.

        * If ``self.model`` is already an existing directory, use it as-is.
        * Otherwise treat it as a HuggingFace repo ID, download/cache it with
          ``huggingface_hub.snapshot_download``, and return the local cache dir.
          Passing a ``model_dir`` overrides the default HF cache location.
        """
        if Path(self.model).exists():
            log.info("Using local model directory: %s", self.model)
            return self.model

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            log.error(
                "huggingface_hub is required to download models. "
                "Install it with:  pip install huggingface_hub"
            )
            raise

        log.info(
            "Downloading model '%s' from HuggingFace (this may take a while on first run) …",
            self.model,
        )
        kwargs: dict = dict(repo_id=self.model)
        if self.model_dir:
            kwargs["local_dir"] = self.model_dir
            log.info("  → saving to: %s", self.model_dir)
        else:
            log.info("  → caching in default HuggingFace cache (~/.cache/huggingface/hub)")

        local_path = snapshot_download(**kwargs)
        log.info("Model available at: %s", local_path)
        return local_path

    # ── generation ────────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt:           str,
        image_data:       dict | None = None,
        seed:             int = 1024,
        num_frames:       int | None = None,
        height:           int | None = None,
        width:            int | None = None,
        negative_prompt:  str = "",
        *,
        job_id:           str | None = None,
    ) -> str:
        """
        Generate a video asynchronously (runs `_generate_sync` in a thread
        pool so the event loop stays responsive).

        ``job_id`` — optional generation UUID prefix for temp dirs and spill files.

        Returns the path of the saved MP4 file.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self._generate_sync,
                prompt,
                image_data,
                seed,
                num_frames or self.num_frames,
                height     or self.height,
                width      or self.width,
                negative_prompt,
                job_id,
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
        """Copy any usable MP4 from the workdir into ``spill_dir`` after failures."""
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

    def _generate_sync(
        self,
        prompt:           str,
        image_data:       dict | None,
        seed:             int,
        num_frames:       int,
        height:           int,
        width:            int,
        negative_prompt:  str,
        job_id:           str | None = None,
    ) -> str:
        """Synchronous generation — executed in a thread-pool worker."""
        self.load()

        ah = _align_ltx2_spatial(height)
        aw = _align_ltx2_spatial(width)
        if ah != height or aw != width:
            log.warning(
                "LTX2 requires H×W divisible by %s; adjusted %s×%s → %s×%s",
                LTX2_SPATIAL_ALIGN,
                height,
                width,
                ah,
                aw,
            )
            height, width = ah, aw

        from fastvideo.api.schema import (
            GenerationRequest,
            InputConfig,
            OutputConfig,
            SamplingConfig,
        )

        tmp_image: str | None = None
        prefix = f"fv_{job_id[:8]}_" if job_id else "fvserver_out_"
        tmpdir = tempfile.mkdtemp(prefix=prefix)
        out_path = os.path.join(tmpdir, "output.mp4")
        retained_tmpdir = False
        self._reset_model_progress()

        try:
            if isinstance(image_data, dict) and image_data:
                tmp_image = _decode_initial_image(image_data)

            # LTX2 text_encoding asserts isinstance(negative_prompt, str); schema default is None.
            request = GenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                inputs=InputConfig(image_path=tmp_image),
                sampling=SamplingConfig(
                    num_inference_steps=self.inference_steps,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    fps=self.fps,
                    guidance_scale=self.guidance_scale,
                    seed=seed,
                ),
                output=OutputConfig(
                    output_path=out_path,
                    save_video=True,
                    return_frames=False,
                ),
            )

            try:
                result = self._generator.generate(request=request)
            except BaseException:
                self._salvage_mp4_to_spill(
                    tmpdir, out_path, job_id, prompt, "ENCODE_FAIL",
                )
                raise

            # generate() may return a list when multiple prompts are given
            if isinstance(result, list):
                result = result[0]

            video_path: str | None = getattr(result, "video_path", None)
            if not video_path or not os.path.exists(video_path):
                # Fall back to the path we requested
                video_path = out_path
            if not os.path.exists(video_path):
                self._salvage_mp4_to_spill(
                    tmpdir, out_path, job_id, prompt, "MISSING_OUTPUT",
                )
                raise RuntimeError(
                    f"Generation completed but output file not found: {video_path}"
                )
            retained_tmpdir = True
            return video_path

        finally:
            if tmp_image:
                try:
                    os.unlink(tmp_image)
                except OSError:
                    pass
            if not retained_tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)


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
        initial_image: dict | None = _resolve_initial_image_payload(msg, self._session)

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

        async def _notify_queue(**kwargs: Any) -> None:
            await self._send_json(**kwargs)

        async with self.scheduler.generation_slot(_notify_queue) as generation_id:
            t_start = time.time()
            log.info(
                "  ▶ generation %s  prompt=%r  start_image=%s  seed=%s  %s×%s  frames=%s",
                generation_id,
                prompt[:72],
                "yes" if initial_image else "no",
                gen_seed,
                gen_height,
                gen_width,
                gen_num_frames,
            )

            # ── gpu_assigned ──────────────────────────────────────────────────
            await self._send_json(
                type="gpu_assigned",
                gpu_id="mps:0",
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
                        negative_prompt=negative_prompt,
                        seed=gen_seed,
                        num_frames=gen_num_frames,
                        height=gen_height,
                        width=gen_width,
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
                                "waiting for encode then spill-saving MP4",
                                generation_id[:8],
                            )
                            try:
                                video_path = await gen_task
                                self._spill_copy(video_path, generation_id, prompt)
                            except Exception as gen_exc:
                                log.error(
                                    "  ✗ generation after client disconnect failed "
                                    "(check %s for spill-salvaged *_ENCODE_FAIL*.mp4): %s",
                                    self.spill_dir,
                                    gen_exc,
                                )
                            else:
                                _cleanup_temp_video(video_path)
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
        prog="videofentanylserver",
        description=(
            "Local FastVideo/LTX2 WebSocket server.\n"
            "Implements the same protocol as wss://1080p.fastvideo.org/ws."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Start with defaults (ws://0.0.0.0:8765/ws)
  python videofentanylserver.py

  # Custom port, larger resolution
  python videofentanylserver.py --port 9000 --height 720 --width 1280

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
        "--model", default=DEFAULT_MODEL,
        help=(
            f"HuggingFace model ID or path to a local model directory "
            f"(default: {DEFAULT_MODEL}). "
            f"Pass a local path (e.g. ./models/LTX2-Distilled-Diffusers) to skip "
            f"the HuggingFace download and load weights directly from disk."
        ),
    )
    mdl.add_argument(
        "--model-dir", default=None, dest="model_dir", metavar="DIR",
        help=(
            "Directory where the model will be downloaded/cached when --model is a "
            "HuggingFace model ID. If omitted the default HuggingFace cache is used "
            "(~/.cache/huggingface/hub). Ignored when --model is already a local path."
        ),
    )
    mdl.add_argument(
        "--num-gpus", type=int, default=DEFAULT_NUM_GPUS, dest="num_gpus",
        help=f"number of devices (default: {DEFAULT_NUM_GPUS})",
    )
    mdl.add_argument(
        "--attention-backend", default=None, dest="attention_backend",
        metavar="BACKEND",
        help="attention backend: TORCH_SDPA (MPS/CPU default) or FLASH_ATTN (CUDA)",
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
        help=f"frames per second (default: {DEFAULT_FPS})",
    )
    vid.add_argument(
        "--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE,
        dest="guidance_scale",
        help=f"CFG guidance scale (default: {DEFAULT_GUIDANCE_SCALE})",
    )

    perf = p.add_argument_group("performance (especially Apple MPS)")
    perf.add_argument(
        "--infer-steps", type=int, default=DEFAULT_INFER_STEPS, dest="infer_steps",
        metavar="N",
        help=(
            f"denoising steps (default: {DEFAULT_INFER_STEPS}, minimum 1). "
            "Distilled LTX2 is intended for low step counts; 50 matches FastVideo's "
            "generic default but is often impractical on MPS — raise for quality, "
            "lower for speed."
        ),
    )
    perf.add_argument(
        "--stage-verification", action="store_true", dest="stage_verification",
        help="enable FastVideo per-stage checks (slower; default off for throughput)",
    )
    perf.add_argument(
        "--torch-compile", action="store_true", dest="torch_compile",
        help="enable torch.compile in FastVideo (PyTorch 2.4+; experimental on MPS)",
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
        "--mac-ipc-safe-offload",
        action="store_true",
        help=(
            "opt-in: enable DiT/VAE/text-encoder CPU offload (very slow on MPS).  "
            "Only if FastVideo multiprocessing workers error with CPU-only tensor "
            "sharing; default keeps models on device for normal speed."
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

    # ── Attention backend ─────────────────────────────────────────────────────
    # For Apple MPS, TORCH_SDPA (PyTorch native scaled dot-product attention)
    # is required; FLASH_ATTN is CUDA-only.
    if args.attention_backend:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.attention_backend
    elif "FASTVIDEO_ATTENTION_BACKEND" not in os.environ:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"

    _apply_pytorch_mps_runtime_tuning()

    if args.infer_steps < 1:
        parser.error("--infer-steps must be >= 1")

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

    # ── Banner ────────────────────────────────────────────────────────────────
    _model_is_local = Path(args.model).exists()
    if _model_is_local:
        _model_source = f"local  ({args.model})"
    elif args.model_dir:
        _model_source = f"HuggingFace → {args.model_dir}  ({args.model})"
    else:
        _model_source = f"HuggingFace cache  ({args.model})"
    print(f"\n{'═' * 60}")
    print(f"  FastVideo Local Server  (videofentanylserver)")
    print(f"  Model    : {_model_source}")
    print(f"  Device   : Apple MPS  (FASTVIDEO_ATTENTION_BACKEND="
          f"{os.environ.get('FASTVIDEO_ATTENTION_BACKEND')})")
    print(f"  Endpoint : ws://{args.host}:{args.port}/ws")
    print(f"  Video    : {args.num_frames} frames @ "
          f"{args.height}×{args.width}  {args.fps} fps")
    print(f"  CFG      : {args.guidance_scale}")
    print(f"  Denoise  : {args.infer_steps} steps  (use --infer-steps to tune)")
    print(f"{'═' * 60}\n")

    spill_dir = Path(args.spill_dir).expanduser().resolve()
    spill_dir.mkdir(parents=True, exist_ok=True)
    log.info("Disconnect spill directory: %s", spill_dir)

    # ── Load model (before WebSocket bind — first client is never a cold-start) ─
    generator = LocalVideoGenerator(
        model                       = args.model,
        num_gpus                    = args.num_gpus,
        num_frames                  = args.num_frames,
        height                      = args.height,
        width                       = args.width,
        fps                         = args.fps,
        guidance_scale              = args.guidance_scale,
        model_dir                   = args.model_dir,
        inference_steps             = args.infer_steps,
        enable_stage_verification   = args.stage_verification,
        enable_torch_compile        = args.torch_compile,
        spill_dir                   = spill_dir,
        mac_ipc_safe_offload        = args.mac_ipc_safe_offload,
    )
    log.info("Loading weights before accepting connections …")
    generator.load()
    log.info("Server ready — model is in memory.")

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


if __name__ == "__main__":
    main()
