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
  python videofentanylserver.py --port 9000 --num-frames 65 --height 480 --width 848

PROTOCOL — FastVideo (1080p) — server side
─────────────────────────────────────────────
  ← session_init_v2    (client handshake; store session state)
  ← simple_generate    (client triggers generation)
  → queue_status       (position in queue while waiting)
  → gpu_assigned       (device ready)
  → ltx2_stream_start
  → ltx2_segment_start
  → [binary chunks]    (raw MP4 video data)
  → ltx2_segment_complete
  → ltx2_stream_complete
  → latency            (timing info)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

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
DEFAULT_WIDTH          = 848
DEFAULT_FPS            = 24
DEFAULT_GUIDANCE_SCALE = 1.0   # LTX2-Distilled uses cfg=1.0
DEFAULT_CHUNK_SIZE     = 64 * 1024  # 64 KB per WebSocket binary message

# LTX2 frame requirement: num_frames must satisfy (frames - 1) % 8 == 0
# e.g. 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fvserver")


# ── Image helpers ──────────────────────────────────────────────────────────────

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
        guidance_scale: float,
        model_dir:      str | None = None,
    ) -> None:
        self.model          = model
        self.num_gpus       = num_gpus
        self.num_frames     = num_frames
        self.height         = height
        self.width          = width
        self.fps            = fps
        self.guidance_scale = guidance_scale
        self.model_dir      = model_dir
        self._generator     = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model weights.  Blocks until ready.  Call once at startup."""
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
            try:
                from fastvideo.configs.pipelines.ltx2 import LTX2T2VConfig
                extra_kwargs["pipeline_config"] = LTX2T2VConfig()
                log.info(
                    "Providing explicit LTX2T2VConfig to guard against "
                    "missing registry entry in installed FastVideo version."
                )
            except ImportError:
                log.warning(
                    "fastvideo.configs.pipelines.ltx2.LTX2T2VConfig not "
                    "importable; falling back to registry auto-detection."
                )

        self._generator = VideoGenerator.from_pretrained(
            local_path,
            num_gpus=self.num_gpus,
            # MPS-compatible offload settings: keep everything on-device
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
            image_encoder_cpu_offload=False,
            pin_cpu_memory=False,
            **extra_kwargs,
        )
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
        prompt:       str,
        image_data:   dict | None = None,
        seed:         int = 1024,
        num_frames:   int | None = None,
        height:       int | None = None,
        width:        int | None = None,
    ) -> str:
        """
        Generate a video asynchronously (runs `_generate_sync` in a thread
        pool so the event loop stays responsive).

        Returns the path of the saved MP4 file.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_sync,
            prompt,
            image_data,
            seed,
            num_frames or self.num_frames,
            height     or self.height,
            width      or self.width,
        )

    def _generate_sync(
        self,
        prompt:     str,
        image_data: dict | None,
        seed:       int,
        num_frames: int,
        height:     int,
        width:      int,
    ) -> str:
        """Synchronous generation — executed in a thread-pool worker."""
        from fastvideo.api.schema import (
            GenerationRequest,
            InputConfig,
            OutputConfig,
            SamplingConfig,
        )

        tmp_image: str | None = None
        tmpdir = tempfile.mkdtemp(prefix="fvserver_out_")
        out_path = os.path.join(tmpdir, "output.mp4")

        try:
            if image_data:
                tmp_image = _decode_initial_image(image_data)

            request = GenerationRequest(
                prompt=prompt,
                inputs=InputConfig(image_path=tmp_image),
                sampling=SamplingConfig(
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

            result = self._generator.generate(request=request)

            # generate() may return a list when multiple prompts are given
            if isinstance(result, list):
                result = result[0]

            video_path: str | None = getattr(result, "video_path", None)
            if not video_path or not os.path.exists(video_path):
                # Fall back to the path we requested
                video_path = out_path
            if not os.path.exists(video_path):
                raise RuntimeError(
                    f"Generation completed but output file not found: {video_path}"
                )
            return video_path

        finally:
            if tmp_image:
                try:
                    os.unlink(tmp_image)
                except OSError:
                    pass


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
    ) -> None:
        self.ws         = ws
        self.generator  = generator
        self.verbose    = verbose
        self.chunk_size = chunk_size
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
        if not prompt:
            await self._send_json(
                type="error",
                error_code="invalid_prompt",
                message="Empty prompt",
            )
            return

        # Resolve initial image: prefer from simple_generate, fall back to
        # session_init_v2 (which is how the 1080p client sends i2v images)
        initial_image: dict | None = (
            msg.get("initial_image") or self._session.get("initial_image")
        )

        t_start = time.time()

        # ── gpu_assigned ──────────────────────────────────────────────────────
        await self._send_json(
            type="gpu_assigned",
            gpu_id="mps:0",
            session_timeout=600,
        )
        self._dbg("→ gpu_assigned")

        # ── stream_start ──────────────────────────────────────────────────────
        await self._send_json(
            type="ltx2_stream_start",
            total_segments=1,
            stream_mode="single",
        )
        self._dbg("→ ltx2_stream_start")

        # ── segment_start ─────────────────────────────────────────────────────
        await self._send_json(
            type="ltx2_segment_start",
            segment_idx=0,
            total_segments=1,
        )
        self._dbg("→ ltx2_segment_start (segment 0/1)")

        # ── generate ──────────────────────────────────────────────────────────
        video_path: str | None = None
        try:
            video_path = await self.generator.generate(
                prompt=prompt,
                image_data=initial_image,
            )
        except Exception as exc:
            log.error("Generation failed: %s", exc)
            await self._send_json(
                type="error",
                error_code="generation_failed",
                message=str(exc),
            )
            return

        t_gen = time.time() - t_start

        # ── stream binary chunks ───────────────────────────────────────────────
        video_bytes = Path(video_path).read_bytes()
        total_bytes = len(video_bytes)
        sent        = 0
        while sent < total_bytes:
            end   = min(sent + self.chunk_size, total_bytes)
            await self.ws.send(video_bytes[sent:end])
            sent = end
        self._dbg(f"→ {total_bytes} bytes sent in chunks")

        # ── segment_complete ──────────────────────────────────────────────────
        await self._send_json(
            type="ltx2_segment_complete",
            segment_idx=0,
            total_segments=1,
        )

        # ── stream_complete ───────────────────────────────────────────────────
        await self._send_json(type="ltx2_stream_complete")
        self._dbg("→ ltx2_stream_complete")

        # ── latency ───────────────────────────────────────────────────────────
        e2e_ms = (time.time() - self._t0) * 1000
        gen_ms = t_gen * 1000
        await self._send_json(
            type="latency",
            generation_ms=int(gen_ms),
            e2e_ms=int(e2e_ms),
        )

        log.info(
            "  ✓ sent %d KB in %.1fs  (gen=%.1fs)  prompt=%r",
            total_bytes // 1024,
            e2e_ms / 1000,
            t_gen,
            prompt[:72],
        )

        # Clean up temp file / directory
        try:
            p = Path(video_path)
            p.unlink(missing_ok=True)
            try:
                p.parent.rmdir()
            except OSError:
                pass
        except OSError:
            pass


# ── Server ─────────────────────────────────────────────────────────────────────

class VideoServer:
    """
    Manages the WebSocket server and request queue.

    Requests are processed one at a time (sequential) because MPS has a single
    compute device.  Clients that connect while a generation is running receive
    a queue_status message and wait for the lock to be released.
    """

    def __init__(
        self,
        host:       str,
        port:       int,
        generator:  LocalVideoGenerator,
        verbose:    bool,
        chunk_size: int,
    ) -> None:
        self.host        = host
        self.port        = port
        self.generator   = generator
        self.verbose     = verbose
        self.chunk_size  = chunk_size
        self._lock       = asyncio.Lock()
        self._queue_len  = 0

    # ── per-connection callback ───────────────────────────────────────────────

    async def _handle_client(self, ws) -> None:
        addr = getattr(ws, "remote_address", "?")
        log.info("  ┌ connect  %s", addr)
        try:
            # Tell client its queue position while it waits for the lock
            if self._lock.locked():
                self._queue_len += 1
                pos = self._queue_len
                try:
                    await ws.send(json.dumps({
                        "type":           "queue_status",
                        "position":       pos,
                        "available_gpus": 0,
                        "total_gpus":     1,
                    }))
                except Exception:
                    pass

            async with self._lock:
                if self._queue_len > 0:
                    self._queue_len -= 1
                handler = RequestHandler(
                    ws        = ws,
                    generator = self.generator,
                    verbose   = self.verbose,
                    chunk_size= self.chunk_size,
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
            ping_timeout=20,
            close_timeout=5,
        ):
            log.info("WebSocket server listening on %s", url)
            await asyncio.Future()  # run until interrupted


# ── CLI ────────────────────────────────────────────────────────────────────────

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
        help=f"output height px (default: {DEFAULT_HEIGHT})",
    )
    vid.add_argument(
        "--width", type=int, default=DEFAULT_WIDTH,
        help=f"output width px (default: {DEFAULT_WIDTH})",
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

    misc = p.add_argument_group("misc")
    misc.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, dest="chunk_size",
        help=f"binary chunk size in bytes (default: {DEFAULT_CHUNK_SIZE})",
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

    # ── Validate / adjust num_frames ─────────────────────────────────────────
    valid_frames = _nearest_valid_frames(args.num_frames)
    if valid_frames != args.num_frames:
        print(
            f"  [warn] --num-frames {args.num_frames} adjusted to {valid_frames} "
            f"(LTX2 requires (8k+1) frames: 9, 17, 25, … 97, …)",
            flush=True,
        )
        args.num_frames = valid_frames

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
    print(f"{'═' * 60}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    generator = LocalVideoGenerator(
        model          = args.model,
        num_gpus       = args.num_gpus,
        num_frames     = args.num_frames,
        height         = args.height,
        width          = args.width,
        fps            = args.fps,
        guidance_scale = args.guidance_scale,
        model_dir      = args.model_dir,
    )
    generator.load()

    # ── Start server ──────────────────────────────────────────────────────────
    server = VideoServer(
        host        = args.host,
        port        = args.port,
        generator   = generator,
        verbose     = args.verbose,
        chunk_size  = args.chunk_size,
    )
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("\n\nServer stopped.")


if __name__ == "__main__":
    main()
