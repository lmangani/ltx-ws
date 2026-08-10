#!/usr/bin/env python3
"""
videofentanyl.py — Unified LTX (local MLX) / Dreamverse Queue Manager
======================================================================
Supports two generation backends selectable via --mode:

  ltx  (default)        **requires** ``--server ws://…/ws`` — local ``server.py``
                        (ltx-2-mlx on Apple Silicon), single-segment clips
  dreamverse            wss://dreamverse.fastvideo.org/ws
                        Long-form videos (~30s, 6 segments), GPT-expanded prompt

USAGE
─────
  # Local LTX (MLX) — server must be running (see README)
  python videofentanyl.py --server ws://localhost:8765/ws --prompt "a fox in snow"

  # Dreamverse long-form
  python videofentanyl.py --mode dreamverse --prompt "a dog learns to fly"

  # 5 videos from the same prompt (ltx + --server)
  python videofentanyl.py --server ws://localhost:8765/ws \\
      --prompt "sunset over the ocean" --count 5

  # Multiple prompts (dreamverse)
  python videofentanyl.py --mode dreamverse \\
      --prompt "dog on skateboard" --prompt "volcano at night"

  # Image-to-video (ltx + local server)
  python videofentanyl.py --server ws://localhost:8765/ws \\
      --prompt "the scene comes alive" --image photo.jpg

  # AI prompt enhancement (ltx: forwarded to server if supported)
  python videofentanyl.py --server ws://localhost:8765/ws \\
      --prompt "girl walking in rain" --enhance

  # Skip GPT expansion (dreamverse)
  python videofentanyl.py --mode dreamverse --prompt "detailed scene…" --no-enhance

  # Dry-run, verbose, custom output
  python videofentanyl.py --prompt "test" --count 3 --dry-run --verbose \\
      --output-dir ./videos --prefix clip

PROTOCOL — LTX local (single-segment)
────────────────────────────────────
  connect → session_init_v2  (handshake)
          → simple_generate  (trigger generation)
          → generation_status  (optional; client may poll while waiting)
          ← gpu_assigned
          ← ltx2_segment_start / ltx2_segment_complete
          ← generation_keepalive  (optional; local server; may include model_progress)
          ← generation_status_ack  (local server reply; may include model_progress)
          ← [binary chunks]
          ← ltx2_stream_complete

PROTOCOL — Dreamverse
──────────────────────
  connect → session_init_v2  (prompt in initial_rollout_prompt; server auto-starts)
          ← queue_status
          ← gpu_assigned
          ← rewrite_seed_prompts_started / seed_prompts_updated / rewrite_seed_prompts_complete
          ← ltx2_stream_start
          ← ltx2_segment_start … ltx2_segment_complete  (×6)
          ← [binary chunks]
          ← ltx2_stream_complete
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import base64
import dataclasses
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime
from urllib.parse import unquote, urlparse
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# ── Dependency bootstrap ───────────────────────────────────────────────────────


def _fmt_model_progress(mp: Any) -> str:
    """Human-readable suffix for server ``model_progress`` (denoise step / ETA)."""
    if not isinstance(mp, dict):
        return ""
    stage = mp.get("stage")
    if not stage:
        return ""
    bits: list[str] = [f"model={stage}"]
    step, tot = mp.get("step"), mp.get("total")
    if step is not None and tot is not None:
        bits.append(f"{step}/{tot}")
    pct = mp.get("pct")
    if pct is not None:
        bits.append(f"{pct}%")
    eta = mp.get("eta_s")
    if eta is not None:
        bits.append(f"eta~{eta}s")
    avg = mp.get("avg_step_s")
    if avg is not None:
        bits.append(f"{avg}s/step")
    return "  " + "  ".join(bits)

def _ensure(pkg: str, import_as: str | None = None):
    """Import a package, auto-installing it if missing.  Exit with a clear
    message only if the installation itself fails."""
    import importlib
    name = import_as or pkg
    try:
        return __import__(name)
    except ImportError:
        print(f"  '{pkg}' not found — installing…")
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


# ── Mode configuration ─────────────────────────────────────────────────────────

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
)

MODES: dict[str, dict] = {
    "ltx": {
        "host":             "",  # must use --server (no public MLX endpoint)
        "display_name":     "LTX (MLX local)",
        "default_prompt":   (
            'people clap as a romantic song ends and a girl speaks italian '
            '"orca madonna raghi"'
        ),
        "default_delay":    1.0,
        "default_prefix":   "ltx",
        "default_preset_id": "simple_custom_prompt",
        "default_preset_label": "",
        "default_enhance":  False,
        "multi_segment":    False,   # single-clip, uses simple_generate
        "requires_server":  True,
    },
    "dreamverse": {
        "host":             "dreamverse.fastvideo.org",
        "display_name":     "Dreamverse",
        "default_prompt":   "a kid burps into a tunnel, with a huge echo",
        "default_delay":    2.0,
        "default_prefix":   "dreamverse",
        "default_preset_id": "custom_editable",
        "default_preset_label": "Custom rollout",
        "default_enhance":  True,    # GPT expands prompt into segments
        "multi_segment":    True,    # no simple_generate; server auto-starts
    },
}


_SERVER_OVERRIDE: str | None = None   # set by --server flag


def _ws_url(mode: str) -> str:
    if _SERVER_OVERRIDE:
        return _SERVER_OVERRIDE
    host = MODES[mode].get("host") or ""
    if not host.strip():
        raise RuntimeError(
            f"mode {mode!r} has no public WebSocket host — set --server ws://HOST:PORT/ws"
        )
    return f"wss://{host}/ws"

def _ws_headers(mode: str) -> dict:
    if _SERVER_OVERRIDE:
        from urllib.parse import urlparse
        parsed = urlparse(_SERVER_OVERRIDE)
        origin = f"{parsed.scheme.replace('ws', 'http')}://{parsed.netloc}"
        return {"Origin": origin, "User-Agent": USER_AGENT}
    return {"Origin": f"https://{MODES[mode]['host']}", "User-Agent": USER_AGENT}


# ── Data models ────────────────────────────────────────────────────────────────

CHAIN_METHOD_AUTOCONTINUE = "autocontinue"
CHAIN_METHOD_NATIVE_EXTEND = "native_extend"
VALID_CHAIN_METHODS = frozenset({CHAIN_METHOD_AUTOCONTINUE, CHAIN_METHOD_NATIVE_EXTEND})


class JobStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclasses.dataclass
class GenerationParams:
    """Unified parameters for LTX local and Dreamverse sessions."""
    prompt:                   str
    preset_id:                str            = "simple_custom_prompt"
    preset_label:             str            = ""
    enhancement_enabled:      bool           = False
    single_clip_mode:         bool           = True
    auto_extension_enabled:   bool           = False
    loop_generation_enabled:  bool           = False
    initial_image:            Optional[dict] = None
    # Passed to local server simple_generate (None → server default seed / resolution).
    seed:                       Optional[int] = None
    num_frames:                Optional[int] = None
    height:                     Optional[int] = None
    width:                      Optional[int] = None
    num_steps:                  Optional[int] = None
    generation_mode:            str           = "generate"  # generate|a2v|retake|extend|ic_lora|keyframe|lipdub|face_swap|id_lora
    audio_input:            Optional[dict] = None
    source_video:           Optional[dict] = None
    end_image:              Optional[dict] = None
    retake_start:              Optional[int] = None
    retake_end:                Optional[int] = None
    extend_frames:             Optional[int] = None
    extend_direction:      Optional[str] = None
    lora_specs: list[tuple[str, float]] = dataclasses.field(default_factory=list)
    video_conditioning_specs: list[tuple[dict, float]] = dataclasses.field(default_factory=list)
    # a2v chain clip 2+: i2v last-frame visual + mux audio (avoids A2V re-conditioning glitches).
    a2v_visual_i2v_continue: bool = False
    # ltx-2-mlx advanced options (optional; server defaults when unset).
    enhance_prompt: bool = False
    pipeline_profile: str = "distilled"
    cfg_scale: Optional[float] = None
    stg_scale: Optional[float] = None
    stage2_steps: Optional[int] = None
    no_regen_audio: bool = False
    reference_strength: Optional[float] = None
    audio_start_seconds: Optional[float] = None
    negative_prompt: str = ""
    skip_stage_2: bool = False
    upsample_only: bool = False
    modality_scale: Optional[float] = None
    audio_cfg_scale: Optional[float] = None
    identity_guidance_scale: Optional[float] = None


@dataclasses.dataclass
class Job:
    id:            int
    params:        GenerationParams
    output_path:   Path
    status:        JobStatus      = JobStatus.PENDING
    attempt:       int            = 0
    max_attempts:  int            = 1
    error:         Optional[str]  = None
    # timing
    started_at:    Optional[float] = None
    finished_at:   Optional[float] = None
    ttff_ms:       Optional[float] = None
    gen_latency_ms: Optional[float] = None
    e2e_latency_ms: Optional[float] = None
    segment_count: int = 0
    chunk_count:   int = 0
    file_bytes:    int = 0
    cleanup_paths: list[Path] = dataclasses.field(default_factory=list)

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at

    @property
    def can_retry(self) -> bool:
        return self.attempt < self.max_attempts

    def summary_line(self) -> str:
        icon = {"done": "✓", "failed": "✗", "skipped": "–",
                "running": "⟳", "pending": "·"}
        st   = icon.get(self.status.value, "?")
        name = self.output_path.name
        if self.status == JobStatus.DONE:
            seg = f"  {self.segment_count} segments" if self.segment_count else ""
            return (f"  {st} [{self.id:02d}] {name}  "
                    f"{self.file_bytes/1024:.0f} KB  "
                    f"{self.elapsed:.1f}s{seg}  "
                    f"{self.chunk_count} chunks")
        if self.status == JobStatus.FAILED:
            return f"  {st} [{self.id:02d}] {name}  ERROR: {self.error}"
        return f"  {st} [{self.id:02d}] {name}  ({self.status.value})"


# ── Protocol message builders ──────────────────────────────────────────────────

def msg_session_init_v2(p: GenerationParams, mode: str) -> str:
    """Build session_init_v2 — structure differs by mode."""
    if mode == "dreamverse":
        return json.dumps({
            "type":                    "session_init_v2",
            "preset_id":               p.preset_id,
            "preset_label":            p.preset_label,
            "curated_prompts":         [],
            "initial_rollout_prompt":  p.prompt.strip(),
            "initial_image":           p.initial_image,
            "single_clip_mode":        False,
            "enhancement_enabled":     p.enhancement_enabled,
            "auto_extension_enabled":  p.auto_extension_enabled,
            "loop_generation_enabled": p.loop_generation_enabled,
        })
    else:  # ltx — single-clip local server
        # Carry ``initial_image`` on session_init so the server fallback (server.py)
        # works for autocontinue / i2v and for clients that only attach the start
        # frame to session_init_v2.
        payload = {
            "type":                    "session_init_v2",
            "preset_id":               p.preset_id,
            "curated_prompts":         [],
            "initial_image":           p.initial_image,
            "audio_input":             p.audio_input,
            "source_video":            p.source_video,
            "single_clip_mode":        p.single_clip_mode,
            "enhancement_enabled":     p.enhancement_enabled,
            "auto_extension_enabled":  p.auto_extension_enabled,
            "loop_generation_enabled": p.loop_generation_enabled,
        }
        return json.dumps(payload)


def msg_simple_generate(p: GenerationParams) -> str:
    """LTX local (single-segment) — trigger generation after session_init_v2."""
    d: dict[str, Any] = {
        "type":                "simple_generate",
        "preset_id":           p.preset_id,
        "prompt_id":           p.preset_id,
        "prompt":              p.prompt.strip(),
        "initial_image":       p.initial_image,
        "single_clip_mode":    True,
        "enhancement_enabled": p.enhancement_enabled,
    }
    if p.seed is not None:
        d["seed"] = int(p.seed)
    if p.num_frames is not None:
        d["num_frames"] = int(p.num_frames)
    if p.height is not None:
        d["height"] = int(p.height)
    if p.width is not None:
        d["width"] = int(p.width)
    if p.num_steps is not None:
        d["num_steps"] = int(p.num_steps)
    if p.generation_mode:
        d["mode"] = p.generation_mode
    if p.audio_input is not None:
        d["audio_input"] = p.audio_input
    if p.source_video is not None:
        d["source_video"] = p.source_video
    if p.retake_start is not None:
        d["retake_start"] = int(p.retake_start)
    if p.retake_end is not None:
        d["retake_end"] = int(p.retake_end)
    if p.extend_frames is not None:
        d["extend_frames"] = int(p.extend_frames)
    if p.extend_direction:
        d["extend_direction"] = str(p.extend_direction)
    if p.lora_specs:
        d["lora_paths"] = [[path, float(scale)] for path, scale in p.lora_specs]
    if p.video_conditioning_specs:
        d["video_conditioning"] = [
            [payload, float(scale)] for payload, scale in p.video_conditioning_specs
        ]
    if p.a2v_visual_i2v_continue:
        d["a2v_visual_i2v_continue"] = True
    if p.end_image is not None:
        d["end_image"] = p.end_image
    if p.enhance_prompt:
        d["enhance_prompt"] = True
    if p.pipeline_profile and p.pipeline_profile != "distilled":
        d["pipeline_profile"] = p.pipeline_profile
    if p.cfg_scale is not None:
        d["cfg_scale"] = float(p.cfg_scale)
    if p.stg_scale is not None:
        d["stg_scale"] = float(p.stg_scale)
    if p.stage2_steps is not None:
        d["stage2_steps"] = int(p.stage2_steps)
    if getattr(p, "skip_stage_2", False):
        d["skip_stage_2"] = True
    if getattr(p, "upsample_only", False):
        d["upsample_only"] = True
    if getattr(p, "modality_scale", None) is not None:
        d["modality_scale"] = float(p.modality_scale)
    if getattr(p, "audio_cfg_scale", None) is not None:
        d["audio_cfg_scale"] = float(p.audio_cfg_scale)
    if getattr(p, "identity_guidance_scale", None) is not None:
        d["identity_guidance_scale"] = float(p.identity_guidance_scale)
    if p.no_regen_audio:
        d["no_regen_audio"] = True
    if p.reference_strength is not None:
        d["reference_strength"] = float(p.reference_strength)
    if p.audio_start_seconds is not None and float(p.audio_start_seconds) > 0:
        d["audio_start_seconds"] = float(p.audio_start_seconds)
    if (p.negative_prompt or "").strip():
        d["negative_prompt"] = p.negative_prompt.strip()
    return json.dumps(d)


def msg_append_prompt(prompt: str, prompt_id: str | None = None) -> str:
    return json.dumps({
        "type":      "append_prompt",
        "prompt_id": prompt_id or str(uuid.uuid4()),
        "prompt":    prompt.strip(),
    })


def msg_rewrite_seed_prompts(instruction: str, window_prompts: list) -> str:
    return json.dumps({
        "type":                  "rewrite_seed_prompts",
        "rewrite_instruction":   instruction,
        "prompt_window_prompts": window_prompts,
    })


def msg_set_auto_extension(enabled: bool) -> str:
    return json.dumps({"type": "set_auto_extension",    "enabled": enabled})
def msg_set_loop_generation(enabled: bool) -> str:
    return json.dumps({"type": "set_loop_generation",   "enabled": enabled})
def msg_set_paused(paused: bool) -> str:
    return json.dumps({"type": "set_generation_paused", "paused":  paused})
def msg_restart_generation() -> str:
    return json.dumps({"type": "restart_generation"})
def msg_reset_to_seed_prompts() -> str:
    return json.dumps({"type": "reset_to_seed_prompts"})


# When ``--server`` leaves ``idle_timeout`` unlimited, we still slice ``recv()`` with
# this interval so we can log + run RFC6455 ping/pong even if the server sends no
# JSON (half-open links, middleboxes).  Server ``generation_keepalive`` usually
# arrives sooner, so this is a back-stop, not a user-facing “timeout”.
SOFT_WS_RECV_TICK_S = 90.0


# ── Single-video WebSocket session ─────────────────────────────────────────────

class VideoSession:
    """
    One WebSocket connection → one video (single-segment or multi-segment).

    LTX local flow:
      connect → session_init_v2 → simple_generate → recv binary → save

    Dreamverse flow:
      connect → session_init_v2 (prompt embedded) → GPT rewrite → recv binary → save
    """

    def __init__(
        self,
        job: Job,
        mode: str,
        verbose: bool = False,
    ):
        self.job     = job
        self.mode    = mode
        self.verbose = verbose
        self._chunks: list[bytes] = []
        self._done   = asyncio.Event()
        self._t0     = 0.0
        self._t_first_chunk: float | None = None
        self._segments_done  = 0
        self._segments_total = 0

    # ── logging ──────────────────────────────────────────────────────────────

    def _log(self, msg: str, always: bool = False):
        if always or self.verbose:
            ts = f"[{time.time()-self._t0:6.2f}s]"
            print(f"    {ts} {msg}", flush=True)

    def _progress(self):
        n  = len(self._chunks)
        kb = sum(len(c) for c in self._chunks) / 1024
        seg_info = (f"  seg {self._segments_done}/{self._segments_total}"
                    if self._segments_total else "")
        print(f"\r      ↓ {n} chunks  {kb:.1f} KB{seg_info}   ", end="", flush=True)

    # ── main ─────────────────────────────────────────────────────────────────

    async def run(self, idle_timeout: float | None) -> bool:
        """
        Waits until ``ltx2_stream_complete`` (or server error), the server closes the
        socket, or an **idle** deadline trips (with WebSocket ping/pong probe).

        There is **no wall-clock session cap** — long generations are not cut off
        by elapsed time since connect.

        ``idle_timeout`` — if not ``None``, no application message for this many
        seconds triggers a WebSocket ping; only a failed pong ends the session.
        If ``None`` (typical with ``--server``), recv is still sliced every
        ``SOFT_WS_RECV_TICK_S`` seconds for **logging + ping/pong** so the process
        never sits silently on a dead TCP socket.

        **Keep-alive layers (all visible with default logging where noted):**

        1. **Library** — ``ping_interval=30``, ``ping_timeout=None``: automatic
           RFC6455 pings; no application-level log line per ping.
        2. **Recv slice + ping** — each ``asyncio.wait_for(ws.recv(), …)`` expiry
           logs ``… sending WebSocket ping`` then awaits the pong (always logged).
        3. **App JSON** (``--server`` only) — background task sends
           ``generation_status`` every 30s (logged as ``→``); server may answer with
           ``generation_status_ack`` (logged as ``←``).
        4. **Server JSON** — ``generation_keepalive`` frames log as ``← keepalive``.
        """
        self._t0 = time.time()
        try:
            async with websockets.connect(
                _ws_url(self.mode),
                additional_headers=_ws_headers(self.mode),
                max_size=200 * 1024 * 1024,
                open_timeout=None,
                ping_interval=30,
                ping_timeout=None,
                close_timeout=5,
            ) as ws:
                await self._on_open(ws)

                recv_cap = (
                    idle_timeout
                    if idle_timeout is not None
                    else SOFT_WS_RECV_TICK_S
                )

                async def _recv_loop() -> None:
                    while not self._done.is_set():
                        try:
                            frame = await asyncio.wait_for(
                                ws.recv(),
                                timeout=recv_cap,
                            )
                        except asyncio.TimeoutError:
                            if idle_timeout is not None:
                                tag = (
                                    f"no application data for {idle_timeout:.0f}s "
                                    "(idle limit)"
                                )
                            else:
                                tag = (
                                    f"no application data for {SOFT_WS_RECV_TICK_S:.0f}s "
                                    "(soft recv slice; server JSON optional)"
                                )
                            self._log(f"{tag} — sending WebSocket ping…", always=True)
                            try:
                                pong_waiter = await ws.ping()
                                await asyncio.wait_for(pong_waiter, timeout=30.0)
                            except Exception as exc:
                                self.job.error = (
                                    f"{tag}: WebSocket ping/pong failed: {exc!r}"
                                )
                                return
                            self._log(
                                "← WebSocket pong OK — still waiting…",
                                always=True,
                            )
                            continue
                        if isinstance(frame, bytes):
                            self._handle_binary(frame)
                        else:
                            await self._handle_json(frame)

                async def _status_pinger() -> None:
                    if not _SERVER_OVERRIDE:
                        return
                    ping_json = json.dumps({"type": "generation_status"})
                    while not self._done.is_set():
                        await asyncio.sleep(30.0)
                        if self._done.is_set():
                            break
                        try:
                            self._log(
                                "→ generation_status  (app-level keepalive ping)",
                                always=True,
                            )
                            await ws.send(ping_json)
                        except Exception:
                            break

                recv_task = asyncio.create_task(_recv_loop())
                ping_task = asyncio.create_task(_status_pinger())
                try:
                    await recv_task
                finally:
                    ping_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ping_task
                    if not recv_task.done():
                        recv_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await recv_task
                if self.job.error:
                    return False
        except websockets.exceptions.ConnectionClosed as exc:
            if not self._done.is_set():
                self.job.error = (
                    f"server closed WebSocket ({exc.code}): "
                    f"{exc.reason or 'no reason'}"
                )
                return False
        except Exception as exc:
            self.job.error = str(exc)
            return False

        if not self._chunks:
            self.job.error = "no video data received"
            return False

        return self._save()

    # ── connection ───────────────────────────────────────────────────────────

    async def _on_open(self, ws):
        self._log("connected ✓", always=True)
        await ws.send(msg_session_init_v2(self.job.params, self.mode))

        if self.mode == "dreamverse":
            self._log(
                f"→ session_init_v2  prompt={self.job.params.prompt[:60]!r}",
                always=True,
            )
            # Dreamverse: server starts automatically after GPT rewrite; no simple_generate
        else:
            self._log("→ session_init_v2")
            await ws.send(msg_simple_generate(self.job.params))
            self._log(
                f"→ simple_generate  prompt={self.job.params.prompt[:60]!r}",
                always=True,
            )

    # ── frame handlers ────────────────────────────────────────────────────────

    def _handle_binary(self, data: bytes):
        if self._t_first_chunk is None:
            self._t_first_chunk = time.time() - self._t0
            self.job.ttff_ms = self._t_first_chunk * 1000
            self._log(f"first chunk  TTFF={self._t_first_chunk:.2f}s", always=True)
        self._chunks.append(data)
        self._progress()

    async def _handle_json(self, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self._log(f"bad JSON: {raw[:80]}")
            return

        t = msg.get("type", "")

        # ── shared events ─────────────────────────────────────────────────────

        if t == "connected":
            self._log("← connected")

        elif t == "queue_position":
            pos = msg.get("position") or msg.get("queue_position", "?")
            self._log(f"← queue_position: {pos}", always=True)

        elif t == "queue_status":
            pos   = msg.get("position", "?")
            avail = msg.get("available_gpus", "?")
            total = msg.get("total_gpus", "?")
            agid  = msg.get("active_generation_id")
            extra = f"  active_gen={agid}" if agid else ""
            self._log(
                f"← queue_status  position={pos}  gpus={avail}/{total}{extra}",
                always=True,
            )

        elif t == "gpu_assigned":
            gpu_id  = msg.get("gpu_id", "?")
            timeout = msg.get("session_timeout", "?")
            gid     = msg.get("generation_id")
            gextra  = f"  generation_id={gid}" if gid else ""
            if self.mode == "dreamverse":
                self._log(
                    f"← gpu_assigned  gpu={gpu_id}  "
                    f"session_timeout={timeout}s{gextra}",
                    always=True,
                )
            else:
                self._log(f"← gpu_assigned ✓{gextra}", always=True)

        elif t == "generation_keepalive":
            elapsed = msg.get("elapsed_s", "?")
            phase   = msg.get("phase", "?")
            extra = _fmt_model_progress(msg.get("model_progress"))
            self._log(
                f"← keepalive  {phase}  {elapsed}s{extra}",
                always=True,
            )

        elif t == "generation_status_ack":
            extra = _fmt_model_progress(msg.get("model_progress"))
            self._log(
                f"← generation_status_ack  phase={msg.get('phase', '?')}  "
                f"elapsed={msg.get('elapsed_s', '?')}s  "
                f"id={msg.get('generation_id', '')[:8]}…{extra}",
                always=True,
            )

        elif t == "session_started":
            self._log("← session_started")

        elif t in ("ltx2_stream_start", "stream_started"):
            if self.mode == "dreamverse":
                self._segments_total = msg.get("total_segments", 0)
                mode_str = msg.get("stream_mode", "?")
                self._log(
                    f"← stream start  segments={self._segments_total}  "
                    f"mode={mode_str}",
                    always=True,
                )
            else:
                self._log("← stream started — receiving frames…", always=True)

        elif t == "ltx2_segment_start":
            seg   = msg.get("segment_idx", "?")
            total = msg.get("total_segments", "?")
            if self.mode == "dreamverse":
                prompt = msg.get("prompt", "")
                self._log(
                    f"← segment {seg}/{total} started  "
                    f"prompt={prompt[:60]!r}",
                    always=True,
                )
            else:
                self._log(f"← segment {seg}/{total} started", always=True)

        elif t == "ltx2_segment_complete":
            seg   = msg.get("segment_idx", "?")
            total = msg.get("total_segments", "?")
            if self.mode == "dreamverse":
                self._segments_done = (int(seg) if str(seg).isdigit()
                                       else self._segments_done)
                self._log(f"← segment {seg}/{total} complete", always=True)
            else:
                self._log(f"← segment {seg} complete")

        elif t == "ltx2_stream_complete":
            print()  # newline after progress bar
            elapsed = time.time() - self._t0
            if self.mode == "dreamverse":
                self._log(
                    f"← ltx2_stream_complete  {elapsed:.1f}s  "
                    f"{self._segments_done} segments  "
                    f"{len(self._chunks)} chunks",
                    always=True,
                )
            else:
                self._log(
                    f"← ltx2_stream_complete  {elapsed:.1f}s  "
                    f"{len(self._chunks)} chunks",
                    always=True,
                )
            self._done.set()

        elif t == "media_init":
            mime = msg.get("mime", "?")
            sid  = msg.get("stream_id", "?")
            if self.mode == "dreamverse":
                self._log(f"← media_init  mime={mime}  stream_id={sid}")
            else:
                self._log(f"← media_init  mime={mime}")

        elif t == "media_segment_complete":
            self._log("← media_segment_complete")

        elif t == "step_complete":
            self._log("← step_complete")

        # ── dreamverse-specific events ────────────────────────────────────────

        elif t == "generation_paused_updated":
            self._log(f"← generation_paused_updated  paused={msg.get('paused')}")

        elif t == "rewrite_seed_prompts_started":
            model = msg.get("model", "?")
            self._log(f"← rewrite started  model={model}  "
                      "(expanding prompt into segments…)", always=True)

        elif t == "seed_prompts_updated":
            prompts = msg.get("prompts", [])
            reason  = msg.get("reason", "?")
            fallback = msg.get("fallback_used", False)
            self._log(
                f"← seed_prompts_updated  {len(prompts)} segments  "
                f"reason={reason}  fallback={fallback}",
                always=True,
            )
            if self.verbose:
                for i, sp in enumerate(prompts, 1):
                    print(f"      seg {i}: {sp[:80]}…")

        elif t == "rewrite_seed_prompts_complete":
            fallback = msg.get("fallback_used", False)
            err      = msg.get("error")
            if err:
                self._log(f"← rewrite complete  WARNING: {err[:120]}", always=True)
            else:
                self._log(f"← rewrite complete  fallback={fallback}", always=True)

        elif t == "seed_prompts_reset_applied":
            self._log(f"← seed_prompts_reset_applied  reason={msg.get('reason','?')}")

        elif t == "segment_prompt_source":
            seg    = msg.get("segment_idx", "?")
            source = msg.get("source", "?")
            self._log(f"← segment {seg} source={source}")

        elif t == "loop_generation_updated":
            self._log(f"← loop_generation_updated  enabled={msg.get('enabled')}")

        elif t == "auto_extension_updated":
            self._log(f"← auto_extension_updated  enabled={msg.get('enabled')}")

        elif t == "session_timeout":
            self._log("← session_timeout", always=True)
            self.job.error = "session timed out"
            self._done.set()

        # ── shared error/misc events ──────────────────────────────────────────

        elif t == "latency":
            gms = msg.get("generation_ms") or msg.get("generation_latency_ms")
            ems = msg.get("e2e_ms")        or msg.get("e2e_latency_ms")
            self.job.gen_latency_ms = gms
            self.job.e2e_latency_ms = ems
            self._log(f"← latency  gen={gms}ms  e2e={ems}ms", always=True)

        elif t == "session_notice":
            notice = msg.get("notice") or msg.get("message") or ""
            self._log(f"← notice: {notice!r}", always=True)

        elif t == "generation_cap":
            self._log(f"← generation_cap: {msg}", always=True)
            self._done.set()

        elif t == "time_remaining":
            self._log(f"← time_remaining: {msg.get('seconds')}s")

        elif t == "error":
            err_code = msg.get("error_code", "")
            err      = msg.get("message") or msg.get("detail") or str(msg)
            if err_code == "ip_session_limit":
                self._log("← ip_session_limit — another session is active on this IP",
                          always=True)
                self.job.error = "ip_session_limit"
            else:
                self._log(f"← ERROR: {err}", always=True)
                self.job.error = err
            self._done.set()

        elif t in ("ping", "pong", ""):
            pass

        else:
            self._log(f"← {t}: {json.dumps(msg)[:160]}")

    # ── save ─────────────────────────────────────────────────────────────────

    def _save(self) -> bool:
        data = b"".join(self._chunks)
        try:
            self.job.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.job.output_path.write_bytes(data)
            self.job.chunk_count   = len(self._chunks)
            self.job.segment_count = self._segments_done
            self.job.file_bytes    = len(data)
            self._log(
                f"saved → {self.job.output_path}  ({len(data)/1024:.1f} KB)",
                always=True,
            )
            return True
        except OSError as exc:
            self.job.error = f"write failed: {exc}"
            return False


# ── Queue manager ──────────────────────────────────────────────────────────────

class GenerationQueue:
    """Runs jobs one at a time (sequential), with retry support."""

    def __init__(
        self,
        jobs:           list[Job],
        mode:           str,
        idle_timeout:   float | None,
        delay:          float,
        verbose:        bool = False,
        autocontinue:   bool = False,
        chain_method:   str = CHAIN_METHOD_AUTOCONTINUE,
    ):
        self.jobs          = jobs
        self.mode          = mode
        self.idle_timeout = idle_timeout
        self.delay = delay
        self.verbose = verbose
        self.autocontinue = autocontinue
        method = (chain_method or CHAIN_METHOD_AUTOCONTINUE).strip().lower()
        self.chain_method = (
            method if method in VALID_CHAIN_METHODS else CHAIN_METHOD_AUTOCONTINUE
        )

    @staticmethod
    def _cleanup_job_temps(job: Job) -> None:
        for p in job.cleanup_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError as exc:
                print(f"  [cleanup] warning: could not remove temp file {p}: {exc}")
        job.cleanup_paths.clear()

    async def run_all(self):
        total  = len(self.jobs)
        done   = 0
        failed = 0
        name   = MODES[self.mode]["display_name"]

        print(f"\n{'═'*60}")
        print(f"  {name} Queue — {total} job(s)")
        print(f"  Endpoint : {_ws_url(self.mode)}")
        _idle = (
            "unlimited (recv waits until server sends or closes)"
            if self.idle_timeout is None
            else f"{self.idle_timeout:.0f}s (+ ping probe if quiet)"
        )
        print(
            f"  Session    : no wall-clock cap  |  idle {_idle}  |  "
            "WS transport ping timeout disabled"
        )
        print(f"  Delay    : {self.delay}s between jobs")
        print(f"{'═'*60}\n")

        for i, job in enumerate(self.jobs):
            print(f"  ┌─ Job {job.id:02d}/{total}  {job.output_path.name}")
            print(f"  │  prompt: {job.params.prompt[:72]!r}")
            if job.params.initial_image:
                print(f"  │  image : {job.params.initial_image.get('name','?')}")
            if job.params.enhancement_enabled:
                print(f"  │  enhance: ON")
            print(f"  └{'─'*50}")

            job.status     = JobStatus.RUNNING
            job.started_at = time.time()

            success = False
            while not success and job.attempt < job.max_attempts:
                job.attempt += 1
                if job.attempt > 1:
                    if job.error == "ip_session_limit":
                        wait = 15
                        print(f"    ip_session_limit — waiting {wait}s for "
                              "previous session to close…")
                    else:
                        wait = min(2 ** job.attempt, 30)
                        print(f"    retry {job.attempt}/{job.max_attempts}  "
                              f"(backoff {wait}s)…")
                    await asyncio.sleep(wait)
                    job.error = None

                session = VideoSession(job, mode=self.mode, verbose=self.verbose)
                success = await session.run(self.idle_timeout)

            job.finished_at = time.time()
            job.status      = JobStatus.DONE if success else JobStatus.FAILED

            if success:
                done += 1
                seg = (f", {job.segment_count} segments"
                       if job.segment_count else "")
                print(f"  ✓ saved  {job.output_path.name}  "
                      f"({job.file_bytes/1024:.0f} KB{seg}, {job.elapsed:.1f}s)")
                if self.autocontinue and i + 1 < len(self.jobs):
                    nxt = self.jobs[i + 1].params
                    if self.chain_method == CHAIN_METHOD_NATIVE_EXTEND:
                        nxt.generation_mode = "extend"
                        nxt.initial_image = None
                        nxt.source_video = load_media_payload(
                            str(job.output_path),
                            kind="video",
                        )
                        if nxt.extend_frames is None:
                            probed = extend_latent_frames_for_video(job.output_path)
                            if probed is not None:
                                nxt.extend_frames = probed
                            else:
                                nf = (
                                    nxt.num_frames
                                    if nxt.num_frames is not None
                                    else job.params.num_frames
                                )
                                nxt.extend_frames = resolve_extend_latent_frames(
                                    num_frames=nf,
                                )
                        if not nxt.extend_direction:
                            nxt.extend_direction = job.params.extend_direction or "after"
                        nxt.seed = int(time.time_ns() % (2**31 - 1)) or 1
                        print(
                            f"  → native_extend: {job.output_path.name} → job {i + 2:02d}  "
                            f"mode=extend  extend_frames={nxt.extend_frames}"
                        )
                    else:
                        frame = extract_last_frame(job.output_path)
                        if frame:
                            nxt.initial_image = frame
                            # Vary seed so a missed i2v path does not reproduce the same noise as clip 1.
                            nxt.seed = int(time.time_ns() % (2**31 - 1)) or 1
                            prev_mode = (job.params.generation_mode or "").strip().lower()
                            if prev_mode == "a2v":
                                # Match Web UI: avoid re-entering A2Vid with last-frame image.
                                nxt.a2v_visual_i2v_continue = True
                            print(f"  → autocontinue: last frame → job {i + 2:02d}  seed={nxt.seed}")
            else:
                failed += 1
                print(f"  ✗ FAILED  {job.error}")

            # Cleanup per-job temporary artifacts (e.g. audiocontinue split chunks),
            # regardless of success/failure, once retries for this job are done.
            self._cleanup_job_temps(job)

            print()

            if job is not self.jobs[-1] and self.delay > 0:
                await asyncio.sleep(self.delay)

        print(f"{'═'*60}")
        print(f"  Summary: {done} done  {failed} failed  ({total} total)")
        print(f"{'═'*60}")
        for job in self.jobs:
            print(job.summary_line())
        print()

        return done, failed



# ── Helpers ────────────────────────────────────────────────────────────────────

def try_finalize_native_extend_chain(
    jobs: list[Job],
    file_prefix: str,
    ext: str,
    verbose: bool,
) -> None:
    """Promote the last native-extend output as the merged deliverable.

    ltx-2-mlx ``extend_from_video`` returns source footage plus new frames in a
    single MP4. Concatenating segment files would duplicate the source at each
    join; the cumulative result is always the last successful clip.
    """
    done = sorted(
        (j for j in jobs if j.status == JobStatus.DONE),
        key=lambda j: j.id,
    )
    if not done:
        print("\n  [native_extend] finalize skipped — no successful clips.")
        return

    final = done[-1]
    out_dir = final.output_path.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged = out_dir / f"{file_prefix}_merged_{ts}.{ext}"

    first_frames = count_video_frames(done[0].output_path) if len(done) > 1 else None
    final_frames = count_video_frames(final.output_path)

    try:
        shutil.copy2(final.output_path, merged)
    except OSError as exc:
        print(f"\n  [native_extend] could not copy final clip to merged: {exc}")
        return

    removed = 0
    for j in done:
        try:
            if j.output_path.exists():
                j.output_path.unlink()
                removed += 1
        except OSError as exc:
            print(f"  [native_extend] warning: could not remove {j.output_path}: {exc}")

    kb = merged.stat().st_size / 1024
    merged_frames = count_video_frames(merged)
    frame_note = ""
    if first_frames is not None and final_frames is not None:
        frame_note = (
            f"  ({first_frames}f source → {final_frames}f extended"
            f"{f', {merged_frames}f merged' if merged_frames else ''})"
        )
    elif final_frames is not None:
        frame_note = f"  ({final_frames} frames)"
    if verbose:
        print(
            f"\n  [native_extend] finalized {len(done)} segment(s) → {merged}  ({kb:.0f} KB)"
            f"{frame_note}"
        )
    else:
        print(
            f"\n  [native_extend] finalized chain → {merged}  ({kb:.0f} KB)"
            f"{frame_note}"
        )
    if (
        first_frames is not None
        and final_frames is not None
        and final_frames <= first_frames
    ):
        print(
            "  [native_extend] warning: extended clip is not longer than the source — "
            "check duration preset (5s = 121 frames @ 24fps) and extend_frames."
        )
    print(f"  [native_extend] removed {removed} fragment file(s).")


def try_autoconcat_clips(
    jobs: list[Job],
    file_prefix: str,
    ext: str,
    verbose: bool,
    compact: bool = False,
) -> None:
    """After a successful autocontinue run, merge DONE outputs with PyAV and delete fragments."""
    from ltx_media import concat_videos, media_available

    done = sorted(
        (j for j in jobs if j.status == JobStatus.DONE),
        key=lambda j: j.id,
    )
    if len(done) < 2:
        print(
            f"\n  [autoconcat] skipped — need at least 2 successful clips "
            f"(got {len(done)})."
        )
        return

    if not media_available():
        print("\n  [autoconcat] PyAV is not available (pip install av).")
        print("  [autoconcat] Without it, individual clip files are left as-is.")
        print("  [autoconcat] Current fragment paths:")
        for j in done:
            print(f"  [autoconcat]   {j.output_path}")
        return

    out_dir = done[0].output_path.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged = out_dir / f"{file_prefix}_merged_{ts}.{ext}"

    try:
        concat_videos(
            [j.output_path for j in done],
            merged,
            reencode_h265=compact,
        )
    except Exception as exc:
        print("\n  [autoconcat] merge failed — leaving fragment files in place.")
        print(f"  [autoconcat]   {exc}")
        if merged.exists():
            try:
                merged.unlink()
            except OSError:
                pass
        return

    removed = 0
    for j in done:
        try:
            j.output_path.unlink()
            removed += 1
        except OSError as exc:
            print(f"  [autoconcat] warning: could not remove {j.output_path}: {exc}")

    kb = merged.stat().st_size / 1024 if merged.exists() else 0
    mode_note = " (libx265 compact)" if compact else ""
    print(f"\n  [autoconcat] merged {len(done)} clips → {merged}  ({kb:.0f} KB){mode_note}")
    print(f"  [autoconcat] removed {removed} fragment file(s).")


# ── Helpers ────────────────────────────────────────────────────────────────────

MAX_IMAGE_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_MEDIA_DOWNLOAD_BYTES = 512 * 1024 * 1024


def _parse_http_content_type(header: str | None) -> str | None:
    if not header:
        return None
    return header.split(";")[0].strip().lower() or None


def _is_http_url(s: str) -> bool:
    u = s.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def _filename_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name
    if name and name not in (".", ".."):
        return name
    return "image.jpg"


def _image_payload(name: str, mime: str, raw: bytes) -> dict:
    if not mime.startswith("image/"):
        mime = "image/jpeg"
    data = base64.b64encode(raw).decode()
    return {
        "name": name,
        "mime_type": mime,
        "data_url": f"data:{mime};base64,{data}",
    }


def _binary_payload(name: str, mime: str, raw: bytes, fallback_mime: str) -> dict:
    mt = (mime or "").strip().lower()
    if not mt:
        mt = fallback_mime
    data = base64.b64encode(raw).decode()
    return {
        "name": name,
        "mime_type": mt,
        "data_url": f"data:{mt};base64,{data}",
    }


def _download_url_to_temp_image(url: str) -> tuple[Path, str]:
    """Download image bytes to a temp file. Caller must unlink the path when done."""
    req = urllib.request.Request(
        url.strip(),
        headers={"User-Agent": USER_AGENT},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            ctype = _parse_http_content_type(resp.headers.get("Content-Type"))
            raw = resp.read(MAX_IMAGE_DOWNLOAD_BYTES + 1)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Image URL returned HTTP {e.code}: {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download image: {e.reason}") from e

    if len(raw) > MAX_IMAGE_DOWNLOAD_BYTES:
        raise ValueError(
            f"Image exceeds {MAX_IMAGE_DOWNLOAD_BYTES // (1024 * 1024)} MiB: {url}"
        )

    mime = ctype if ctype and ctype.startswith("image/") else None
    if not mime:
        mime, _ = mimetypes.guess_type(_filename_from_url(url))
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"

    ext = mimetypes.guess_extension(mime)
    if ext == ".jpe":
        ext = ".jpg"
    if not ext:
        ext = ".bin"

    fd, path_str = tempfile.mkstemp(
        suffix=ext, prefix="videofentanyl_img_", dir=None
    )
    path = Path(path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return path, mime
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_image_payload(path_or_url: str) -> dict:
    """Load an image from a local path or http(s) URL and return the initial_image payload dict.

    Remote images are written to a temporary file, read into the payload, then removed.
    """
    s = path_or_url.strip()
    if _is_http_url(s):
        tmp: Path | None = None
        try:
            tmp, mime = _download_url_to_temp_image(s)
            raw = tmp.read_bytes()
            name = _filename_from_url(s)
            return _image_payload(name, mime, raw)
        finally:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    p = Path(s)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path_or_url}")
    mime, _ = mimetypes.guess_type(str(p))
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    return _image_payload(p.name, mime, p.read_bytes())


def _download_url_to_temp_binary(url: str, max_bytes: int) -> tuple[Path, str | None]:
    req = urllib.request.Request(
        url.strip(),
        headers={"User-Agent": USER_AGENT},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            ctype = _parse_http_content_type(resp.headers.get("Content-Type"))
            raw = resp.read(max_bytes + 1)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"URL returned HTTP {e.code}: {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download URL: {e.reason}") from e
    if len(raw) > max_bytes:
        raise ValueError(
            f"Input exceeds {max_bytes // (1024 * 1024)} MiB: {url}"
        )
    name = _filename_from_url(url)
    mime = ctype
    if not mime:
        mime, _ = mimetypes.guess_type(name)
    ext = Path(name).suffix or ".bin"
    fd, path_str = tempfile.mkstemp(suffix=ext, prefix="videofentanyl_media_", dir=None)
    path = Path(path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return path, mime
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_media_payload(path_or_url: str, *, kind: str) -> dict:
    """
    Load audio/video input from local path or URL and return data_url payload dict.
    This keeps client/server decoupled when they run on different machines.
    """
    s = path_or_url.strip()
    if kind == "audio":
        fallback = "audio/mpeg"
    elif kind == "video":
        fallback = "video/mp4"
    else:
        fallback = "application/octet-stream"

    if _is_http_url(s):
        tmp: Path | None = None
        try:
            tmp, mime = _download_url_to_temp_binary(s, MAX_MEDIA_DOWNLOAD_BYTES)
            raw = tmp.read_bytes()
            name = _filename_from_url(s)
            return _binary_payload(name, mime or fallback, raw, fallback)
        finally:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    p = Path(s)
    if not p.exists():
        raise FileNotFoundError(f"{kind.title()} not found: {path_or_url}")
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = fallback
    return _binary_payload(p.name, mime, p.read_bytes(), fallback)


def split_audio_for_jobs(
    audio_path: str,
    *,
    segment_seconds: float,
    required_segments: int,
) -> tuple[list[Path], Path]:
    """
    Split one audio file into sequential chunks for audiocontinue.

    Returns: (segment_paths, temp_directory).
    Caller owns cleanup (unlink each segment and remove temp directory).
    """
    from ltx_media import media_available, split_audio

    if not media_available():
        raise RuntimeError(
            "--audiocontinue requires PyAV (pip install av) to segment input audio"
        )
    source_temp: Path | None = None
    raw_audio = audio_path.strip()
    if _is_http_url(raw_audio):
        req = urllib.request.Request(
            raw_audio,
            headers={"User-Agent": USER_AGENT},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = resp.read(512 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Audio URL returned HTTP {e.code}: {raw_audio}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to download audio: {e.reason}") from e
        if len(payload) > 512 * 1024 * 1024:
            raise RuntimeError("Audio URL exceeds 512 MiB limit")
        ext = Path(urlparse(raw_audio).path).suffix or ".wav"
        fd, tmp_path = tempfile.mkstemp(prefix="videofentanyl_audio_", suffix=ext)
        source_temp = Path(tmp_path)
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        src = source_temp.resolve()
    else:
        src = Path(raw_audio).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be > 0")

    temp_dir = Path(tempfile.mkdtemp(prefix="videofentanyl_audio_segments_"))
    try:
        segs = split_audio(
            src,
            temp_dir,
            segment_seconds=segment_seconds,
            required_segments=required_segments,
            suffix=".wav",
        )
    except Exception:
        if source_temp is not None:
            source_temp.unlink(missing_ok=True)
        raise
    if source_temp is not None:
        source_temp.unlink(missing_ok=True)
    return segs[:required_segments], temp_dir


def snap_ltx_pixel_frames(raw: int) -> int:
    """Snap to LTX temporal constraint: frame counts must be 8k+1."""
    k = max(0, round((int(raw) - 1) / 8))
    return 8 * k + 1


def segment_pixel_frames_from_request(
    num_frames: int | None,
    duration_seconds: float | None = None,
    *,
    fps: float = 24.0,
) -> int:
    """One chain segment's pixel frame count (matches Web UI duration presets)."""
    if duration_seconds is not None:
        return snap_ltx_pixel_frames(int(float(duration_seconds) * fps))
    if num_frames is not None:
        return snap_ltx_pixel_frames(int(num_frames))
    return snap_ltx_pixel_frames(int(5.0 * fps))


def extend_latent_frames_for_pixel_frames(pixel_frames: int) -> int:
    """
    Latent frames to add so ``extend_from_video`` grows a clip by ~one segment.

    ltx-2-mlx extend adds ``extend_frames`` latent tokens; each latent token is
    ~8 pixel frames. For an 8k+1 source, ``(pixel_frames - 1) // 8 + 1`` matches
    the source latent depth and roughly doubles duration when used as extend_frames.
    """
    nf = snap_ltx_pixel_frames(max(9, int(pixel_frames)))
    return max(2, (nf - 1) // 8 + 1)


def count_video_frames(video_path: Path | str) -> int | None:
    """Return decoded video frame count, or None if probing fails."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if stream.frames and int(stream.frames) > 0:
                return int(stream.frames)
            count = 0
            for _ in container.decode(stream):
                count += 1
            return count if count > 0 else None
    except Exception:
        return None


def extend_latent_frames_for_video(video_path: Path | str) -> int | None:
    """Derive native_extend ``extend_frames`` from an on-disk MP4."""
    count = count_video_frames(video_path)
    if count is None:
        return None
    return extend_latent_frames_for_pixel_frames(count)


def resolve_extend_latent_frames(
    *,
    video_path: Path | str | None = None,
    num_frames: int | None = None,
    duration_seconds: float | None = None,
    fps: float = 24.0,
) -> int:
    """Best-effort extend_frames for native_extend chaining."""
    if video_path is not None:
        probed = extend_latent_frames_for_video(video_path)
        if probed is not None:
            return probed
    pixel = segment_pixel_frames_from_request(num_frames, duration_seconds, fps=fps)
    return extend_latent_frames_for_pixel_frames(pixel)


def extract_last_frame(video_path: Path) -> Optional[dict]:
    """Extract the last presented frame from a video for autocontinue conditioning.

    Uses PyAV for decode and Pillow for PNG encoding (less loss than JPEG before
  the pipeline's I2V preprocess round-trip).
    """
    import importlib
    for pkg, mod in (("av", "av"), ("Pillow", "PIL")):
        try:
            __import__(mod)
        except ImportError:
            print(f"  '{pkg}' not found — installing…")
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
    import av
    import io
    try:
        last = None
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            # Seek near the end so we decode only tail frames (not the whole clip).
            if stream.duration is not None and stream.time_base is not None:
                dur_s = float(stream.duration * stream.time_base)
                if dur_s > 0.2:
                    try:
                        container.seek(int((dur_s - 0.1) / stream.time_base), stream=stream)
                    except av.AVError:
                        pass
            for frame in container.decode(stream):
                last = frame
        if last is None:
            return None
        buf = io.BytesIO()
        last.to_image().save(buf, "PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        return {
            "name": "autocontinue.png",
            "mime_type": "image/png",
            "data_url": f"data:image/png;base64,{data}",
        }
    except Exception as exc:
        print(f"  [autocontinue] frame extraction failed: {exc}")
        return None


def sanitize_filename(s: str, maxlen: int = 48) -> str:
    """Turn a prompt string into a safe filename fragment."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "_", s)
    return s[:maxlen].strip("_")


def build_jobs(
    prompts:       list[str],
    count:         int,
    params_kwargs: dict,
    output_dir:    Path,
    prefix:        str,
    ext:           str,
    max_attempts:  int,
    image_path:    str | None = None,
    audio_path:    str | None = None,
    video_path:    str | None = None,
    end_image_path: str | None = None,
    video_conditioning_specs: list[tuple[dict, float]] | None = None,
) -> list[Job]:
    """
    Build the sequential job list.
    Each prompt is repeated `count` times before moving to the next prompt.
    """
    initial_image = load_image_payload(image_path) if image_path else None
    end_image = load_image_payload(end_image_path) if end_image_path else None
    audio_input = load_media_payload(audio_path, kind="audio") if audio_path else None
    source_video = load_media_payload(video_path, kind="video") if video_path else None
    jobs: list[Job] = []
    job_id = 1
    for prompt in prompts:
        for _ in range(count):
            slug  = sanitize_filename(prompt)
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{prefix}_{job_id:03d}_{slug}_{ts}.{ext}"
            jobs.append(Job(
                id=job_id,
                params=GenerationParams(
                    prompt=prompt,
                    initial_image=initial_image,
                    end_image=end_image,
                    audio_input=audio_input,
                    source_video=source_video,
                    video_conditioning_specs=video_conditioning_specs or [],
                    **params_kwargs,
                ),
                output_path=output_dir / fname,
                max_attempts=max_attempts,
            ))
            job_id += 1
    return jobs


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="videofentanyl",
        description=(
            "Unified LTX (local MLX) / Dreamverse queue manager.\n"
            "Select backend with --mode (default: ltx; ltx requires --server)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # LTX local (default — needs --server)
  python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
      --prompt "a fox running through snow"

  # Dreamverse long-form (~30s, GPT-expanded)
  python videofentanyl.py --mode dreamverse --prompt "a dog learns to fly"

  # 3 videos from one prompt
  python videofentanyl.py --prompt "sunset over ocean" --count 3

  # two different prompts (dreamverse), 2 each
  python videofentanyl.py --mode dreamverse \\
      --prompt "forest rain" --prompt "city lights" --count 2

  # image-to-video with AI enhancement
  python videofentanyl.py --prompt "anime girl walking" --image photo.jpg --enhance

  # skip GPT expansion (dreamverse only)
  python videofentanyl.py --mode dreamverse --prompt "detailed desc…" --no-enhance

  # dry-run: preview queue without connecting
  python videofentanyl.py --prompt "test" --count 3 --dry-run
        """,
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--mode", "-m",
        choices=list(MODES.keys()), default="ltx",
        help="generation backend (default: ltx; ltx requires --server)",
    )

    # ── Generation ────────────────────────────────────────────────────────────
    gen = p.add_argument_group("generation")
    gen.add_argument(
        "--prompt", "-p",
        action="append", dest="prompts", metavar="TEXT",
        help="prompt text (repeat for multiple prompts)",
    )
    gen.add_argument(
        "--count", "-n",
        type=int, default=1, metavar="N",
        help="videos to generate per prompt (default: 1)",
    )
    gen.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="override random seed for local server",
    )
    gen.add_argument(
        "--num-frames",
        type=int,
        default=None,
        metavar="N",
        help="override frame count for local server",
    )
    gen.add_argument(
        "--height",
        type=int,
        default=None,
        metavar="PX",
        help="override output height for local server",
    )
    gen.add_argument(
        "--width",
        type=int,
        default=None,
        metavar="PX",
        help="override output width for local server",
    )
    gen.add_argument(
        "--enhance", "-e",
        action="store_true", default=None,
        help="enable AI prompt enhancement / GPT rewrite "
             "(ltx: off by default; dreamverse: on by default)",
    )
    gen.add_argument(
        "--no-enhance",
        action="store_true",
        help="disable GPT prompt expansion (dreamverse default is on)",
    )
    gen.add_argument(
        "--image", "-i",
        metavar="PATH_OR_URL",
        help="input image for i2v / keyframe start (local path or http(s) URL)",
    )
    gen.add_argument(
        "--end-image",
        metavar="PATH_OR_URL",
        help="end image for keyframe interpolation (local path or http(s) URL)",
    )
    gen.add_argument(
        "--audio",
        metavar="PATH_OR_URL",
        help="audio input for audio-to-video mode (local path or http(s) URL)",
    )
    gen.add_argument(
        "--video",
        metavar="PATH_OR_URL",
        help="source video for retake/extend mode (local path or http(s) URL)",
    )
    gen.add_argument(
        "--generation-mode",
        choices=(
            "generate",
            "a2v",
            "retake",
            "extend",
            "ic_lora",
            "keyframe",
            "lipdub",
            "face_swap",
            "id_lora",
        ),
        default="generate",
        help="local generation route (default: generate)",
    )
    gen.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="run ltx-2-mlx Gemma prompt enhancement before generation",
    )
    gen.add_argument(
        "--pipeline-profile",
        choices=("distilled", "two_stage", "hq", "one_stage"),
        default="distilled",
        help="ltx-2-mlx generate pipeline profile (default: distilled)",
    )
    gen.add_argument("--cfg-scale", type=float, default=None, metavar="F")
    gen.add_argument("--stg-scale", type=float, default=None, metavar="F")
    gen.add_argument("--stage2-steps", type=int, default=None, metavar="N")
    gen.add_argument(
        "--skip-stage-2",
        action="store_true",
        help="IC-LoRA / HDR / ID-LoRA: skip upscale stage (half-res output, faster)",
    )
    gen.add_argument(
        "--upsample-only",
        action="store_true",
        help="ID-LoRA: 2× upsample after stage 1, skip stage-2 denoise/reload",
    )
    gen.add_argument(
        "--modality-scale",
        type=float,
        default=None,
        metavar="F",
        help="ID-LoRA modality guidance scale (default 1.0 balanced; 3.0 faithful)",
    )
    gen.add_argument(
        "--audio-cfg-scale",
        type=float,
        default=None,
        metavar="F",
        help="ID-LoRA audio CFG scale (default 7.0)",
    )
    gen.add_argument(
        "--identity-guidance-scale",
        type=float,
        default=None,
        metavar="F",
        help="ID-LoRA identity guidance (default 3.0; try 4–6 for stronger voice match)",
    )
    gen.add_argument(
        "--no-regen-audio",
        action="store_true",
        help="retake/extend: keep source audio instead of regenerating",
    )
    gen.add_argument("--reference-strength", type=float, default=None, metavar="F")
    gen.add_argument(
        "--num-steps",
        type=int,
        default=None,
        metavar="N",
        help="override denoising steps for local server (uses server default when omitted)",
    )
    gen.add_argument(
        "--retake-start",
        type=int,
        default=None,
        metavar="N",
        help="retake start latent frame index (generation-mode=retake)",
    )
    gen.add_argument(
        "--retake-end",
        type=int,
        default=None,
        metavar="N",
        help="retake end latent frame index (generation-mode=retake)",
    )
    gen.add_argument(
        "--extend-frames",
        type=int,
        default=None,
        metavar="N",
        help="number of latent frames to extend (generation-mode=extend)",
    )
    gen.add_argument(
        "--extend-direction",
        choices=("before", "after"),
        default=None,
        help="extend direction for generation-mode=extend",
    )
    gen.add_argument(
        "--lora",
        action="append",
        nargs=2,
        metavar=("LORA", "SCALE"),
        default=None,
        help="LoRA spec for generation-mode=ic_lora; repeatable: --lora <path_or_repo> <scale>",
    )
    gen.add_argument(
        "--video-conditioning",
        action="append",
        nargs=2,
        metavar=("VIDEO", "SCALE"),
        default=None,
        help="Weighted conditioning video for generation-mode=ic_lora; repeatable",
    )
    gen.add_argument(
        "--preset-id",
        default=None, metavar="ID",
        help="override preset ID sent in session_init_v2",
    )
    gen.add_argument(
        "--preset-label",
        default=None, metavar="STR",
        help="override preset label (dreamverse only, default: 'Custom rollout')",
    )
    gen.add_argument(
        "--auto-extension",
        action="store_true",
        help="enable server-side segment auto-extension",
    )
    gen.add_argument(
        "--loop",
        action="store_true",
        help="enable loop generation",
    )

    # ── Queue / network ───────────────────────────────────────────────────────
    q = p.add_argument_group("queue & network")
    q.add_argument(
        "--idle-timeout",
        type=float,
        default=None,
        metavar="SECS",
        help=(
            "if the server sends no application message for this many seconds, "
            "probe with a WebSocket ping (default: 120 hosted; with --server "
            "default is unlimited — use this flag to set a finite idle cap)"
        ),
    )
    q.add_argument(
        "--delay", "-d",
        type=float, default=None, metavar="SECS",
        help="delay between consecutive jobs "
             "(default: 1.0s for ltx, 2.0s for dreamverse)",
    )
    q.add_argument(
        "--retries", "-r",
        type=int, default=1, metavar="N",
        help="max attempts per job (default: 1)",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    o = p.add_argument_group("output")
    o.add_argument(
        "--output-dir", "-o",
        default=".", metavar="DIR",
        help="output directory (default: current directory)",
    )
    o.add_argument(
        "--prefix",
        default=None, metavar="STR",
        help="filename prefix (default: 'ltx' for ltx, "
             "'dreamverse' for dreamverse)",
    )
    o.add_argument(
        "--ext",
        default="mp4", metavar="EXT",
        help="file extension (default: mp4)",
    )

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--verbose", "-v", action="store_true",
                   help="verbose protocol logging")
    p.add_argument("--dry-run", action="store_true",
                   help="print queue plan and exit without connecting")
    p.add_argument(
        "--autocontinue",
        action="store_true",
        help="chain clips: last frame of each clip feeds the next (see --chain-method)",
    )
    p.add_argument(
        "--chain-method",
        choices=sorted(VALID_CHAIN_METHODS),
        default=CHAIN_METHOD_AUTOCONTINUE,
        metavar="METHOD",
        help="how chained clips connect when --autocontinue is set: "
             "autocontinue (extract last frame → i2v, default) or "
             "native_extend (ltx-2-mlx RetakePipeline.extend_from_video on prior MP4)",
    )
    p.add_argument(
        "--autoconcat",
        action="store_true",
        help="after generation, merge successful autocontinue clips with PyAV "
             "(-c copy), then delete the fragments (requires --autocontinue; "
             "pip install av)",
    )
    p.add_argument(
        "--audiocontinue",
        action="store_true",
        help=(
            "music-video helper: implies --autocontinue --autoconcat --autocompact, "
            "splits --audio into per-clip segments, and assigns one segment per job "
            "for generation-mode a2v"
        ),
    )
    p.add_argument(
        "--autocompact",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--server",
        default=None, metavar="URL",
        help="override WebSocket endpoint (e.g. ws://localhost:8765/ws); "
             "use with server.py for fully local generation",
    )

    return p


async def async_main(args: argparse.Namespace):
    global _SERVER_OVERRIDE
    if args.server:
        _SERVER_OVERRIDE = args.server

    mode    = args.mode
    cfg     = MODES[mode]
    temp_audio_segment_dir: Path | None = None

    if cfg.get("requires_server") and not args.server:
        print(
            "Error: mode 'ltx' requires a local WebSocket server.\n"
            "  Example:  --server ws://127.0.0.1:8765/ws\n"
            "Start server.py first (see README — ltx-2-mlx / MLX).",
            file=sys.stderr,
        )
        sys.exit(2)

    # ── Resolve prompts ───────────────────────────────────────────────────────
    prompts: list[str] = args.prompts or [cfg["default_prompt"]]
    prompts = [p.strip() for p in prompts if p.strip()]
    if not prompts:
        print("Error: at least one non-empty --prompt is required")
        sys.exit(1)
    if args.count < 1:
        print("Error: --count must be >= 1")
        sys.exit(1)
    if args.audiocontinue:
        args.autocontinue = True
        args.autoconcat = True
        args.autocompact = True
        if args.generation_mode == "generate":
            args.generation_mode = "a2v"
    if args.autoconcat and not args.autocontinue:
        print("Error: --autoconcat requires --autocontinue")
        sys.exit(2)
    if args.autocompact and not args.autoconcat:
        print("Error: --autocompact requires --autoconcat")
        sys.exit(2)
    if args.generation_mode != "generate" and mode != "ltx":
        print("Error: --generation-mode is supported only with --mode ltx")
        sys.exit(2)
    if args.audiocontinue and args.generation_mode != "a2v":
        print("Error: --audiocontinue only supports --generation-mode a2v")
        sys.exit(2)
    if args.generation_mode == "a2v" and not args.audio:
        print("Error: --generation-mode a2v requires --audio")
        sys.exit(2)
    if args.audiocontinue and not args.audio:
        print("Error: --audiocontinue requires --audio")
        sys.exit(2)
    if args.generation_mode == "retake":
        if not args.video:
            print("Error: --generation-mode retake requires --video")
            sys.exit(2)
        if args.retake_start is None or args.retake_end is None:
            print("Error: retake mode requires both --retake-start and --retake-end")
            sys.exit(2)
    if args.generation_mode == "extend":
        if not args.video:
            print("Error: --generation-mode extend requires --video")
            sys.exit(2)
        if args.extend_frames is None:
            print("Error: extend mode requires --extend-frames")
            sys.exit(2)
    if args.generation_mode == "ic_lora":
        if not args.lora:
            print("Error: --generation-mode ic_lora requires at least one --lora <path_or_repo> <scale>")
            sys.exit(2)
    if args.generation_mode == "keyframe":
        if not args.image or not args.end_image:
            print("Error: --generation-mode keyframe requires --image and --end-image")
            sys.exit(2)
    if args.generation_mode == "lipdub":
        if not args.video:
            print("Error: --generation-mode lipdub requires --video (reference video)")
            sys.exit(2)
        if not args.lora or len(args.lora) != 1:
            print("Error: --generation-mode lipdub requires exactly one --lora")
            sys.exit(2)
    if args.generation_mode == "face_swap":
        if not args.video:
            print("Error: --generation-mode face_swap requires --video (reference video)")
            sys.exit(2)
        if not args.image:
            print("Error: --generation-mode face_swap requires --image (face identity)")
            sys.exit(2)
        if not args.lora or len(args.lora) != 1:
            print("Error: --generation-mode face_swap requires exactly one --lora (head swap)")
            sys.exit(2)
    if args.generation_mode == "id_lora":
        if not args.image:
            print("Error: --generation-mode id_lora requires --image (first-frame identity)")
            sys.exit(2)
        if not args.audio:
            print("Error: --generation-mode id_lora requires --audio (identity reference, ~5s)")
            sys.exit(2)
        if args.lora and len(args.lora) != 1:
            print("Error: --generation-mode id_lora accepts at most one --lora (ID-LoRA adapter)")
            sys.exit(2)
    if args.chain_method == CHAIN_METHOD_NATIVE_EXTEND:
        if not args.autocontinue:
            print("Error: --chain-method native_extend requires --autocontinue")
            sys.exit(2)
        if args.audiocontinue:
            print("Error: --chain-method native_extend is incompatible with --audiocontinue")
            sys.exit(2)
        if args.generation_mode not in ("generate",):
            print(
                "Error: --chain-method native_extend only supports --generation-mode generate "
                "(use i2v via --image on clip 1)",
            )
            sys.exit(2)
    if args.num_steps is not None and args.num_steps < 1:
        print("Error: --num-steps must be >= 1")
        sys.exit(2)

    lora_specs: list[tuple[str, float]] = []
    if args.lora:
        for entry in args.lora:
            path = str(entry[0]).strip()
            try:
                scale = float(entry[1])
            except (TypeError, ValueError):
                print(f"Error: invalid LoRA scale: {entry[1]!r}")
                sys.exit(2)
            lora_specs.append((path, scale))

    vc_specs: list[tuple[dict, float]] = []
    if args.video_conditioning:
        for entry in args.video_conditioning:
            src = str(entry[0]).strip()
            try:
                scale = float(entry[1])
            except (TypeError, ValueError):
                print(f"Error: invalid video-conditioning scale: {entry[1]!r}")
                sys.exit(2)
            vc_specs.append((load_media_payload(src, kind="video"), scale))

    # ── Resolve mode-specific defaults ───────────────────────────────────────
    if args.idle_timeout is not None:
        idle_timeout: float | None = args.idle_timeout
    elif args.server:
        idle_timeout = None
    else:
        idle_timeout = 120.0

    if idle_timeout is not None and idle_timeout < 10:
        print("Error: --idle-timeout must be >= 10 seconds (or omit for unlimited with --server)")
        sys.exit(1)

    delay   = args.delay   if args.delay   is not None else cfg["default_delay"]
    prefix  = args.prefix  if args.prefix  is not None else cfg["default_prefix"]
    preset_id    = args.preset_id    if args.preset_id    is not None else cfg["default_preset_id"]
    preset_label = args.preset_label if args.preset_label is not None else cfg["default_preset_label"]

    # enhancement: --enhance / --no-enhance override the mode default
    if args.no_enhance:
        enhancement = False
    elif args.enhance:
        enhancement = True
    else:
        enhancement = cfg["default_enhance"]

    # ── Generation params ─────────────────────────────────────────────────────
    params_kwargs: dict = {
        "preset_id":               preset_id,
        "preset_label":            preset_label,
        "enhancement_enabled":     enhancement,
        "single_clip_mode":        not cfg["multi_segment"],
        "auto_extension_enabled":  args.auto_extension,
        "loop_generation_enabled": args.loop,
        "seed":                    args.seed,
        "num_frames":              args.num_frames,
        "height":                  args.height,
        "width":                   args.width,
        "num_steps":               args.num_steps,
        "generation_mode":         args.generation_mode,
        "retake_start":            args.retake_start,
        "retake_end":              args.retake_end,
        "extend_frames":           args.extend_frames,
        "extend_direction":        args.extend_direction,
        "lora_specs":              lora_specs,
        "enhance_prompt":          bool(args.enhance_prompt),
        "pipeline_profile":        args.pipeline_profile,
        "cfg_scale":               args.cfg_scale,
        "stg_scale":               args.stg_scale,
        "stage2_steps":            args.stage2_steps,
        "no_regen_audio":          bool(args.no_regen_audio),
        "reference_strength":      args.reference_strength,
        "skip_stage_2":            bool(args.skip_stage_2),
        "upsample_only":           bool(getattr(args, "upsample_only", False)),
        "modality_scale":          getattr(args, "modality_scale", None),
        "audio_cfg_scale":         getattr(args, "audio_cfg_scale", None),
        "identity_guidance_scale": getattr(args, "identity_guidance_scale", None),
    }

    # ── Build jobs ────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    jobs = build_jobs(
        prompts=prompts,
        count=args.count,
        params_kwargs=params_kwargs,
        output_dir=output_dir,
        prefix=prefix,
        ext=args.ext.lstrip("."),
        max_attempts=max(1, args.retries),
        image_path=args.image,
        audio_path=args.audio,
        video_path=args.video,
        end_image_path=args.end_image,
        video_conditioning_specs=vc_specs,
    )

    segment_seconds: float | None = None
    if args.audiocontinue:
        nf = int(args.num_frames) if args.num_frames is not None else 97
        segment_seconds = max(0.25, nf / 24.0)
        try:
            segs, temp_audio_segment_dir = split_audio_for_jobs(
                args.audio,
                segment_seconds=segment_seconds,
                required_segments=len(jobs),
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(2)
        for i, job in enumerate(jobs):
            seg = segs[i]
            job.params.audio_input = load_media_payload(str(seg), kind="audio")
            job.cleanup_paths.append(seg)

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n[dry-run] mode={mode}  {len(jobs)} job(s) in queue:\n")
        for job in jobs:
            print(f"  [{job.id:02d}] {job.output_path.name}")
            print(f"        prompt  : {job.params.prompt[:72]!r}")
            print(f"        enhance : {job.params.enhancement_enabled}")
            print(f"        genmode : {job.params.generation_mode}")
            if job.params.initial_image:
                print(f"        image   : {job.params.initial_image['name']}")
            if job.params.audio_input:
                if isinstance(job.params.audio_input, dict):
                    print(f"        audio   : {job.params.audio_input.get('name', 'audio')}")
                else:
                    print(f"        audio   : {job.params.audio_input}")
            if job.params.source_video:
                if isinstance(job.params.source_video, dict):
                    print(f"        video   : {job.params.source_video.get('name', 'video')}")
                else:
                    print(f"        video   : {job.params.source_video}")
            if job.params.lora_specs:
                print(f"        loras   : {len(job.params.lora_specs)}")
            if job.params.video_conditioning_specs:
                print(f"        vcond   : {len(job.params.video_conditioning_specs)}")
            print()
        print(f"  Endpoint   : {_ws_url(mode)}")
        print(f"  Output dir : {output_dir.resolve()}")
        _idle = "unlimited" if idle_timeout is None else f"{idle_timeout:.0f}s (+ ping)"
        print(
            f"  Limits     : no wall-clock cap  idle {_idle}  "
            f"Delay: {delay}s  Retries: {args.retries}"
        )
        if args.autoconcat:
            print("  autoconcat : after run, merge successful clips with PyAV (pip install av); "
                  "remove fragments if merge succeeds")
        if args.audiocontinue:
            print(f"  audiocontinue : ON  ({len(jobs)} segment(s), ~{segment_seconds:.2f}s each)")
        if temp_audio_segment_dir is not None:
            shutil.rmtree(temp_audio_segment_dir, ignore_errors=True)
        return

    # ── Run queue ─────────────────────────────────────────────────────────────
    queue = GenerationQueue(
        jobs=jobs,
        mode=mode,
        idle_timeout=idle_timeout,
        delay=delay,
        verbose=args.verbose,
        autocontinue=args.autocontinue,
        chain_method=args.chain_method,
    )
    try:
        done, failed = await queue.run_all()
        if args.autoconcat:
            await asyncio.to_thread(
                try_autoconcat_clips,
                jobs,
                prefix,
                args.ext.lstrip("."),
                args.verbose,
                args.autocompact,
            )
    finally:
        if temp_audio_segment_dir is not None:
            shutil.rmtree(temp_audio_segment_dir, ignore_errors=True)
    sys.exit(0 if failed == 0 else 1)


def main():
    parser = build_parser()
    args   = parser.parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
