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
        return json.dumps({
            "type":                    "session_init_v2",
            "preset_id":               p.preset_id,
            "curated_prompts":         [],
            "initial_image":           p.initial_image,
            "single_clip_mode":        p.single_clip_mode,
            "enhancement_enabled":     p.enhancement_enabled,
            "auto_extension_enabled":  p.auto_extension_enabled,
            "loop_generation_enabled": p.loop_generation_enabled,
        })


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
    ):
        self.jobs          = jobs
        self.mode          = mode
        self.idle_timeout = idle_timeout
        self.delay = delay
        self.verbose = verbose
        self.autocontinue = autocontinue

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
                    frame = extract_last_frame(job.output_path)
                    if frame:
                        nxt = self.jobs[i + 1].params
                        nxt.initial_image = frame
                        # Vary seed so a missed i2v path does not reproduce the same noise as clip 1.
                        nxt.seed = int(time.time_ns() % (2**31 - 1)) or 1
                        print(f"  → autocontinue: last frame → job {i + 2:02d}  seed={nxt.seed}")
            else:
                failed += 1
                print(f"  ✗ FAILED  {job.error}")

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


def _ffmpeg_concat_list_line(path: Path) -> str:
    """One line for ffmpeg's concat demuxer (see ffmpeg -f concat)."""
    p = path.resolve().as_posix().replace("'", r"'\''")
    return f"file '{p}'"


def try_autoconcat_clips(
    jobs: list[Job],
    file_prefix: str,
    ext: str,
    verbose: bool,
    compact: bool = False,
) -> None:
    """After a successful autocontinue run, merge DONE outputs with ffmpeg and delete fragments.

    If ffmpeg is not on PATH, logs several lines and leaves all fragment files unchanged.
    On ffmpeg failure, logs stderr and leaves fragments in place.

    When ``compact`` is True, concat is re-encoded with libx265 (CRF 28, ``faster`` preset)
    instead of stream copy; ffmpeg is invoked with ``-n`` (no overwrite).  (x264's ``film``
    tune is not valid for libx265 and is omitted.)
    """
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

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("\n  [autoconcat] ffmpeg was not found on your PATH.")
        print("  [autoconcat] Install ffmpeg to merge autocontinue fragments into one file.")
        print("  [autoconcat] Without it, individual clip files are left as-is.")
        print("  [autoconcat] Manual merge example (build list.txt with one `file 'path'` per line):")
        print("  [autoconcat]   ffmpeg -y -f concat -safe 0 -i list.txt -c copy output.mp4")
        print("  [autoconcat] Current fragment paths:")
        for j in done:
            print(f"  [autoconcat]   {j.output_path}")
        return

    out_dir = done[0].output_path.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged = out_dir / f"{file_prefix}_merged_{ts}.{ext}"

    list_body = "\n".join(_ffmpeg_concat_list_line(j.output_path) for j in done) + "\n"
    fd, list_path_str = tempfile.mkstemp(
        suffix=".ffconcat", prefix="videofentanyl_"
    )
    list_path = Path(list_path_str)
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        list_path.write_text(list_body, encoding="utf-8")
    except OSError as exc:
        print(f"\n  [autoconcat] could not write concat list: {exc}")
        try:
            list_path.unlink(missing_ok=True)
        except OSError:
            pass
        return

    log_level = "info" if verbose else "error"
    if compact:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-n",
            "-loglevel",
            log_level,
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-vcodec",
            "libx265",
            "-crf",
            "28",
            "-preset",
            "faster",
            str(merged),
        ]
        ff_timeout = 3600
    else:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            log_level,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(merged),
        ]
        ff_timeout = 600
    proc: subprocess.CompletedProcess[str] | None = None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=ff_timeout,
        )
    except subprocess.TimeoutExpired:
        print(
            f"\n  [autoconcat] ffmpeg timed out after {ff_timeout}s — "
            "leaving fragments in place."
        )
    except FileNotFoundError:
        print("\n  [autoconcat] ffmpeg executable disappeared — leaving fragments in place.")
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except OSError:
            pass

    if proc is None:
        return

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        print("\n  [autoconcat] ffmpeg failed — leaving fragment files in place.")
        if err:
            elines = err.splitlines()
            for line in elines[:40]:
                print(f"  [autoconcat]   {line}")
            if len(elines) > 40:
                print("  [autoconcat]   …")
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


def extract_last_frame(video_path: Path) -> Optional[dict]:
    """Extract the last full frame from a video file without calling external tools.

    Uses PyAV (av) for decoding and Pillow (PIL) for JPEG encoding.
    Returns an initial_image payload dict, or None on any failure.
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
            stream.codec_context.skip_frame = "NONREF"   # skip non-reference frames for speed
            for frame in container.decode(stream):
                last = frame
        if last is None:
            return None
        buf = io.BytesIO()
        last.to_image().save(buf, "JPEG")
        data = base64.b64encode(buf.getvalue()).decode()
        return {
            "name":      "autocontinue.jpg",
            "mime_type": "image/jpeg",
            "data_url":  f"data:image/jpeg;base64,{data}",
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
) -> list[Job]:
    """
    Build the sequential job list.
    Each prompt is repeated `count` times before moving to the next prompt.
    """
    initial_image = load_image_payload(image_path) if image_path else None
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
        help="input image for image-to-video (local path or http(s) URL)",
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
        help="extract the last frame of each clip and use it as the first "
             "frame (--image) of the next one; ideal for multi-clip local runs",
    )
    p.add_argument(
        "--autoconcat",
        action="store_true",
        help="after generation, merge successful autocontinue clips with ffmpeg "
             "(-c copy), then delete the fragments (requires --autocontinue; "
             "needs ffmpeg on PATH)",
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
    if args.autoconcat and not args.autocontinue:
        print("Error: --autoconcat requires --autocontinue")
        sys.exit(2)
    if args.autocompact and not args.autoconcat:
        print("Error: --autocompact requires --autoconcat")
        sys.exit(2)

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
    )

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n[dry-run] mode={mode}  {len(jobs)} job(s) in queue:\n")
        for job in jobs:
            print(f"  [{job.id:02d}] {job.output_path.name}")
            print(f"        prompt  : {job.params.prompt[:72]!r}")
            print(f"        enhance : {job.params.enhancement_enabled}")
            if job.params.initial_image:
                print(f"        image   : {job.params.initial_image['name']}")
            print()
        print(f"  Endpoint   : {_ws_url(mode)}")
        print(f"  Output dir : {output_dir.resolve()}")
        _idle = "unlimited" if idle_timeout is None else f"{idle_timeout:.0f}s (+ ping)"
        print(
            f"  Limits     : no wall-clock cap  idle {_idle}  "
            f"Delay: {delay}s  Retries: {args.retries}"
        )
        if args.autoconcat:
            print("  autoconcat : after run, merge successful clips with ffmpeg; "
                  "remove fragments if merge succeeds")
        return

    # ── Run queue ─────────────────────────────────────────────────────────────
    queue = GenerationQueue(
        jobs=jobs,
        mode=mode,
        idle_timeout=idle_timeout,
        delay=delay,
        verbose=args.verbose,
        autocontinue=args.autocontinue,
    )
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
