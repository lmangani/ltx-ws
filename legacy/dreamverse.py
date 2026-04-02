#!/usr/bin/env python3
"""
dreamverse.py — Dreamverse API Queue Manager
=============================================
Reverse-engineered from https://dreamverse.fastvideo.org/ (HAR capture)

Generates LONG-FORM videos (~30s, 6 segments) sequentially from a queue
of prompts. Each video opens a fresh WebSocket connection.

Key differences from fastvideo.py (1080p):
  - Endpoint: wss://dreamverse.fastvideo.org/ws
  - preset_id: "custom_editable", preset_label: "Custom rollout"
  - single_clip_mode: false  (multi-segment, ~6 clips stitched)
  - enhancement_enabled: true by default (GPT rewrites prompt into segments)
  - NO simple_generate message — prompt goes in session_init_v2 as
    initial_rollout_prompt, generation starts automatically after rewrite

PROTOCOL (from HAR capture)
──────────────────────────
  Client→Server:
    session_init_v2         single message — contains the full prompt
                            server rewrites it into 6 segment prompts via GPT

  Server→Client JSON events (in order):
    queue_status            {position, total_gpus, available_gpus}
    gpu_assigned            {gpu_id, session_timeout}
    loop_generation_updated {enabled}
    generation_paused_updated {paused: true}   ← session starts paused
    rewrite_seed_prompts_started {model}
    generation_paused_updated {paused: false}  ← unpauses when rewrite done
    seed_prompts_updated    {prompts: [...6 strings...], reason: "rewrite"}
    rewrite_seed_prompts_complete
    ltx2_stream_start       {total_segments: 6, stream_mode: "av_fmp4"}
    seed_prompts_reset_applied
    segment_prompt_source   (per segment)
    ltx2_segment_start      (per segment)
    media_init              {mime: "video/mp4; codecs=..."}
    [binary chunks]         raw fMP4 data
    ltx2_segment_complete   (per segment)
    ltx2_stream_complete    ← done

  Server→Client binary: raw fMP4 chunks (H.264+AAC, av_fmp4 mode)

USAGE
─────
  # Single ~30s video
  python dreamverse.py --prompt "a kid burps into a tunnel, with a huge echo"

  # Queue of 3 videos
  python dreamverse.py --prompt "sunset over the ocean" --count 3

  # Multiple prompts
  python dreamverse.py \\
      --prompt "a dog learns to skateboard" \\
      --prompt "a volcano erupts at night"

  # Save to folder, verbose protocol log
  python dreamverse.py --prompt "test" --output-dir ./videos --verbose

  # Preview queue without connecting
  python dreamverse.py --prompt "test" --count 3 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import json
import mimetypes
import os
import re
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

# ── Dependency bootstrap ───────────────────────────────────────────────────────

def _ensure(pkg: str, import_as: str | None = None):
    name = import_as or pkg
    try:
        return __import__(name)
    except ImportError:
        print(f"  Installing {pkg}…")
        os.system(f"{sys.executable} -m pip install {pkg} -q")
        return __import__(name)

websockets = _ensure("websockets")


# ── Constants ──────────────────────────────────────────────────────────────────

HOST       = "dreamverse.fastvideo.org"
WS_URL     = f"wss://{HOST}/ws"
ORIGIN     = f"https://{HOST}"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
)
WS_HEADERS = {"Origin": ORIGIN, "User-Agent": USER_AGENT}

DEFAULT_PROMPT  = "a kid burps into a tunnel, with a huge echo"
DEFAULT_TIMEOUT = 480   # longer timeout — ~30s video, 6 segments
DEFAULT_RETRIES = 1
DEFAULT_DELAY   = 2.0   # slightly longer delay between jobs


# ── Data models ────────────────────────────────────────────────────────────────

class JobStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclasses.dataclass
class GenerationParams:
    """Parameters for a Dreamverse generation session."""
    prompt:                  str
    preset_id:               str  = "custom_editable"
    preset_label:            str  = "Custom rollout"
    enhancement_enabled:     bool = True    # GPT rewrites prompt into segments
    auto_extension_enabled:  bool = False
    loop_generation_enabled: bool = False
    initial_image:           Optional[dict] = None


@dataclasses.dataclass
class Job:
    id:           int
    params:       GenerationParams
    output_path:  Path
    status:       JobStatus      = JobStatus.PENDING
    attempt:      int            = 0
    max_attempts: int            = DEFAULT_RETRIES
    error:        Optional[str]  = None
    # timing
    started_at:   Optional[float] = None
    finished_at:  Optional[float] = None
    ttff_ms:      Optional[float] = None
    segment_count: int = 0
    chunk_count:  int  = 0
    file_bytes:   int  = 0

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def can_retry(self) -> bool:
        return self.attempt < self.max_attempts

    def summary_line(self) -> str:
        icon = {"done":"✓","failed":"✗","skipped":"–","running":"⟳","pending":"·"}
        st   = icon.get(self.status.value, "?")
        name = self.output_path.name
        if self.status == JobStatus.DONE:
            return (f"  {st} [{self.id:02d}] {name}  "
                    f"{self.file_bytes/1024:.0f} KB  "
                    f"{self.elapsed:.1f}s  "
                    f"{self.segment_count} segments  "
                    f"{self.chunk_count} chunks")
        if self.status == JobStatus.FAILED:
            return f"  {st} [{self.id:02d}] {name}  ERROR: {self.error}"
        return f"  {st} [{self.id:02d}] {name}  ({self.status.value})"


# ── Protocol message builder ───────────────────────────────────────────────────

def msg_session_init_v2(p: GenerationParams) -> str:
    """
    The only client→server message needed for Dreamverse.
    Prompt goes in initial_rollout_prompt. Server GPT-rewrites it into
    segment prompts, then starts generation automatically.
    No simple_generate needed.
    """
    return json.dumps({
        "type":                    "session_init_v2",
        "preset_id":               p.preset_id,
        "preset_label":            p.preset_label,
        "curated_prompts":         [],
        "initial_rollout_prompt":  p.prompt.strip(),
        "initial_image":           p.initial_image,
        "single_clip_mode":        False,   # always multi-segment
        "enhancement_enabled":     p.enhancement_enabled,
        "auto_extension_enabled":  p.auto_extension_enabled,
        "loop_generation_enabled": p.loop_generation_enabled,
    })


# ── Single-video WebSocket session ─────────────────────────────────────────────

class VideoSession:
    """
    One WebSocket connection → one long-form video (multi-segment).

    Flow:
      1. Connect → send session_init_v2 (with prompt)
      2. Server rewrites prompt via GPT (~6s)
      3. Server starts generation automatically (no client trigger)
      4. Collect binary fMP4 chunks across all segments
      5. ltx2_stream_complete → save file
    """

    def __init__(self, job: Job, verbose: bool = False):
        self.job     = job
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
        print(f"\r      ↓ {n} chunks  {kb:.1f} KB{seg_info}   ",
              end="", flush=True)

    # ── main ─────────────────────────────────────────────────────────────────

    async def run(self, timeout: float = DEFAULT_TIMEOUT) -> bool:
        self._t0 = time.time()
        try:
            async with asyncio.timeout(timeout):
                async with websockets.connect(
                    WS_URL,
                    additional_headers=WS_HEADERS,
                    max_size=200 * 1024 * 1024,
                    ping_interval=30,
                    ping_timeout=20,
                    close_timeout=5,
                ) as ws:
                    await self._on_open(ws)
                    await self._recv_loop(ws)
        except asyncio.TimeoutError:
            self.job.error = f"timed out after {timeout:.0f}s"
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
        await ws.send(msg_session_init_v2(self.job.params))
        self._log(f"→ session_init_v2  prompt={self.job.params.prompt[:60]!r}",
                  always=True)
        # No simple_generate — server starts automatically after GPT rewrite

    async def _recv_loop(self, ws):
        async for frame in ws:
            if isinstance(frame, bytes):
                self._handle_binary(frame)
            else:
                await self._handle_json(frame)
            if self._done.is_set():
                break

    # ── frame handlers ────────────────────────────────────────────────────────

    def _handle_binary(self, data: bytes):
        if self._t_first_chunk is None:
            self._t_first_chunk = time.time() - self._t0
            self.job.ttff_ms = self._t_first_chunk * 1000
            self._log(f"first chunk  TTFF={self._t_first_chunk:.2f}s",
                      always=True)
        self._chunks.append(data)
        self._progress()

    async def _handle_json(self, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self._log(f"bad JSON: {raw[:80]}")
            return

        t = msg.get("type", "")

        if t == "queue_status":
            pos   = msg.get("position", "?")
            avail = msg.get("available_gpus", "?")
            total = msg.get("total_gpus", "?")
            self._log(f"← queue_status  position={pos}  "
                      f"gpus={avail}/{total}", always=True)

        elif t == "gpu_assigned":
            gpu_id  = msg.get("gpu_id", "?")
            timeout = msg.get("session_timeout", "?")
            self._log(f"← gpu_assigned  gpu={gpu_id}  "
                      f"session_timeout={timeout}s", always=True)

        elif t == "generation_paused_updated":
            paused = msg.get("paused", "?")
            self._log(f"← generation_paused_updated  paused={paused}")

        elif t == "rewrite_seed_prompts_started":
            model = msg.get("model", "?")
            self._log(f"← rewrite started  model={model}  "
                      f"(expanding prompt into segments…)", always=True)

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
                self._log(f"← rewrite complete  WARNING: {err[:120]}",
                          always=True)
            else:
                self._log(f"← rewrite complete  fallback={fallback}",
                          always=True)

        elif t == "ltx2_stream_start":
            self._segments_total = msg.get("total_segments", 0)
            mode = msg.get("stream_mode", "?")
            self._log(
                f"← stream start  segments={self._segments_total}  "
                f"mode={mode}",
                always=True,
            )

        elif t == "seed_prompts_reset_applied":
            self._log(f"← seed_prompts_reset_applied  "
                      f"reason={msg.get('reason','?')}")

        elif t == "segment_prompt_source":
            seg    = msg.get("segment_idx", "?")
            source = msg.get("source", "?")
            self._log(f"← segment {seg} source={source}")

        elif t == "ltx2_segment_start":
            seg   = msg.get("segment_idx", "?")
            total = msg.get("total_segments", "?")
            prompt = msg.get("prompt", "")
            self._log(
                f"← segment {seg}/{total} started  "
                f"prompt={prompt[:60]!r}",
                always=True,
            )

        elif t == "ltx2_segment_complete":
            seg   = msg.get("segment_idx", "?")
            total = msg.get("total_segments", "?")
            self._segments_done = int(seg) if str(seg).isdigit() else self._segments_done
            self._log(f"← segment {seg}/{total} complete", always=True)

        elif t == "media_init":
            mime = msg.get("mime", "?")
            sid  = msg.get("stream_id", "?")
            self._log(f"← media_init  mime={mime}  stream_id={sid}")

        elif t == "media_segment_complete":
            self._log("← media_segment_complete")

        elif t == "step_complete":
            self._log("← step_complete")

        elif t == "ltx2_stream_complete":
            print()  # newline after progress bar
            elapsed = time.time() - self._t0
            self._log(
                f"← ltx2_stream_complete  {elapsed:.1f}s  "
                f"{self._segments_done} segments  "
                f"{len(self._chunks)} chunks",
                always=True,
            )
            self._done.set()

        elif t == "loop_generation_updated":
            self._log(f"← loop_generation_updated  enabled={msg.get('enabled')}")

        elif t == "auto_extension_updated":
            self._log(f"← auto_extension_updated  enabled={msg.get('enabled')}")

        elif t == "session_notice":
            notice = msg.get("notice") or msg.get("message") or ""
            self._log(f"← notice: {notice!r}", always=True)

        elif t == "session_timeout":
            self._log("← session_timeout", always=True)
            self.job.error = "session timed out"
            self._done.set()

        elif t == "error":
            err_code = msg.get("error_code", "")
            err      = msg.get("message") or msg.get("detail") or str(msg)
            if err_code == "ip_session_limit":
                self._log("← ip_session_limit — another session active on this IP",
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
                f"saved → {self.job.output_path}  "
                f"({len(data)/1024:.1f} KB)",
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
        jobs:    list[Job],
        timeout: float = DEFAULT_TIMEOUT,
        delay:   float = DEFAULT_DELAY,
        verbose: bool  = False,
    ):
        self.jobs    = jobs
        self.timeout = timeout
        self.delay   = delay
        self.verbose = verbose

    async def run_all(self):
        total  = len(self.jobs)
        done   = 0
        failed = 0

        print(f"\n{'═'*60}")
        print(f"  Dreamverse Queue — {total} job(s)")
        print(f"  Endpoint : {WS_URL}")
        print(f"  Timeout  : {self.timeout}s per video")
        print(f"  Delay    : {self.delay}s between jobs")
        print(f"{'═'*60}\n")

        for job in self.jobs:
            print(f"  ┌─ Job {job.id:02d}/{total}  {job.output_path.name}")
            print(f"  │  prompt: {job.params.prompt[:72]!r}")
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
                              f"previous session to close…")
                    else:
                        wait = min(2 ** job.attempt, 30)
                        print(f"    retry {job.attempt}/{job.max_attempts}  "
                              f"(backoff {wait}s)…")
                    await asyncio.sleep(wait)
                    job.error = None

                session = VideoSession(job, verbose=self.verbose)
                success = await session.run(timeout=self.timeout)

            job.finished_at = time.time()
            job.status      = JobStatus.DONE if success else JobStatus.FAILED

            if success:
                done += 1
                print(f"  ✓ saved  {job.output_path.name}  "
                      f"({job.file_bytes/1024:.0f} KB, "
                      f"{job.segment_count} segments, "
                      f"{job.elapsed:.1f}s)")
            else:
                failed += 1
                print(f"  ✗ FAILED  {job.error}")

            print()

            if job is not self.jobs[-1] and self.delay > 0:
                await asyncio.sleep(self.delay)

        # ── Summary ──────────────────────────────────────────────────────────
        print(f"{'═'*60}")
        print(f"  Summary: {done} done  {failed} failed  ({total} total)")
        print(f"{'═'*60}")
        for job in self.jobs:
            print(job.summary_line())
        print()

        return done, failed


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_image_payload(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(str(p))
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode()
    return {"name": p.name, "mime_type": mime,
            "data_url": f"data:{mime};base64,{data}"}


def sanitize_filename(s: str, maxlen: int = 48) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "_", s)
    return s[:maxlen].strip("_")


def build_jobs(
    prompts:      list[str],
    count:        int,
    params_kwargs: dict,
    output_dir:   Path,
    prefix:       str,
    ext:          str,
    max_attempts: int,
    image_path:   str | None = None,
) -> list[Job]:
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
        prog="dreamverse",
        description=(
            "Dreamverse queue manager — generate long-form (~30s) videos\n"
            "from wss://dreamverse.fastvideo.org/ws"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # single long video
  python dreamverse.py --prompt "a kid burps into a tunnel, with a huge echo"

  # queue of 3 videos
  python dreamverse.py --prompt "sunset over ocean" --count 3

  # two prompts, one each
  python dreamverse.py --prompt "dog learns to skateboard" --prompt "volcano at night"

  # save to folder, show full protocol log
  python dreamverse.py --prompt "test" --output-dir ./videos --verbose

  # dry run — preview queue without connecting
  python dreamverse.py --prompt "test" --count 3 --dry-run
        """,
    )

    gen = p.add_argument_group("generation")
    gen.add_argument("--prompt", "-p", action="append", dest="prompts",
                     metavar="TEXT",
                     help="prompt text (repeat for multiple prompts)")
    gen.add_argument("--count", "-n", type=int, default=1, metavar="N",
                     help="videos to generate per prompt (default: 1)")
    gen.add_argument("--image", "-i", metavar="PATH",
                     help="input image for image-to-video mode")
    gen.add_argument("--no-enhance", action="store_true",
                     help="disable GPT prompt rewrite (use raw prompt)")
    gen.add_argument("--loop", action="store_true",
                     help="enable loop generation")
    gen.add_argument("--auto-extension", action="store_true",
                     help="enable auto-extension")
    gen.add_argument("--preset-id", default="custom_editable", metavar="ID",
                     help="preset ID (default: custom_editable)")
    gen.add_argument("--preset-label", default="Custom rollout", metavar="STR",
                     help="preset label (default: Custom rollout)")

    q = p.add_argument_group("queue & network")
    q.add_argument("--timeout", "-t", type=float, default=DEFAULT_TIMEOUT,
                   metavar="SECS",
                   help=f"per-video timeout (default: {DEFAULT_TIMEOUT}s)")
    q.add_argument("--delay", "-d", type=float, default=DEFAULT_DELAY,
                   metavar="SECS",
                   help=f"delay between jobs (default: {DEFAULT_DELAY}s)")
    q.add_argument("--retries", "-r", type=int, default=DEFAULT_RETRIES,
                   metavar="N",
                   help=f"max attempts per job (default: {DEFAULT_RETRIES})")

    o = p.add_argument_group("output")
    o.add_argument("--output-dir", "-o", default=".", metavar="DIR",
                   help="output directory (default: current directory)")
    o.add_argument("--prefix", default="dreamverse", metavar="STR",
                   help="filename prefix (default: dreamverse)")
    o.add_argument("--ext", default="mp4", metavar="EXT",
                   help="file extension (default: mp4)")

    p.add_argument("--verbose", "-v", action="store_true",
                   help="verbose protocol logging")
    p.add_argument("--dry-run", action="store_true",
                   help="print queue plan and exit without connecting")

    return p


async def async_main(args: argparse.Namespace):
    prompts: list[str] = args.prompts or [DEFAULT_PROMPT]
    prompts = [p.strip() for p in prompts if p.strip()]
    if not prompts:
        print("Error: at least one non-empty --prompt is required")
        sys.exit(1)
    if args.count < 1:
        print("Error: --count must be >= 1")
        sys.exit(1)

    params_kwargs = {
        "preset_id":               args.preset_id,
        "preset_label":            args.preset_label,
        "enhancement_enabled":     not args.no_enhance,
        "auto_extension_enabled":  args.auto_extension,
        "loop_generation_enabled": args.loop,
    }

    output_dir = Path(args.output_dir)
    jobs = build_jobs(
        prompts=prompts,
        count=args.count,
        params_kwargs=params_kwargs,
        output_dir=output_dir,
        prefix=args.prefix,
        ext=args.ext.lstrip("."),
        max_attempts=max(1, args.retries),
        image_path=args.image,
    )

    if args.dry_run:
        print(f"\n[dry-run] {len(jobs)} job(s) in queue:\n")
        for job in jobs:
            print(f"  [{job.id:02d}] {job.output_path.name}")
            print(f"        prompt  : {job.params.prompt[:72]!r}")
            print(f"        enhance : {job.params.enhancement_enabled}")
            if job.params.initial_image:
                print(f"        image   : {job.params.initial_image['name']}")
            print()
        print(f"  Output dir : {output_dir.resolve()}")
        print(f"  Timeout    : {args.timeout}s  "
              f"Delay: {args.delay}s  Retries: {args.retries}")
        return

    queue = GenerationQueue(
        jobs=jobs,
        timeout=args.timeout,
        delay=args.delay,
        verbose=args.verbose,
    )
    done, failed = await queue.run_all()
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
