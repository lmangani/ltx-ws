#!/usr/bin/env python3
"""
fastvideo.py — FastVideo API Queue Manager
==========================================
Reverse-engineered from https://1080p.fastvideo.org/ (src/App.svelte)

Generates videos SEQUENTIALLY from a queue of prompts, saving each to disk.
Each video opens a fresh WebSocket connection (one session per clip).

USAGE
─────
  # Single video, default prompt
  python fastvideo.py

  # Single video with custom prompt
  python fastvideo.py --prompt "a cat playing piano in moonlight"

  # Generate 5 videos from the same prompt
  python fastvideo.py --prompt "sunset over the ocean" --count 5

  # Generate from multiple prompts (one video each)
  python fastvideo.py \
      --prompt "forest rain timelapse" \
      --prompt "city lights at night" \
      --prompt "volcano eruption aerial"

  # Multiple prompts × multiple videos each
  python fastvideo.py \
      --prompt "coral reef" --prompt "arctic tundra" \
      --count 3

  # Enable AI prompt enhancement
  python fastvideo.py --prompt "dog running" --enhance

  # Custom output directory, file prefix, format
  python fastvideo.py --prompt "test" --count 3 \
      --output-dir ./videos --prefix clip --ext mp4

  # Retry failed jobs automatically
  python fastvideo.py --prompt "test" --count 5 --retries 3

  # Dry-run: show queue without connecting
  python fastvideo.py --prompt "test" --count 3 --dry-run

  # Image-to-video (base64 encode the image)
  python fastvideo.py --prompt "anime girl walking" --image ./photo.jpg

  # Full parameter control
  python fastvideo.py \
      --prompt "epic battle scene" \
      --count 2 \
      --enhance \
      --delay 2.5 \
      --timeout 300 \
      --retries 2 \
      --output-dir ./out \
      --verbose

PROTOCOL (from App.svelte)
──────────────────────────
  Client→Server:
    session_init_v2      open handshake (sent on connect)
    simple_generate      trigger generation (sent after 'connected' event)
    append_prompt        continuation prompt (non-simple mode)
    rewrite_seed_prompts rewrite via GPT (livePromptRewriteMode)
    set_auto_extension   toggle auto-extension
    set_loop_generation  toggle loop generation
    set_generation_paused pause/resume
    reset_to_seed_prompts reset prompt window
    restart_generation   restart after segment cap

  Server→Client JSON events:
    connected            session ack → triggers simple_generate
    queue_position       {position: N} waiting in queue
    gpu_assigned         GPU ready
    session_started      session active
    stream_started       frames incoming
    ltx2_stream_complete all frames sent → done
    session_notice       informational string
    generation_cap       hit segment limit
    time_remaining       {seconds: N}
    latency              {generation_ms, e2e_ms}
    error                {message/detail}

  Server→Client binary: raw fMP4/WebM MediaSource chunks
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
import uuid
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

HOST       = "1080p.fastvideo.org"
WS_URL     = f"wss://{HOST}/ws"
ORIGIN     = f"https://{HOST}"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
)
WS_HEADERS = {"Origin": ORIGIN, "User-Agent": USER_AGENT}

DEFAULT_PROMPT  = (
    'people clap as a romantic song ends and a girl speaks italian '
    '"orca madonna raghi"'
)
DEFAULT_TIMEOUT = 240   # seconds per video
DEFAULT_RETRIES = 1
DEFAULT_DELAY   = 1.0   # seconds between consecutive jobs


# ── Data models ────────────────────────────────────────────────────────────────

class JobStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclasses.dataclass
class GenerationParams:
    """All parameters the server accepts (from App.svelte analysis)."""
    prompt:                   str
    preset_id:                str   = "simple_custom_prompt"
    enhancement_enabled:      bool  = False   # AI prompt enhancement via GPT
    single_clip_mode:         bool  = True    # simple mode = one clip per session
    auto_extension_enabled:   bool  = False   # auto-extend generation
    loop_generation_enabled:  bool  = False   # loop generation
    initial_image:            Optional[dict] = None  # image-to-video payload


@dataclasses.dataclass
class Job:
    id:          int
    params:      GenerationParams
    output_path: Path
    status:      JobStatus = JobStatus.PENDING
    attempt:     int       = 0
    max_attempts: int      = DEFAULT_RETRIES
    error:       Optional[str] = None
    # timing
    started_at:  Optional[float] = None
    finished_at: Optional[float] = None
    ttff_ms:     Optional[float] = None
    gen_latency_ms: Optional[float] = None
    e2e_latency_ms: Optional[float] = None
    chunk_count: int   = 0
    file_bytes:  int   = 0

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
                    f"{self.chunk_count} chunks")
        if self.status == JobStatus.FAILED:
            return f"  {st} [{self.id:02d}] {name}  ERROR: {self.error}"
        return f"  {st} [{self.id:02d}] {name}  ({self.status.value})"


# ── Protocol message builders ──────────────────────────────────────────────────

def msg_session_init_v2(p: GenerationParams) -> str:
    """
    sendSessionInitMessage() from App.svelte.
    Sent immediately on WebSocket open.
    curated_prompts=[] because we use simple_generate to actually trigger.
    """
    return json.dumps({
        "type":                    "session_init_v2",
        "preset_id":               p.preset_id,
        "curated_prompts":         [],
        "initial_image":           None,          # sent later in simple_generate
        "single_clip_mode":        p.single_clip_mode,
        "enhancement_enabled":     p.enhancement_enabled,
        "auto_extension_enabled":  p.auto_extension_enabled,
        "loop_generation_enabled": p.loop_generation_enabled,
    })


def msg_simple_generate(p: GenerationParams) -> str:
    """
    sendSimpleGenerateMessage() from App.svelte.
    Sent after receiving the 'connected' server event.
    """
    return json.dumps({
        "type":               "simple_generate",
        "preset_id":          p.preset_id,
        "prompt_id":          p.preset_id,
        "prompt":             p.prompt.strip(),
        "initial_image":      p.initial_image,
        "single_clip_mode":   True,
        "enhancement_enabled": p.enhancement_enabled,
    })


def msg_append_prompt(prompt: str, prompt_id: str | None = None) -> str:
    """submitLivePrompt() — for non-simple (regular) mode continuations."""
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


def msg_set_auto_extension(enabled: bool)   -> str:
    return json.dumps({"type": "set_auto_extension",    "enabled": enabled})
def msg_set_loop_generation(enabled: bool)  -> str:
    return json.dumps({"type": "set_loop_generation",   "enabled": enabled})
def msg_set_paused(paused: bool)            -> str:
    return json.dumps({"type": "set_generation_paused", "paused":  paused})
def msg_restart_generation()               -> str:
    return json.dumps({"type": "restart_generation"})
def msg_reset_to_seed_prompts()            -> str:
    return json.dumps({"type": "reset_to_seed_prompts"})


# ── Single-video WebSocket session ─────────────────────────────────────────────

class VideoSession:
    """
    Manages one WebSocket connection that generates exactly one video clip.
    Collects binary chunks and writes them to disk when complete.
    """

    def __init__(self, job: Job, verbose: bool = False):
        self.job     = job
        self.verbose = verbose
        self._chunks: list[bytes] = []
        self._done   = asyncio.Event()
        self._t0     = 0.0
        self._t_first_chunk: float | None = None

    # ── logging ──────────────────────────────────────────────────────────────

    def _log(self, msg: str, always: bool = False):
        if always or self.verbose:
            ts = f"[{time.time()-self._t0:6.2f}s]"
            print(f"    {ts} {msg}", flush=True)

    def _progress(self):
        n     = len(self._chunks)
        kb    = sum(len(c) for c in self._chunks) / 1024
        print(f"\r      ↓ {n} chunks  {kb:.1f} KB   ", end="", flush=True)

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

    # ── connection lifecycle ──────────────────────────────────────────────────

    async def _on_open(self, ws):
        self._log("connected ✓", always=True)
        await ws.send(msg_session_init_v2(self.job.params))
        self._log("→ session_init_v2")

    async def _recv_loop(self, ws):
        async for frame in ws:
            if isinstance(frame, bytes):
                self._handle_binary(frame)
            else:
                await self._handle_json(ws, frame)
            if self._done.is_set():
                break

    # ── frame handlers ────────────────────────────────────────────────────────

    def _handle_binary(self, data: bytes):
        if self._t_first_chunk is None:
            self._t_first_chunk = time.time() - self._t0
            self.job.ttff_ms = self._t_first_chunk * 1000
            self._log(f"first chunk  TTFF={self._t_first_chunk:.2f}s", always=True)
        self._chunks.append(data)
        self._progress()

    async def _handle_json(self, ws, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self._log(f"bad JSON: {raw[:80]}")
            return

        t = msg.get("type", "")

        if t == "connected":
            self._log("← connected → sending simple_generate")
            await ws.send(msg_simple_generate(self.job.params))
            self._log(f"→ simple_generate  prompt={self.job.params.prompt[:60]!r}")

        elif t == "queue_position":
            pos = msg.get("position") or msg.get("queue_position", "?")
            self._log(f"← queue_position: {pos}", always=True)

        elif t == "gpu_assigned":
            self._log("← gpu_assigned ✓", always=True)

        elif t == "session_started":
            self._log("← session_started")

        elif t == "stream_started":
            self._log("← stream_started — receiving frames…", always=True)

        elif t == "ltx2_stream_complete":
            print()   # newline after progress bar
            elapsed = time.time() - self._t0
            self._log(
                f"← ltx2_stream_complete  {elapsed:.1f}s  "
                f"{len(self._chunks)} chunks",
                always=True,
            )
            self._done.set()

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
            err = msg.get("message") or msg.get("detail") or str(msg)
            if err_code == "ip_session_limit":
                # Server enforces one active WS session per IP.
                # Mark as retryable so the queue manager will back off and retry.
                self._log(
                    "← ip_session_limit — another session is active on this IP",
                    always=True,
                )
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
            self.job.chunk_count = len(self._chunks)
            self.job.file_bytes  = len(data)
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
    """
    Runs jobs one at a time (sequential), with retry support.
    """

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
        total   = len(self.jobs)
        done    = 0
        failed  = 0

        print(f"\n{'═'*60}")
        print(f"  FastVideo Queue — {total} job(s)")
        print(f"  Endpoint : {WS_URL}")
        print(f"  Timeout  : {self.timeout}s per video")
        print(f"  Delay    : {self.delay}s between jobs")
        print(f"{'═'*60}\n")

        for job in self.jobs:
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
                        # Server allows only one active WS per IP.
                        # The previous session may still be closing — wait longer.
                        wait = 15
                        print(f"    ip_session_limit — waiting {wait}s for "
                              f"previous session to close…")
                    else:
                        wait = min(2 ** job.attempt, 30)
                        print(f"    retry {job.attempt}/{job.max_attempts}  "
                              f"(backoff {wait}s)…")
                    await asyncio.sleep(wait)
                    job.error = None  # clear so next attempt starts fresh

                session = VideoSession(job, verbose=self.verbose)
                success = await session.run(timeout=self.timeout)

            job.finished_at = time.time()
            job.status      = JobStatus.DONE if success else JobStatus.FAILED

            if success:
                done += 1
                print(f"  ✓ saved  {job.output_path.name}  "
                      f"({job.file_bytes/1024:.0f} KB, {job.elapsed:.1f}s)")
            else:
                failed += 1
                print(f"  ✗ FAILED  {job.error}")

            print()

            # Pause between jobs (avoids hammering the server)
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
    """Load an image file and build the initial_image payload."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(str(p))
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode()
    data_url = f"data:{mime};base64,{data}"
    return {"name": p.name, "mime_type": mime, "data_url": data_url}


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
    Build the job list.

    If count > 1 and multiple prompts, each prompt is repeated `count` times.
    Final queue is sequential: all repeats of prompt-1, then all of prompt-2, …
    """
    initial_image = load_image_payload(image_path) if image_path else None
    jobs: list[Job] = []
    job_id = 1

    for prompt in prompts:
        for idx in range(1, count + 1):
            slug  = sanitize_filename(prompt)
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{prefix}_{job_id:03d}_{slug}_{ts}.{ext}"
            opath = output_dir / fname

            p = GenerationParams(
                prompt=prompt,
                initial_image=initial_image,
                **params_kwargs,
            )
            jobs.append(Job(
                id=job_id,
                params=p,
                output_path=opath,
                max_attempts=max_attempts,
            ))
            job_id += 1

    return jobs


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fastvideo",
        description=(
            "FastVideo queue manager — generate 1080p videos sequentially\n"
            "from wss://1080p.fastvideo.org/ws"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # 3 videos from one prompt
  python fastvideo.py --prompt "sunset over ocean" --count 3

  # two different prompts, 2 each
  python fastvideo.py --prompt "forest rain" --prompt "city lights" --count 2

  # image-to-video with AI enhancement
  python fastvideo.py --prompt "anime girl walking" --image photo.jpg --enhance

  # aggressive retry, custom output folder
  python fastvideo.py --prompt "test" --count 10 --retries 3 --output-dir ./out
        """,
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
        action="store_true",
        help="enable AI prompt enhancement (GPT rewrite)",
    )
    gen.add_argument(
        "--image", "-i",
        metavar="PATH",
        help="path to input image for image-to-video mode",
    )
    gen.add_argument(
        "--preset-id",
        default="simple_custom_prompt", metavar="ID",
        help="preset ID sent in session_init_v2 (default: simple_custom_prompt)",
    )
    gen.add_argument(
        "--auto-extension",
        action="store_true",
        help="enable auto-extension (server-side segment extension)",
    )
    gen.add_argument(
        "--loop",
        action="store_true",
        help="enable loop generation",
    )

    # ── Queue / network ───────────────────────────────────────────────────────
    q = p.add_argument_group("queue & network")
    q.add_argument(
        "--timeout", "-t",
        type=float, default=DEFAULT_TIMEOUT, metavar="SECS",
        help=f"per-video timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    q.add_argument(
        "--delay", "-d",
        type=float, default=DEFAULT_DELAY, metavar="SECS",
        help=f"delay between consecutive jobs (default: {DEFAULT_DELAY}s)",
    )
    q.add_argument(
        "--retries", "-r",
        type=int, default=DEFAULT_RETRIES, metavar="N",
        help=f"max attempts per job (default: {DEFAULT_RETRIES})",
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
        default="video", metavar="STR",
        help="filename prefix (default: video)",
    )
    o.add_argument(
        "--ext",
        default="mp4", metavar="EXT",
        help="file extension (default: mp4)",
    )

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--verbose", "-v", action="store_true",
                   help="verbose protocol logging")
    p.add_argument("--dry-run",       action="store_true",
                   help="print queue plan and exit without connecting")

    return p


async def async_main(args: argparse.Namespace):
    # ── Resolve prompts ───────────────────────────────────────────────────────
    prompts: list[str] = args.prompts or [DEFAULT_PROMPT]
    prompts = [p.strip() for p in prompts if p.strip()]
    if not prompts:
        print("Error: at least one non-empty --prompt is required")
        sys.exit(1)

    if args.count < 1:
        print("Error: --count must be >= 1")
        sys.exit(1)

    # ── Generation params ─────────────────────────────────────────────────────
    params_kwargs = {
        "preset_id":               args.preset_id,
        "enhancement_enabled":     args.enhance,
        "single_clip_mode":        True,
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
        prefix=args.prefix,
        ext=args.ext.lstrip("."),
        max_attempts=max(1, args.retries),
        image_path=args.image,
    )

    # ── Dry run ───────────────────────────────────────────────────────────────
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
        print(f"  Timeout    : {args.timeout}s  Delay: {args.delay}s  Retries: {args.retries}")
        return

    # ── Run queue ─────────────────────────────────────────────────────────────
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
