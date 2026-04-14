#!/usr/bin/env python3
"""
benchmark_local_generation.py — reproducible local LTX (MLX) timing
=====================================================================
Starts ``server.py`` (unless disabled), waits until the WebSocket port accepts
TCP connections (model load + bind), runs one minimal ``videofentanyl.py``
job, prints timings, then shuts the server down.

**Interpreter:** uses ``REPO_ROOT/.venv/bin/python3`` when present (see README
*Install*: ``uv venv``, ``uv pip install -r requirements.txt``,
then ``uv pip install`` the ``ltx-*-mlx`` git URLs from ``requirements.txt`` comments).
Pass ``--allow-system-python`` only if you intentionally skip a venv.

Default workload matches a short distilled run (~2 s @ 24 fps, 8 denoise steps).

Examples
--------
  # Full benchmark (free port 8765 first if something else listens)
  source .venv/bin/activate   # optional; script still picks .venv/bin/python3
  ./scripts/benchmark_local_generation.py

  # Fixed port + custom prompt
  ./scripts/benchmark_local_generation.py --port 9000 -p "a red balloon"

  # Server already running (e.g. on another machine)
  ./scripts/benchmark_local_generation.py --no-server \\
      --server-url ws://moysas-mac-studio:8765/ws

  # Same, but wait until that host accepts TCP on the port
  ./scripts/benchmark_local_generation.py --no-server --wait-ready \\
      --connect-host moysas-mac-studio --port 8765 \\
      --server-url ws://moysas-mac-studio:8765/ws

Machine-readable line (last line, easy to grep):
  BENCHMARK_JSON:{...}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Repo root (parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_PY = REPO_ROOT / "server.py"
CLIENT_PY = REPO_ROOT / "videofentanyl.py"

# LTX2 temporal constraint: (num_frames - 1) % 8 == 0.  49 frames @ 24 fps ≈ 2.04 s.
DEFAULT_NUM_FRAMES = 49
DEFAULT_INFER_STEPS = 8
DEFAULT_FPS = 24
DEFAULT_PORT = 8765

LATENCY_RE = re.compile(
    r"← latency\s+gen=([0-9.]+)ms\s+e2e=([0-9.]+)ms",
)
SAVED_RE = re.compile(
    r"✓ saved\s+.+\(\d+ KB.*,\s*([0-9.]+)s\)",
)


def resolve_interpreter(
    repo: Path,
    explicit: Path | None,
    allow_system: bool,
) -> Path:
    """
    Prefer ./.venv/bin/python3 so ``server.py`` / ``videofentanyl.py`` match README
    local MLX setup (ltx-2-mlx + deps).
    """
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            print(f"Error: --python not found: {p}", file=sys.stderr)
            sys.exit(2)
        return p

    bindir = repo / ".venv" / "bin"
    for name in ("python3", "python"):
        cand = bindir / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()

    if allow_system:
        return Path(sys.executable).resolve()

    print(
        "Error: no virtualenv at ./.venv/bin/python3\n\n"
        "Local generation needs a venv with ltx-2-mlx (see README "
        "« Local server (Apple Silicon / MLX) »). From the repo root:\n\n"
        "  uv venv --python 3.12 --seed && source .venv/bin/activate\n"
        "  uv pip install -r requirements.txt\n"
        "  uv pip install \\\n"
        '    "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git'
        '#subdirectory=packages/ltx-core-mlx" \\\n'
        '    "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git'
        '#subdirectory=packages/ltx-pipelines-mlx"\n\n'
        "Then re-run this script (it uses .venv/bin/python3 automatically).\n\n"
        "Escape hatch (not recommended): --allow-system-python\n",
        file=sys.stderr,
    )
    sys.exit(2)


def wait_tcp(host: str, port: int, timeout_s: float, interval_s: float = 0.25) -> float:
    """Block until ``host:port`` accepts TCP connections. Returns seconds waited."""
    deadline = time.monotonic() + timeout_s
    t0 = time.monotonic()
    last_err: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return time.monotonic() - t0
        except OSError as e:
            last_err = e
            time.sleep(interval_s)
    msg = f"timeout waiting for {host}:{port} ({timeout_s:.0f}s)"
    if last_err is not None:
        msg += f"  last error: {last_err}"
    raise TimeoutError(msg)


def stop_server_gracefully(proc: subprocess.Popen | None) -> None:
    """SIGINT (same as Ctrl+C), then terminate/kill if needed."""
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def parse_client_timings(stdout: str) -> dict:
    out: dict = {
        "generation_ms": None,
        "e2e_ms": None,
        "job_elapsed_s": None,
    }
    for line in stdout.splitlines():
        m = LATENCY_RE.search(line)
        if m:
            out["generation_ms"] = float(m.group(1))
            out["e2e_ms"] = float(m.group(2))
        m = SAVED_RE.search(line)
        if m:
            out["job_elapsed_s"] = float(m.group(1))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark local server + videofentanyl client with a fixed minimal "
            "distilled workload (~2 s clip, 8 steps)."
        ),
    )
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"WebSocket port (default: {DEFAULT_PORT})",
    )
    p.add_argument(
        "--connect-host",
        default="127.0.0.1",
        metavar="HOST",
        help="Host for TCP readiness probe (default: 127.0.0.1)",
    )
    p.add_argument(
        "--server-url",
        default=None,
        metavar="WS_URL",
        help=(
            "WebSocket URL passed to videofentanyl --server "
            f"(default: ws://127.0.0.1:PORT/ws)"
        ),
    )
    p.add_argument(
        "--no-server",
        action="store_true",
        help="Do not spawn server.py (connect to an already-running server)",
    )
    p.add_argument(
        "--wait-ready",
        action="store_true",
        help=(
            "With --no-server: wait for --connect-host:--port to accept TCP before "
            "running the client (default: skip wait)"
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help="Forwarded to server.py --model (default: server default distilled ID)",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"server.py --num-frames (default: {DEFAULT_NUM_FRAMES}, ~2s @ 24fps)",
    )
    p.add_argument(
        "--infer-steps",
        type=int,
        default=DEFAULT_INFER_STEPS,
        help=f"server.py --infer-steps (default: {DEFAULT_INFER_STEPS})",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"server.py --fps (default: {DEFAULT_FPS})",
    )
    p.add_argument(
        "--ready-timeout",
        type=float,
        default=900.0,
        metavar="SECS",
        help="Max wait for server TCP port after spawn (default: 900)",
    )
    p.add_argument(
        "-p", "--prompt",
        default="first person view of mouse eating cheese",
        help="Generation prompt",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmark_runs",
        help="Directory for benchmark MP4 output (default: ./benchmark_runs)",
    )
    p.add_argument(
        "--prefix",
        default="bench",
        help="videofentanyl --prefix (default: bench)",
    )
    p.add_argument(
        "--verbose-server",
        action="store_true",
        help="Forward server stdout/stderr to this terminal (default: discard)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and exit",
    )
    p.add_argument(
        "--python",
        type=Path,
        default=None,
        metavar="EXE",
        help="Python for server + client (default: ./.venv/bin/python3 if present)",
    )
    p.add_argument(
        "--allow-system-python",
        action="store_true",
        help="If ./.venv is missing, use the interpreter running this script",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    py = resolve_interpreter(
        REPO_ROOT,
        args.python,
        allow_system=args.allow_system_python,
    )
    port = args.port
    server_url = args.server_url or f"ws://127.0.0.1:{port}/ws"

    server_cmd = [
        str(py),
        str(SERVER_PY),
        "--port", str(port),
        "--num-frames", str(args.num_frames),
        "--infer-steps", str(args.infer_steps),
        "--fps", str(args.fps),
    ]
    if args.model:
        server_cmd.extend(["--model", args.model])

    out_dir = args.output_dir.expanduser().resolve()
    client_cmd = [
        str(py),
        str(CLIENT_PY),
        "--mode", "ltx",
        "--server", server_url,
        "--prompt", args.prompt,
        "--output-dir", str(out_dir),
        "--prefix", args.prefix,
        "--delay", "0",
    ]

    if args.dry_run:
        print("[dry-run] python:", py)
        print("[dry-run] server:", subprocess.list2cmdline(server_cmd))
        print("[dry-run] client:", subprocess.list2cmdline(client_cmd))
        return 0

    if not SERVER_PY.is_file():
        print(f"Error: missing {SERVER_PY}", file=sys.stderr)
        return 2
    if not CLIENT_PY.is_file():
        print(f"Error: missing {CLIENT_PY}", file=sys.stderr)
        return 2

    print(f"Using interpreter: {py}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    server_proc: subprocess.Popen | None = None
    server_ready_s: float = 0.0
    t_server_start = time.perf_counter()

    if not args.no_server:
        stdout_dest = None if args.verbose_server else subprocess.DEVNULL
        stderr_dest = None if args.verbose_server else subprocess.DEVNULL
        server_proc = subprocess.Popen(
            server_cmd,
            cwd=str(REPO_ROOT),
            stdout=stdout_dest,
            stderr=stderr_dest,
        )

        try:
            server_ready_s = wait_tcp(
                args.connect_host, port, timeout_s=args.ready_timeout
            )
        except TimeoutError as e:
            print(f"Error: {e}", file=sys.stderr)
            if server_proc.poll() is not None:
                print(
                    f"  server.py exited early (code {server_proc.returncode})",
                    file=sys.stderr,
                )
            stop_server_gracefully(server_proc)
            return 3
    else:
        if args.wait_ready:
            try:
                server_ready_s = wait_tcp(
                    args.connect_host,
                    port,
                    timeout_s=min(args.ready_timeout, 120.0),
                )
            except TimeoutError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 3
        else:
            server_ready_s = 0.0

    startup_to_client_s = time.perf_counter() - t_server_start

    t_client0 = time.perf_counter()
    try:
        cp = subprocess.run(
            client_cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=args.ready_timeout + 3600.0,
        )
    except subprocess.TimeoutError:
        print("Error: videofentanyl.py exceeded subprocess timeout", file=sys.stderr)
        stop_server_gracefully(server_proc)
        return 4
    client_wall_s = time.perf_counter() - t_client0

    # Echo client output for interactive runs (benchmark line is short).
    if cp.stdout:
        sys.stdout.write(cp.stdout)
    if cp.stderr:
        sys.stderr.write(cp.stderr)

    parsed = parse_client_timings(cp.stdout or "")

    record = {
        "version": 1,
        "ts": datetime.now(timezone.utc).isoformat(),
        "config": {
            "python": str(py),
            "num_frames": args.num_frames,
            "infer_steps": args.infer_steps,
            "fps": args.fps,
            "server_url": server_url,
            "prompt": args.prompt,
            "connect_host": args.connect_host,
            "wait_ready": args.wait_ready or not args.no_server,
        },
        "server_spawned": not args.no_server,
        "server_ready_tcp_s": server_ready_s,
        "startup_to_client_begin_s": startup_to_client_s,
        "client_subprocess_wall_s": client_wall_s,
        "generation_ms": parsed["generation_ms"],
        "e2e_ms": parsed["e2e_ms"],
        "job_elapsed_s": parsed["job_elapsed_s"],
        "client_exit_code": cp.returncode,
    }

    print()
    print("  ── benchmark summary ──")
    if args.no_server and not args.wait_ready:
        ready_note = "(wait skipped; use --wait-ready to probe TCP)"
    else:
        ready_note = f"{server_ready_s:.2f}s"
    print(f"  server TCP ready:                {ready_note}")
    print(f"  client subprocess wall:          {client_wall_s:.2f}s")
    if parsed["generation_ms"] is not None:
        print(
            f"  server-reported gen / e2e:       "
            f"{parsed['generation_ms']:.0f} ms / {parsed['e2e_ms']:.0f} ms"
        )
    else:
        print("  server-reported gen / e2e:       (not found in client stdout)")
    if parsed["job_elapsed_s"] is not None:
        print(f"  client job elapsed:              {parsed['job_elapsed_s']:.2f}s")
    print(f"  videofentanyl exit code:         {cp.returncode}")
    print()

    json_line = "BENCHMARK_JSON:" + json.dumps(record, separators=(",", ":"))
    print(json_line)

    stop_server_gracefully(server_proc)

    return 0 if cp.returncode == 0 else cp.returncode


if __name__ == "__main__":
    raise SystemExit(main())
