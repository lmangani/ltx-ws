#!/usr/bin/env python3
"""
Patch FastVideo's pyproject.toml so `fastvideo-kernel` is not required on macOS.

Upstream lists `fastvideo-kernel` as a hard dependency; that package only ships
manylinux x86_64 wheels and depends on Triton, which has no macOS arm64 wheels.
FastVideo's MPS backend uses Torch SDPA only and does not import fastvideo-kernel
at runtime for normal LTX2 inference.

Usage (from repo root, after `git submodule update --init --recursive`):

  python scripts/patch_fastvideo_pyproject_for_apple_silicon.py
  cd third_party/FastVideo && uv pip install -e .

Re-run is safe: already-patched lines are left unchanged.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER = "platform_system == 'Linux' and platform_machine == 'x86_64'"
KERNEL_DEP_RE = re.compile(
    r'^(\s*")fastvideo-kernel==([^"]+)(")\s*,?\s*$',
    re.MULTILINE,
)


def patch(content: str) -> tuple[str, bool]:
    """Return (new_content, changed)."""

    def repl(m: re.Match[str]) -> str:
        # group(1) = leading spaces + opening quote, e.g. '    "'
        # group(2) = version, group(3) = closing quote
        return (
            f'{m.group(1)}fastvideo-kernel=={m.group(2)}; {MARKER}{m.group(3)},'
        )

    if "fastvideo-kernel==" not in content:
        return content, False

    # Idempotent: upstream or we already added a platform marker on same line
    for line in content.splitlines():
        if "fastvideo-kernel==" in line and "platform_system" in line:
            return content, False

    new_content, n = KERNEL_DEP_RE.subn(repl, content, count=1)
    return new_content, n == 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "pyproject",
        nargs="?",
        default=None,
        help="Path to FastVideo pyproject.toml (default: third_party/FastVideo/pyproject.toml)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = Path(args.pyproject) if args.pyproject else root / "third_party" / "FastVideo" / "pyproject.toml"
    path = path.resolve()

    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    new_text, changed = patch(text)
    if not changed:
        print(f"No changes needed (already patched or no fastvideo-kernel line): {path}")
        return 0

    path.write_text(new_text, encoding="utf-8")
    print(f"Patched {path}")
    print("fastvideo-kernel is now only installed on Linux x86_64 (PyPI + Triton).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
