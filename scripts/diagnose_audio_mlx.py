#!/usr/bin/env python3
"""
diagnose_audio_mlx.py — why generated audio might be hiss / near-silent
=======================================================================
Checks the known mlx 0.31.2 Metal ``.at[strided].add()`` vocoder collapse
(ltx-2-mlx#34) and whether ltx-ws / ltx-core-mlx already work around it.

Safe on M3 Ultra / M4 / M5 — read-only except applying the same runtime
patch ``server.py`` uses (idempotent).

Usage
-----
  python scripts/diagnose_audio_mlx.py
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> int:
    print("ltx-ws audio / MLX diagnose")
    print(f"  mlx            : {_pkg('mlx')}")
    print(f"  mlx-metal      : {_pkg('mlx-metal')}")
    print(f"  ltx-core-mlx   : {_pkg('ltx-core-mlx')}")
    print(f"  ltx-pipelines  : {_pkg('ltx-pipelines-mlx')}")

    from ltx_mlx_backend import (
        _mlx_strided_scatter_add_broken,
        _patch_mlx_audio_upsample_scatter,
        _source_uses_strided_at_add,
        _apply_ltx_mlx_patches,
    )

    broken = _mlx_strided_scatter_add_broken()
    if broken is None:
        print("  scatter canary : skipped (mlx unavailable)")
    elif broken:
        print("  scatter canary : BROKEN  (mlx 0.31.2 Metal strided .at[].add)")
    else:
        print("  scatter canary : OK")

    try:
        from ltx_core_mlx.model.audio_vae.vocoder import UpSample1d
        from ltx_core_mlx.model.audio_vae.bwe import HannSincResampler

        up_bad = _source_uses_strided_at_add(UpSample1d.__call__)
        bwe_bad = _source_uses_strided_at_add(HannSincResampler.__call__)
        print(f"  UpSample1d     : {'BROKEN .at[].add' if up_bad else 'safe slice assign'}")
        print(f"  HannSincResamp : {'BROKEN .at[].add' if bwe_bad else 'safe slice assign'}")
    except ImportError as exc:
        print(f"  vocoder inspect: failed ({exc})")
        return 1

    _apply_ltx_mlx_patches()
    _patch_mlx_audio_upsample_scatter()

    up_bad2 = _source_uses_strided_at_add(UpSample1d.__call__)
    bwe_bad2 = _source_uses_strided_at_add(HannSincResampler.__call__)
    print(f"  after ltx-ws patch — UpSample1d: {'still broken' if up_bad2 else 'safe'}")
    print(f"  after ltx-ws patch — HannSinc  : {'still broken' if bwe_bad2 else 'safe'}")

    print()
    if broken and (up_bad or bwe_bad):
        print(
            "Diagnosis: this environment would produce T2V/I2V audio hiss without "
            "the ltx-ws runtime patch (or ltx-core-mlx ≥ 0.14.19)."
        )
        print(
            "Fix: upgrade packages per README, or pull latest ltx-ws (patch is automatic), "
            "and rebuild any packaged .app still on ltx-core-mlx 0.14.9."
        )
        print("Note: A2V remuxes source audio (no vocoder) — if only T2V hisses, this is it.")
        return 2
    if broken and not up_bad2 and not bwe_bad2:
        print(
            "Diagnosis: Metal scatter is broken, but vocoder workaround is active — "
            "generated audio should be fine."
        )
        return 0
    print("Diagnosis: no known vocoder-hiss configuration detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
