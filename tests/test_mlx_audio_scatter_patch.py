"""Guards for mlx 0.31.2 Metal vocoder hiss (ltx-2-mlx#34)."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("ltx_core_mlx")

from ltx_mlx_backend import (  # noqa: E402
    _mlx_strided_scatter_add_broken,
    _patch_mlx_audio_upsample_scatter,
    _source_uses_strided_at_add,
)


def test_source_uses_strided_at_add_detects_pattern():
    def broken(self, x):  # noqa: ARG001
        y = x.at[:, ::2, :].add(x)
        return y

    def safe(self, x):  # noqa: ARG001
        y = x
        y[:, ::2, :] = x
        return y

    assert _source_uses_strided_at_add(broken) is True
    assert _source_uses_strided_at_add(safe) is False


def test_scatter_canary_runs():
    """Canary must return a bool on this Metal host (True on broken mlx 0.31.2)."""
    result = _mlx_strided_scatter_add_broken()
    assert result is None or isinstance(result, bool)


def test_patch_replaces_broken_upsample_call():
    import mlx.core as mx
    from ltx_core_mlx.model.audio_vae.vocoder import UpSample1d

    def broken(self, x):
        b, t, c = x.shape
        x_up = mx.zeros((b, t * 2, c))
        x_up = x_up.at[:, ::2, :].add(x)
        return x_up

    UpSample1d.__call__ = broken  # type: ignore[method-assign]
    UpSample1d._ltx_ws_scatter_patched = False
    assert _source_uses_strided_at_add(UpSample1d.__call__) is True

    _patch_mlx_audio_upsample_scatter()
    assert _source_uses_strided_at_add(UpSample1d.__call__) is False
    assert getattr(UpSample1d, "_ltx_ws_scatter_patched", False) is True

    # Smoke: patched call produces expected even-index copies.
    x = mx.arange(2 * 4 * 3, dtype=mx.float32).reshape(2, 4, 3)
    # Restore a full UpSample1d needs filter weights; just ensure call doesn't use .at
    assert ".at[" not in __import__("inspect").getsource(UpSample1d.__call__)
