# SPDX-License-Identifier: Apache-2.0
"""
Local LTX-2.3 generation using ``ltx-2-mlx`` (MLX on Apple Silicon).

See: https://github.com/dgrauet/ltx-2-mlx
"""

from __future__ import annotations

import asyncio
import base64
import functools
import logging
import mimetypes
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("fvserver")

LTX2_SPATIAL_ALIGN = 32

# Hugging Face repo id: ``org/name`` (used with huggingface_hub.snapshot_download,
# same file set as ``huggingface-cli download org/name``).
_HF_REPO_ID_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"
)
REPO_ROOT = Path(__file__).resolve().parent
VIDEOFENTANYL_MODELS_ENV = "VIDEOFENTANYL_MODELS"


def looks_like_hf_repo_id(model: str) -> bool:
    """True if ``model`` looks like ``author/repo`` and is not an existing directory path."""
    s = (model or "").strip()
    if not s or _HF_REPO_ID_RE.match(s) is None:
        return False
    p = Path(s).expanduser()
    if p.is_dir():
        return False
    return True


def _snapshot_download_weights(snapshot_download: Any, repo_id: str, dest: Path) -> str:
    """Call ``snapshot_download`` with kwargs compatible across huggingface_hub versions."""
    import inspect

    kw: dict[str, Any] = {"repo_id": repo_id, "local_dir": str(dest)}
    sig = inspect.signature(snapshot_download)
    if "resume_download" in sig.parameters:
        kw["resume_download"] = True
    if "local_dir_use_symlinks" in sig.parameters:
        kw["local_dir_use_symlinks"] = False
    out = snapshot_download(**kw)
    return str(Path(out).resolve())


def hf_local_weights_directory(repo_id: str, explicit_model_dir: str | None) -> Path:
    """
    Directory where we store a full ``snapshot_download`` for ``repo_id``.

    If ``explicit_model_dir`` is set, that path is used. Otherwise:
    ``$VIDEOFENTANYL_MODELS/<org>__<name>/`` when the env var is set, else
    ``<repo_root>/models/<org>__<name>/``.
    """
    rid = repo_id.strip()
    if explicit_model_dir:
        return Path(explicit_model_dir).expanduser().resolve()
    env = os.environ.get(VIDEOFENTANYL_MODELS_ENV, "").strip()
    root = Path(env).expanduser().resolve() if env else (REPO_ROOT / "models")
    safe = rid.replace("/", "__")
    return (root / safe).resolve()


def _spill_slug(prompt: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^\w\s-]+", "", prompt.lower().strip())[:maxlen]
    s = re.sub(r"[\s_]+", "_", s).strip("_")
    return s or "clip"


def _largest_mp4_under(root: Path) -> Path | None:
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


def _align_ltx2_spatial(n: int, align: int = LTX2_SPATIAL_ALIGN) -> int:
    if n < align:
        return align
    lower = (n // align) * align
    upper = lower + align
    return lower if (n - lower) <= (upper - n) else upper


def _nearest_valid_frames(n: int) -> int:
    if n < 9:
        return 9
    remainder = (n - 1) % 8
    if remainder == 0:
        return n
    lower = n - remainder
    upper = lower + 8
    return lower if (n - lower) <= (upper - n) else upper


def _decode_initial_image_dict(image_data: dict) -> str:
    """Data URL / path / base64 → path or URL (same contract as ``server._decode_initial_image``)."""
    data_url: str = (image_data.get("data_url") or "").strip()
    if data_url.startswith(("http://", "https://")):
        return data_url
    if data_url.startswith("file://"):
        from urllib.parse import unquote
        from urllib.request import url2pathname

        path = url2pathname(unquote(data_url[7:]))
        if os.path.isfile(path):
            return path
    if data_url and os.path.isfile(data_url):
        return data_url

    if data_url.startswith("data:"):
        header, encoded = data_url.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
    else:
        mime = image_data.get("mime_type", "image/jpeg")
        encoded = data_url

    ext = mimetypes.guess_extension(mime) or ".jpg"
    if ext == ".jpe":
        ext = ".jpg"

    fd, path = tempfile.mkstemp(suffix=ext, prefix="fvserver_img_")
    with os.fdopen(fd, "wb") as f:
        f.write(base64.b64decode(encoded))
    return path


class LocalVideoGenerator:
    """
    ``ImageToVideoPipeline`` from ``ltx-2-mlx``: text-to-video when no image,
    image-to-video otherwise. Weights loaded once at ``load()``.
    """

    def __init__(
        self,
        model: str,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        model_dir: str | None,
        inference_steps: int,
        spill_dir: Path | None = None,
        low_memory: bool = False,
    ) -> None:
        self.model = model
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.fps = float(fps)
        self.model_dir = model_dir
        self.inference_steps = max(1, int(inference_steps))
        self.spill_dir = spill_dir
        self.low_memory = bool(low_memory)
        self._pipe = None

    def _resolve_model_dir(self) -> str:
        raw = (self.model or "").strip()
        local = Path(raw).expanduser()
        if local.is_dir():
            return str(local.resolve())

        if looks_like_hf_repo_id(raw):
            try:
                from huggingface_hub import snapshot_download
            except ImportError as e:
                raise RuntimeError(
                    "huggingface_hub is required to download MLX weights from Hugging Face. "
                    "Install with:  pip install huggingface_hub\n"
                    "Or use a local directory for --model."
                ) from e
            dest = hf_local_weights_directory(raw, self.model_dir)
            dest.mkdir(parents=True, exist_ok=True)
            log.info(
                "Ensuring Hugging Face weights %r under %s "
                "(huggingface_hub.snapshot_download; same payload as `huggingface-cli download`) …",
                raw,
                dest,
            )
            _snapshot_download_weights(snapshot_download, raw, dest)
            return str(dest)

        return raw

    def load(self) -> None:
        if self._pipe is not None:
            return
        try:
            from ltx_pipelines_mlx import ImageToVideoPipeline
        except ImportError as e:
            raise RuntimeError(
                "Missing ltx_pipelines_mlx. Install the MLX monorepo packages, e.g.:\n"
                "  uv pip install \\\n"
                "    \"ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git"
                "#subdirectory=packages/ltx-core-mlx\" \\\n"
                "    \"ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git"
                "#subdirectory=packages/ltx-pipelines-mlx\""
            ) from e
        path = self._resolve_model_dir()
        log.info("Loading MLX LTX weights from %s …", path)
        self._pipe = ImageToVideoPipeline(model_dir=path, low_memory=self.low_memory)
        self._pipe.load()
        log.info("MLX pipeline ready ✓")

    def model_progress_for_ws(self) -> dict[str, Any] | None:
        return None

    async def generate(
        self,
        prompt: str,
        image_data: dict | str | None = None,
        seed: int = 1024,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        negative_prompt: str = "",
        *,
        job_id: str | None = None,
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self._generate_sync,
                prompt,
                image_data,
                seed,
                num_frames or self.num_frames,
                height or self.height,
                width or self.width,
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
        prompt: str,
        image_data: dict | str | None,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        negative_prompt: str,
        job_id: str | None = None,
    ) -> str:
        del negative_prompt  # one-stage MLX distilled path; reserved for future CFG APIs
        self.load()

        ah = _align_ltx2_spatial(height)
        aw = _align_ltx2_spatial(width)
        if ah != height or aw != width:
            log.warning(
                "LTX requires H×W divisible by %s; adjusted %s×%s → %s×%s",
                LTX2_SPATIAL_ALIGN,
                height,
                width,
                ah,
                aw,
            )
            height, width = ah, aw

        nf = _nearest_valid_frames(int(num_frames))

        tmp_image: str | None = None
        prefix = f"fv_{job_id[:8]}_" if job_id else "fvserver_out_"
        tmpdir = tempfile.mkdtemp(prefix=prefix)
        out_path = os.path.join(tmpdir, "output.mp4")
        retained_tmpdir = False

        try:
            if isinstance(image_data, str) and image_data.strip():
                p = image_data.strip()
                if os.path.isfile(p) or p.startswith(("http://", "https://")):
                    tmp_image = p
            elif isinstance(image_data, dict) and image_data:
                tmp_image = _decode_initial_image_dict(image_data)

            assert self._pipe is not None
            try:
                if tmp_image:
                    self._pipe.generate_and_save(
                        prompt=prompt,
                        output_path=out_path,
                        image=tmp_image,
                        height=height,
                        width=width,
                        num_frames=nf,
                        seed=int(seed),
                        num_steps=self.inference_steps,
                    )
                else:
                    self._pipe.generate_and_save(
                        prompt=prompt,
                        output_path=out_path,
                        image=None,
                        height=height,
                        width=width,
                        num_frames=nf,
                        seed=int(seed),
                        num_steps=self.inference_steps,
                    )
            except BaseException:
                self._salvage_mp4_to_spill(tmpdir, out_path, job_id, prompt, "ENCODE_FAIL")
                raise

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
            if tmp_image and os.path.isfile(tmp_image) and "fvserver_img_" in tmp_image:
                try:
                    os.unlink(tmp_image)
                except OSError:
                    pass
            if not retained_tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
