# SPDX-License-Identifier: Apache-2.0
"""
Local LTX-2.3 generation using ``ltx-2-mlx`` (MLX on Apple Silicon).

See: https://github.com/dgrauet/ltx-2-mlx
"""

from __future__ import annotations

import asyncio
import base64
import functools
import inspect
import logging
import mimetypes
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname, urlopen

log = logging.getLogger("fvserver")

LTX2_SPATIAL_ALIGN = 32

# Hugging Face repo id: ``org/name`` (used with huggingface_hub.snapshot_download,
# same file set as ``huggingface-cli download org/name``).
_HF_REPO_ID_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"
)
REPO_ROOT = Path(__file__).resolve().parent
VIDEOFENTANYL_MODELS_ENV = "VIDEOFENTANYL_MODELS"
VIDEOFENTANYL_LORA_DIR_ENV = "VIDEOFENTANYL_LORA_DIR"
MAX_REMOTE_INPUT_BYTES = 512 * 1024 * 1024  # 512 MiB safety ceiling for remote audio/video


@dataclass
class GenerationRequest:
    prompt: str
    image_data: dict | str | None = None
    audio_data: dict | str | None = None
    source_video_data: dict | str | None = None
    seed: int = 1024
    num_frames: int | None = None
    height: int | None = None
    width: int | None = None
    negative_prompt: str = ""
    mode: str = "generate"  # generate|a2v|retake|extend
    num_steps: int | None = None
    retake_start: int | None = None
    retake_end: int | None = None
    extend_frames: int | None = None
    extend_direction: str = "after"
    lora_specs: list[tuple[str, float]] | None = None
    video_conditioning_specs: list[tuple[dict | str, float]] | None = None
    job_id: str | None = None


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


def _model_snapshot_present(dest: Path) -> bool:
    """
    Heuristic to detect an already materialized HF snapshot in ``dest``.
    """
    if not dest.is_dir():
        return False
    try:
        has_config = (dest / "config.json").is_file() or (dest / "embedded_config.json").is_file()
        has_weights = any(dest.glob("*.safetensors"))
    except OSError:
        return False
    return bool(has_config and has_weights)


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


def _looks_like_models_dir_leaf(name: str) -> bool:
    """True if ``name`` is a single path segment (safe to join under ``models/``)."""
    s = (name or "").strip()
    if not s or s in (".", "..") or s.startswith(".."):
        return False
    if "/" in s or "\\" in s:
        return False
    return Path(s).name == s


def _path_candidates_for_user_string(user_path: str) -> list[Path]:
    """For a filesystem path string: absolutes resolve once; relatives try cwd then repo root.

    This fixes ``python /path/to/server.py`` started from ``$HOME`` where
    ``./models/foo`` must resolve next to the checkout, not under ``$HOME``.
    """
    raw = (user_path or "").strip()
    if not raw:
        return []
    p = Path(raw).expanduser()
    if p.is_absolute():
        return [p.resolve()]
    return [(Path.cwd() / p).resolve(), (REPO_ROOT / p).resolve()]


def _first_existing_dir(user_path: str) -> Path | None:
    for c in _path_candidates_for_user_string(user_path):
        if c.is_dir():
            return c
    return None


def _resolve_non_hf_disk_path(model: str, explicit_model_dir: str | None) -> str | None:
    """
    Resolve to an existing weights directory without calling the Hub.

    Tries: ``--model`` as a directory path (cwd, then repo root for relatives),
    then ``--model-dir`` the same way, then ``models/<model>/`` under cwd and
    under repo root for a shorthand leaf (e.g. ``ltx-2.3-mlx``).
    """
    raw = (model or "").strip()
    if not raw:
        return None

    hit = _first_existing_dir(raw)
    if hit is not None:
        return str(hit)

    md = (explicit_model_dir or "").strip()
    if md:
        hit = _first_existing_dir(md)
        if hit is not None:
            return str(hit)

    if _looks_like_models_dir_leaf(raw):
        leaf = Path(raw).name
        for base in (Path.cwd(), REPO_ROOT):
            candidate = (base / "models" / leaf).resolve()
            try:
                candidate.relative_to(base.resolve())
            except ValueError:
                continue
            if candidate.is_dir():
                return str(candidate)

    return None


def preview_mlx_weights_source(model: str, explicit_model_dir: str | None) -> str:
    """Where weights are expected on disk (for UI); may not exist yet for fresh HF pulls."""
    raw = (model or "").strip()
    got = _resolve_non_hf_disk_path(raw, explicit_model_dir)
    if got is not None:
        return got
    if looks_like_hf_repo_id(raw):
        return str(hf_local_weights_directory(raw, explicit_model_dir))
    return raw


def resolve_mlx_weights_directory(model: str, explicit_model_dir: str | None) -> str:
    """Resolve ``model`` and optional ``explicit_model_dir`` to an on-disk MLX weights tree."""
    raw = (model or "").strip()
    disk = _resolve_non_hf_disk_path(raw, explicit_model_dir)
    if disk is not None:
        return disk

    if looks_like_hf_repo_id(raw):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required to download MLX weights from Hugging Face. "
                "Install with:  pip install huggingface_hub\n"
                "Or use a local directory for --model."
            ) from e
        dest = hf_local_weights_directory(raw, explicit_model_dir)
        dest.mkdir(parents=True, exist_ok=True)
        if _model_snapshot_present(dest):
            log.info("Using existing local MLX snapshot for %r at %s", raw, dest)
            return str(dest)
        log.info(
            "Ensuring Hugging Face weights %r under %s "
            "(huggingface_hub.snapshot_download; same payload as `huggingface-cli download`) …",
            raw,
            dest,
        )
        _snapshot_download_weights(snapshot_download, raw, dest)
        return str(dest)

    return raw


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


def _download_remote_to_temp(
    url: str,
    prefix: str,
    suffix_hint: str = "",
    max_bytes: int | None = MAX_REMOTE_INPUT_BYTES,
) -> str:
    req_url = (url or "").strip()
    if not req_url.startswith(("http://", "https://")):
        raise ValueError(f"Unsupported remote input URL: {url!r}")
    with urlopen(req_url, timeout=180) as resp:
        if max_bytes is None:
            payload = resp.read()
        else:
            payload = resp.read(max_bytes + 1)
    if max_bytes is not None and len(payload) > max_bytes:
        raise RuntimeError(
            f"Remote media exceeds {max_bytes // (1024 * 1024)} MiB limit"
        )
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix_hint)
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
    return path


def _local_lora_cache_dir() -> Path:
    env = (os.environ.get(VIDEOFENTANYL_LORA_DIR_ENV) or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "loras").resolve()


def _pick_safetensors_file(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*.safetensors"))
    if not candidates:
        return None
    # Prefer explicit loras/ subdir when present.
    for c in candidates:
        if "loras" in {p.lower() for p in c.parts}:
            return c
    return candidates[0]


def _resolve_lora_path(spec: str) -> tuple[str, str | None]:
    """
    Resolve LoRA spec to a local safetensors path.
    Returns (path, cleanup_temp_path_or_none).
    """
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("Empty LoRA spec")

    p = Path(raw).expanduser()
    if p.is_file():
        return str(p.resolve()), None
    if raw.startswith(("http://", "https://")):
        parsed = urlparse(raw)
        # Support Hugging Face resolve URLs directly by routing through hf_hub_download,
        # which handles large files and cache efficiently.
        if parsed.netloc.endswith("huggingface.co") and "/resolve/" in parsed.path:
            parts = [p for p in parsed.path.strip("/").split("/") if p]
            # Expected: <repo_owner>/<repo_name>/resolve/<revision>/<filename...>
            if len(parts) >= 5 and parts[2] == "resolve":
                repo_id = f"{parts[0]}/{parts[1]}"
                revision = parts[3]
                filename = "/".join(parts[4:])
                try:
                    from huggingface_hub import hf_hub_download
                except ImportError as e:
                    raise RuntimeError(
                        "huggingface_hub is required to download LoRA from Hugging Face"
                    ) from e
                cache_root = _local_lora_cache_dir()
                cache_root.mkdir(parents=True, exist_ok=True)
                log.info(
                    "Downloading/using cached LoRA %s (%s @ %s) …",
                    repo_id,
                    filename,
                    revision,
                )
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    local_dir=str(cache_root / repo_id.replace("/", "__")),
                )
                return str(Path(local).resolve()), None

        # Generic URL fallback (no 512MiB cap for LoRA artifacts).
        tmp = _download_remote_to_temp(
            raw,
            "fvserver_lora_",
            ".safetensors",
            max_bytes=None,
        )
        return tmp, tmp

    if looks_like_hf_repo_id(raw):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required to download LoRA from Hugging Face"
            ) from e
        dest_root = _local_lora_cache_dir()
        dest = (dest_root / raw.replace("/", "__")).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        snap = _snapshot_download_weights(snapshot_download, raw, dest)
        snap_path = Path(snap)
        lora_file = _pick_safetensors_file(snap_path)
        if lora_file is None:
            raise RuntimeError(f"No .safetensors LoRA file found under {snap_path}")
        return str(lora_file.resolve()), None

    raise FileNotFoundError(f"LoRA spec not found or unsupported: {raw}")


def _decode_media_input(
    media_data: dict | str | None,
    *,
    temp_prefix: str,
    default_suffix: str,
) -> tuple[str | None, str | None]:
    """
    Resolve media input to a local path or URL.

    Returns: (resolved_path_or_url, temp_file_to_cleanup_or_none)
    """
    if media_data is None:
        return None, None

    if isinstance(media_data, str):
        raw = media_data.strip()
        if not raw:
            return None, None
        if raw.startswith(("http://", "https://")):
            tmp = _download_remote_to_temp(raw, temp_prefix, default_suffix)
            return tmp, tmp
        if raw.startswith("file://"):
            path = url2pathname(unquote(raw[7:]))
            if os.path.isfile(path):
                return path, None
            raise FileNotFoundError(f"File URL does not exist: {raw}")
        if os.path.isfile(raw):
            return raw, None
        raise FileNotFoundError(f"Media input not found: {raw}")

    if isinstance(media_data, dict):
        data_url = str(media_data.get("data_url") or "").strip()
        if not data_url:
            return None, None
        if data_url.startswith(("http://", "https://")):
            tmp = _download_remote_to_temp(data_url, temp_prefix, default_suffix)
            return tmp, tmp
        if data_url.startswith("file://"):
            path = url2pathname(unquote(data_url[7:]))
            if os.path.isfile(path):
                return path, None
            raise FileNotFoundError(f"File URL does not exist: {data_url}")
        if os.path.isfile(data_url):
            return data_url, None
        if data_url.startswith("data:"):
            header, encoded = data_url.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
        else:
            mime = str(media_data.get("mime_type") or "")
            encoded = data_url
        ext = mimetypes.guess_extension(mime) or default_suffix
        if ext == ".jpe":
            ext = ".jpg"
        fd, path = tempfile.mkstemp(prefix=temp_prefix, suffix=ext)
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(encoded))
        return path, path

    return None, None


def _decode_weighted_media_inputs(
    items: list[tuple[dict | str, float]] | None,
    *,
    temp_prefix: str,
    default_suffix: str,
) -> tuple[list[tuple[str, float]], list[str]]:
    decoded: list[tuple[str, float]] = []
    temps: list[str] = []
    for src, weight in (items or []):
        path, cleanup = _decode_media_input(
            src,
            temp_prefix=temp_prefix,
            default_suffix=default_suffix,
        )
        if path:
            decoded.append((path, float(weight)))
        if cleanup:
            temps.append(cleanup)
    return decoded, temps


def _invoke_generate_and_save(pipe: Any, **kwargs: Any) -> None:
    """
    Call ``pipe.generate_and_save`` while tolerating API drift between ltx-2-mlx versions.

    - Drops unsupported kwargs.
    - Maps ``num_steps`` -> ``steps`` when needed.
    """
    fn = getattr(pipe, "generate_and_save", None)
    if fn is None:
        raise RuntimeError(f"{type(pipe).__name__} has no generate_and_save()")

    sig = inspect.signature(fn)
    params = sig.parameters
    accepted = set(params.keys())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    call_kwargs = dict(kwargs)
    if "num_steps" in call_kwargs and "num_steps" not in accepted and "steps" in accepted:
        call_kwargs["steps"] = call_kwargs.pop("num_steps")

    if not has_varkw:
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted}

    fn(**call_kwargs)


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
        default_lora_specs: list[tuple[str, float]] | None = None,
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
        self.default_lora_specs = list(default_lora_specs or [])
        self.spill_dir = spill_dir
        self.low_memory = bool(low_memory)
        self._model_path: str | None = None
        self._pipe_classes: dict[str, Any] = {}
        self._pipes: dict[str, Any] = {}
        self._resolved_default_loras: list[tuple[str, float]] | None = None

    def _resolve_model_dir(self) -> str:
        return resolve_mlx_weights_directory(self.model, self.model_dir)

    def load(self) -> None:
        if self._model_path is not None:
            return
        try:
            import ltx_pipelines_mlx as lpm
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
        self._model_path = path
        self._pipe_classes = {
            "t2v": lpm.TextToVideoPipeline,
            "i2v": lpm.ImageToVideoPipeline,
            "a2v": lpm.AudioToVideoPipeline,
            "retake": lpm.RetakePipeline,
            "extend": lpm.ExtendPipeline,
        }
        ic_cls = getattr(lpm, "ICLoraPipeline", None)
        if ic_cls is not None:
            self._pipe_classes["ic_lora"] = ic_cls
        log.info("MLX model path resolved ✓ %s", path)

    def _get_pipe(self, key: str) -> Any:
        if key in self._pipes:
            return self._pipes[key]
        self.load()
        if self._model_path is None:
            raise RuntimeError("MLX model path not initialized")
        cls = self._pipe_classes.get(key)
        if cls is None:
            raise RuntimeError(f"Unsupported pipeline key: {key}")
        log.info("Loading MLX pipeline %s from %s …", key, self._model_path)
        pipe = cls(model_dir=self._model_path, low_memory=self.low_memory)
        pipe.load()
        self._pipes[key] = pipe
        log.info("MLX pipeline ready ✓ (%s)", key)
        return pipe

    def _resolve_lora_specs(self, specs: list[tuple[str, float]]) -> tuple[list[tuple[str, float]], list[str]]:
        resolved: list[tuple[str, float]] = []
        temps: list[str] = []
        for lora_spec, lora_scale in specs:
            lora_path, cleanup = _resolve_lora_path(str(lora_spec))
            resolved.append((lora_path, float(lora_scale)))
            if cleanup:
                temps.append(cleanup)
        return resolved, temps

    def ensure_default_loras_ready(self) -> None:
        """
        Resolve/download default LoRAs at startup when LoRA mode is enabled.
        """
        self.load()
        if not self.default_lora_specs:
            self._resolved_default_loras = []
            return
        resolved, temps = self._resolve_lora_specs(self.default_lora_specs)
        for tmp in temps:
            if tmp and os.path.isfile(tmp) and "fvserver_lora_" in tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        self._resolved_default_loras = resolved
        log.info("Resolved %d default LoRA(s) for global use", len(resolved))

    def model_progress_for_ws(self) -> dict[str, Any] | None:
        return None

    def default_lora_count(self) -> int:
        if self._resolved_default_loras is not None:
            return len(self._resolved_default_loras)
        return len(self.default_lora_specs)

    async def generate(
        self,
        prompt: str,
        image_data: dict | str | None = None,
        audio_data: dict | str | None = None,
        source_video_data: dict | str | None = None,
        seed: int = 1024,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        negative_prompt: str = "",
        mode: str = "generate",
        num_steps: int | None = None,
        retake_start: int | None = None,
        retake_end: int | None = None,
        extend_frames: int | None = None,
        extend_direction: str = "after",
        lora_specs: list[tuple[str, float]] | None = None,
        video_conditioning_specs: list[tuple[dict | str, float]] | None = None,
        *,
        job_id: str | None = None,
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self._generate_sync,
                GenerationRequest(
                    prompt=prompt,
                    image_data=image_data,
                    audio_data=audio_data,
                    source_video_data=source_video_data,
                    seed=seed,
                    num_frames=num_frames or self.num_frames,
                    height=height or self.height,
                    width=width or self.width,
                    negative_prompt=negative_prompt,
                    mode=mode or "generate",
                    num_steps=num_steps,
                    retake_start=retake_start,
                    retake_end=retake_end,
                    extend_frames=extend_frames,
                    extend_direction=extend_direction or "after",
                    lora_specs=lora_specs,
                    video_conditioning_specs=video_conditioning_specs,
                    job_id=job_id,
                ),
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

    def _generate_sync(self, req: GenerationRequest) -> str:
        del req.negative_prompt  # reserved for future CFG-enabled variants
        self.load()

        assert self._model_path is not None
        requested_height = int(req.height or self.height)
        requested_width = int(req.width or self.width)
        ah = _align_ltx2_spatial(requested_height)
        aw = _align_ltx2_spatial(requested_width)
        if ah != requested_height or aw != requested_width:
            log.warning(
                "LTX requires H×W divisible by %s; adjusted %s×%s → %s×%s",
                LTX2_SPATIAL_ALIGN,
                requested_height,
                requested_width,
                ah,
                aw,
            )
        height, width = ah, aw

        requested_num_frames = int(req.num_frames or self.num_frames)
        nf = _nearest_valid_frames(requested_num_frames)
        if nf != requested_num_frames:
            log.warning(
                "LTX requires (frames-1)%%8==0; adjusted frames %s → %s",
                requested_num_frames,
                nf,
            )
        mode = (req.mode or "generate").strip().lower()
        requested_steps = int(req.num_steps or self.inference_steps)
        steps = max(1, requested_steps)
        if steps != requested_steps:
            log.warning("LTX steps must be >=1; adjusted steps %s → %s", requested_steps, steps)
        effective_loras: list[tuple[str, float]] = []
        if self._resolved_default_loras is not None:
            effective_loras.extend(self._resolved_default_loras)
        else:
            effective_loras.extend(self.default_lora_specs)
        effective_loras.extend(req.lora_specs or [])
        resolved_loras: list[tuple[str, float]] = []

        tmp_image: str | None = None
        tmp_audio: str | None = None
        tmp_video: str | None = None
        tmp_video_conditioning_cleanup: list[str] = []
        tmp_lora_cleanup: list[str] = []
        prefix = f"fv_{req.job_id[:8]}_" if req.job_id else "fvserver_out_"
        tmpdir = tempfile.mkdtemp(prefix=prefix)
        out_path = os.path.join(tmpdir, "output.mp4")
        retained_tmpdir = False

        try:
            tmp_image, tmp_image_cleanup = _decode_media_input(
                req.image_data,
                temp_prefix="fvserver_img_",
                default_suffix=".jpg",
            )
            if not tmp_image and isinstance(req.image_data, dict):
                tmp_image = _decode_initial_image_dict(req.image_data)
                tmp_image_cleanup = tmp_image
            tmp_audio, tmp_audio_cleanup = _decode_media_input(
                req.audio_data,
                temp_prefix="fvserver_audio_",
                default_suffix=".wav",
            )
            tmp_video, tmp_video_cleanup = _decode_media_input(
                req.source_video_data,
                temp_prefix="fvserver_video_",
                default_suffix=".mp4",
            )
            vc_items, vc_cleanup = _decode_weighted_media_inputs(
                req.video_conditioning_specs,
                temp_prefix="fvserver_vcond_",
                default_suffix=".mp4",
            )
            tmp_video_conditioning_cleanup = vc_cleanup
            if self._resolved_default_loras is not None and not req.lora_specs:
                resolved_loras = list(self._resolved_default_loras)
            else:
                for lora_spec, lora_scale in effective_loras:
                    lora_path, lora_cleanup = _resolve_lora_path(str(lora_spec))
                    resolved_loras.append((lora_path, float(lora_scale)))
                    if lora_cleanup:
                        tmp_lora_cleanup.append(lora_cleanup)
            log.info(
                "Generation effective params: mode=%s seed=%s size=%sx%s frames=%s steps=%s "
                "(requested size=%sx%s frames=%s steps=%s) image=%s audio=%s video=%s "
                "retake=%s-%s extend=%s/%s vcond=%s loras=%s model_path=%s",
                mode,
                int(req.seed),
                height,
                width,
                nf,
                steps,
                requested_height,
                requested_width,
                requested_num_frames,
                requested_steps,
                "yes" if tmp_image else "no",
                "yes" if tmp_audio else "no",
                "yes" if tmp_video else "no",
                req.retake_start if req.retake_start is not None else "-",
                req.retake_end if req.retake_end is not None else "-",
                req.extend_frames if req.extend_frames is not None else "-",
                (req.extend_direction or "after").strip().lower(),
                len(vc_items),
                len(resolved_loras),
                self._model_path,
            )
            if resolved_loras:
                log.info(
                    "Applying %d LoRA(s) for mode=%s (request=%d, defaults=%d)",
                    len(resolved_loras),
                    mode,
                    len(req.lora_specs or []),
                    self.default_lora_count(),
                )

            try:
                if mode == "a2v":
                    pipe = self._get_pipe("a2v")
                    _invoke_generate_and_save(
                        pipe,
                        prompt=req.prompt,
                        output_path=out_path,
                        audio_path=tmp_audio,
                        image=tmp_image,
                        height=height,
                        width=width,
                        num_frames=nf,
                        seed=int(req.seed),
                        num_steps=steps,
                        lora_paths=resolved_loras,
                    )
                elif mode == "retake":
                    if not tmp_video:
                        raise RuntimeError("retake mode requires source video input")
                    start_frame = int(req.retake_start if req.retake_start is not None else 1)
                    end_frame = int(req.retake_end if req.retake_end is not None else start_frame)
                    pipe = self._get_pipe("retake")
                    _invoke_generate_and_save(
                        pipe,
                        prompt=req.prompt,
                        output_path=out_path,
                        video_path=tmp_video,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        height=height,
                        width=width,
                        num_frames=nf,
                        seed=int(req.seed),
                        num_steps=steps,
                        lora_paths=resolved_loras,
                    )
                elif mode == "extend":
                    if not tmp_video:
                        raise RuntimeError("extend mode requires source video input")
                    ext_frames = int(req.extend_frames if req.extend_frames is not None else 2)
                    direction = (req.extend_direction or "after").strip().lower()
                    pipe = self._get_pipe("extend")
                    _invoke_generate_and_save(
                        pipe,
                        prompt=req.prompt,
                        output_path=out_path,
                        video_path=tmp_video,
                        extend_frames=ext_frames,
                        direction=direction,
                        height=height,
                        width=width,
                        num_frames=nf,
                        seed=int(req.seed),
                        num_steps=steps,
                        lora_paths=resolved_loras,
                    )
                elif mode == "ic_lora":
                    if not resolved_loras:
                        raise RuntimeError("ic_lora mode requires at least one LoRA spec")
                    if not vc_items:
                        raise RuntimeError("ic_lora mode requires video_conditioning entries")
                    pipe = self._get_pipe("ic_lora")
                    _invoke_generate_and_save(
                        pipe,
                        prompt=req.prompt,
                        output_path=out_path,
                        lora_paths=[(str(p), float(s)) for p, s in resolved_loras],
                        video_conditioning=[(str(p), float(s)) for p, s in vc_items],
                        height=height,
                        width=width,
                        num_frames=nf,
                        seed=int(req.seed),
                        num_steps=steps,
                    )
                elif tmp_image:
                    pipe = self._get_pipe("i2v")
                    _invoke_generate_and_save(
                        pipe,
                        prompt=req.prompt,
                        output_path=out_path,
                        image=tmp_image,
                        height=height,
                        width=width,
                        num_frames=nf,
                        seed=int(req.seed),
                        num_steps=steps,
                        lora_paths=resolved_loras,
                    )
                else:
                    pipe = self._get_pipe("t2v")
                    _invoke_generate_and_save(
                        pipe,
                        prompt=req.prompt,
                        output_path=out_path,
                        height=height,
                        width=width,
                        num_frames=nf,
                        seed=int(req.seed),
                        num_steps=steps,
                        lora_paths=resolved_loras,
                    )
            except BaseException:
                self._salvage_mp4_to_spill(tmpdir, out_path, req.job_id, req.prompt, "ENCODE_FAIL")
                raise

            video_path = out_path
            if not os.path.exists(video_path):
                self._salvage_mp4_to_spill(
                    tmpdir, out_path, req.job_id, req.prompt, "MISSING_OUTPUT",
                )
                raise RuntimeError(
                    f"Generation completed but output file not found: {video_path}"
                )
            retained_tmpdir = True
            return video_path

        finally:
            for tmp, marker in (
                (tmp_image, "fvserver_img_"),
                (tmp_audio, "fvserver_audio_"),
                (tmp_video, "fvserver_video_"),
            ):
                if tmp and os.path.isfile(tmp) and marker in tmp:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            for tmp in tmp_video_conditioning_cleanup:
                if tmp and os.path.isfile(tmp) and "fvserver_vcond_" in tmp:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            for tmp in tmp_lora_cleanup:
                if tmp and os.path.isfile(tmp) and "fvserver_lora_" in tmp:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            if not retained_tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
