#!/usr/bin/env python3
"""
mcp_server.py — MCP interface for local ltx-ws generation.

Exposes standardized MCP tools so any MCP client can drive the existing
WebSocket workflow implemented by ``server.py`` + ``videofentanyl.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from videofentanyl import (
    CHAIN_METHOD_AUTOCONTINUE,
    CHAIN_METHOD_NATIVE_EXTEND,
    VALID_CHAIN_METHODS,
    GenerationParams,
    Job,
    JobStatus,
    VideoSession,
    extract_last_frame,
    load_image_payload,
    load_media_payload,
    sanitize_filename,
    try_autoconcat_clips,
    try_finalize_native_extend_chain,
)

DEFAULT_SERVER_URL = "ws://127.0.0.1:8765/ws"
DEFAULT_OUTPUT_DIR = "mcp_outputs"
DEFAULT_PREFIX = "ltx_mcp"

_SERVER_URL = DEFAULT_SERVER_URL
_OUTPUT_DIR = Path(DEFAULT_OUTPUT_DIR)
_VERBOSE = False

mcp = FastMCP("ltx-ws")


def _ts_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _new_output_path(prompt: str, output_dir: Path, prefix: str) -> Path:
    slug = sanitize_filename(prompt) or "clip"
    return output_dir / f"{prefix}_{slug}_{_ts_slug()}.mp4"


def _normalize_mode(mode: str) -> str:
    val = (mode or "generate").strip().lower()
    allowed = {
        "generate",
        "a2v",
        "retake",
        "extend",
        "ic_lora",
        "keyframe",
        "lipdub",
        "face_swap",
        "id_lora",
    }
    if val not in allowed:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(allowed)}")
    return val


def _build_params(
    *,
    prompt: str,
    mode: str,
    image: str | None,
    audio: str | None,
    video: str | None,
    seed: int | None,
    num_frames: int | None,
    height: int | None,
    width: int | None,
    num_steps: int | None,
    retake_start: int | None,
    retake_end: int | None,
    extend_frames: int | None,
    extend_direction: str | None,
    lora_specs: list[list[Any]] | None,
    video_conditioning: list[list[Any]] | None,
    end_image: str | None = None,
    enhance_prompt: bool = False,
    pipeline_profile: str = "distilled",
    cfg_scale: float | None = None,
    stg_scale: float | None = None,
    stage2_steps: int | None = None,
    no_regen_audio: bool = False,
    reference_strength: float | None = None,
    skip_stage_2: bool = False,
    upsample_only: bool = False,
    modality_scale: float | None = None,
    audio_cfg_scale: float | None = None,
    identity_guidance_scale: float | None = None,
) -> GenerationParams:
    normalized_mode = _normalize_mode(mode)

    image_payload = load_image_payload(image) if image else None
    end_image_payload = load_image_payload(end_image) if end_image else None
    audio_payload = load_media_payload(audio, kind="audio") if audio else None
    video_payload = load_media_payload(video, kind="video") if video else None

    parsed_loras: list[tuple[str, float]] = []
    for item in lora_specs or []:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each lora_specs item must be [path_or_repo_or_url, scale]")
        parsed_loras.append((str(item[0]).strip(), float(item[1])))

    parsed_vcond: list[tuple[dict, float]] = []
    for item in video_conditioning or []:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each video_conditioning item must be [video_path_or_url, scale]")
        payload = load_media_payload(str(item[0]).strip(), kind="video")
        parsed_vcond.append((payload, float(item[1])))

    return GenerationParams(
        prompt=prompt.strip(),
        preset_id="simple_custom_prompt",
        enhancement_enabled=False,
        single_clip_mode=True,
        auto_extension_enabled=False,
        loop_generation_enabled=False,
        initial_image=image_payload,
        end_image=end_image_payload,
        seed=seed,
        num_frames=num_frames,
        height=height,
        width=width,
        num_steps=num_steps,
        generation_mode=normalized_mode,
        audio_input=audio_payload,
        source_video=video_payload,
        retake_start=retake_start,
        retake_end=retake_end,
        extend_frames=extend_frames,
        extend_direction=extend_direction,
        lora_specs=parsed_loras,
        video_conditioning_specs=parsed_vcond,
        enhance_prompt=enhance_prompt,
        pipeline_profile=pipeline_profile,
        cfg_scale=cfg_scale,
        stg_scale=stg_scale,
        stage2_steps=stage2_steps,
        no_regen_audio=no_regen_audio,
        reference_strength=reference_strength,
        skip_stage_2=bool(skip_stage_2),
        upsample_only=bool(upsample_only),
        modality_scale=modality_scale,
        audio_cfg_scale=audio_cfg_scale,
        identity_guidance_scale=identity_guidance_scale,
    )


async def _run_job(params: GenerationParams, output_path: Path) -> dict[str, Any]:
    job = Job(
        id=1,
        params=params,
        output_path=output_path,
        max_attempts=1,
    )
    job.status = JobStatus.RUNNING
    job.started_at = time.time()

    session = VideoSession(job=job, mode="ltx", verbose=_VERBOSE)
    ok = await session.run(idle_timeout=None)

    job.finished_at = time.time()
    job.status = JobStatus.DONE if ok else JobStatus.FAILED

    if not ok:
        raise RuntimeError(job.error or "Generation failed")

    return {
        "output_path": str(output_path.resolve()),
        "bytes": int(job.file_bytes),
        "chunks": int(job.chunk_count),
        "segments": int(job.segment_count),
        "elapsed_s": round(job.elapsed, 3),
        "ttff_ms": job.ttff_ms,
        "generation_ms": job.gen_latency_ms,
        "e2e_ms": job.e2e_latency_ms,
    }


def _build_multi_job(
    *,
    job_id: int,
    prompt: str,
    mode: str,
    image_payload: dict | None,
    audio_payload: dict | None,
    video_payload: dict | None,
    seed: int | None,
    num_frames: int | None,
    height: int | None,
    width: int | None,
    num_steps: int | None,
    retake_start: int | None,
    retake_end: int | None,
    extend_frames: int | None,
    extend_direction: str | None,
    lora_specs: list[tuple[str, float]],
    video_conditioning_specs: list[tuple[dict, float]],
    output_path: Path,
    skip_stage_2: bool = False,
    upsample_only: bool = False,
    modality_scale: float | None = None,
    end_image_payload: dict | None = None,
    no_regen_audio: bool = False,
    reference_strength: float | None = None,
    cfg_scale: float | None = None,
    stg_scale: float | None = None,
    audio_cfg_scale: float | None = None,
    identity_guidance_scale: float | None = None,
) -> Job:
    params = GenerationParams(
        prompt=prompt.strip(),
        preset_id="simple_custom_prompt",
        enhancement_enabled=False,
        single_clip_mode=True,
        auto_extension_enabled=False,
        loop_generation_enabled=False,
        initial_image=image_payload,
        end_image=end_image_payload,
        seed=seed,
        num_frames=num_frames,
        height=height,
        width=width,
        num_steps=num_steps,
        generation_mode=mode,
        audio_input=audio_payload,
        source_video=video_payload,
        retake_start=retake_start,
        retake_end=retake_end,
        extend_frames=extend_frames,
        extend_direction=extend_direction,
        lora_specs=lora_specs,
        video_conditioning_specs=video_conditioning_specs,
        skip_stage_2=bool(skip_stage_2),
        upsample_only=bool(upsample_only),
        modality_scale=modality_scale,
        no_regen_audio=bool(no_regen_audio),
        reference_strength=reference_strength,
        cfg_scale=cfg_scale,
        stg_scale=stg_scale,
        audio_cfg_scale=audio_cfg_scale,
        identity_guidance_scale=identity_guidance_scale,
    )
    return Job(
        id=job_id,
        params=params,
        output_path=output_path,
        max_attempts=1,
    )


@mcp.tool()
async def ltx_generate_video(
    prompt: str,
    mode: str = "generate",
    image: str | None = None,
    audio: str | None = None,
    video: str | None = None,
    seed: int | None = None,
    num_frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
    num_steps: int | None = None,
    retake_start: int | None = None,
    retake_end: int | None = None,
    extend_frames: int | None = None,
    extend_direction: str | None = None,
    lora_specs: list[list[Any]] | None = None,
    video_conditioning: list[list[Any]] | None = None,
    end_image: str | None = None,
    enhance_prompt: bool = False,
    pipeline_profile: str = "distilled",
    cfg_scale: float | None = None,
    stg_scale: float | None = None,
    stage2_steps: int | None = None,
    no_regen_audio: bool = False,
    reference_strength: float | None = None,
    skip_stage_2: bool = False,
    upsample_only: bool = False,
    modality_scale: float | None = None,
    audio_cfg_scale: float | None = None,
    identity_guidance_scale: float | None = None,
    output_filename: str | None = None,
) -> dict[str, Any]:
    """
    Generate a single video clip through ltx-ws and return file/latency metadata.

    For mode=ic_lora (HDR): video_conditioning and image are independently optional
    (T2V / V2V / I2V). Pass HDR LoRA in lora_specs. skip_stage_2 skips the upscale stage.

    For mode=id_lora: requires image (first frame) + audio (~5s identity reference) and
    a structured prompt ``[VISUAL] / [SPEECH] / [SOUNDS]``. Optional lora_specs defaults
    to CelebV-HQ ID-LoRA. Reference audio is **voice identity only** — spoken words come
    from ``[SPEECH]``, not the WAV. Balanced defaults: 20 steps, stg=0, modality=1;
    pass num_steps=30 + stg_scale=1 + modality_scale=3 for the faithful preset.
    Tunable: cfg_scale (video), audio_cfg_scale, identity_guidance_scale.
    skip_stage_2 / upsample_only are faster preview escapes.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")

    if mode == "a2v" and not audio:
        raise ValueError("mode=a2v requires audio")
    if mode == "retake":
        if not video:
            raise ValueError("mode=retake requires video")
        if retake_start is None or retake_end is None:
            raise ValueError("mode=retake requires retake_start and retake_end")
        if int(retake_end) <= int(retake_start):
            raise ValueError("mode=retake requires retake_end > retake_start (end is exclusive)")
    if mode == "extend":
        if not video:
            raise ValueError("mode=extend requires video")
        if extend_frames is None:
            raise ValueError("mode=extend requires extend_frames")
    if mode == "ic_lora":
        if not lora_specs:
            raise ValueError("mode=ic_lora requires lora_specs")
    if mode == "keyframe":
        if not image or not end_image:
            raise ValueError("mode=keyframe requires image and end_image")
    if mode == "lipdub":
        if not video:
            raise ValueError("mode=lipdub requires video (reference video)")
        if not lora_specs or len(lora_specs) != 1:
            raise ValueError("mode=lipdub requires exactly one lora_specs entry")
    if mode == "face_swap":
        if not video:
            raise ValueError("mode=face_swap requires video (reference video)")
        if not image:
            raise ValueError("mode=face_swap requires image (face identity)")
        if not lora_specs or len(lora_specs) != 1:
            raise ValueError("mode=face_swap requires exactly one lora_specs entry (head swap LoRA)")
    if mode == "id_lora":
        if not image:
            raise ValueError("mode=id_lora requires image (first-frame identity)")
        if not audio:
            raise ValueError("mode=id_lora requires audio (identity reference, ~5s)")
        if lora_specs is not None and len(lora_specs) != 1:
            raise ValueError("mode=id_lora accepts at most one lora_specs entry (ID-LoRA adapter)")
        if skip_stage_2 and upsample_only:
            raise ValueError("skip_stage_2 and upsample_only are mutually exclusive")

    params = _build_params(
        prompt=prompt,
        mode=mode,
        image=image,
        audio=audio,
        video=video,
        seed=seed,
        num_frames=num_frames,
        height=height,
        width=width,
        num_steps=num_steps,
        retake_start=retake_start,
        retake_end=retake_end,
        extend_frames=extend_frames,
        extend_direction=extend_direction,
        lora_specs=lora_specs,
        video_conditioning=video_conditioning,
        end_image=end_image,
        enhance_prompt=enhance_prompt,
        pipeline_profile=pipeline_profile,
        cfg_scale=cfg_scale,
        stg_scale=stg_scale,
        stage2_steps=stage2_steps,
        no_regen_audio=no_regen_audio,
        reference_strength=reference_strength,
        skip_stage_2=skip_stage_2,
        upsample_only=upsample_only,
        modality_scale=modality_scale,
        audio_cfg_scale=audio_cfg_scale,
        identity_guidance_scale=identity_guidance_scale,
    )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_filename:
        output_path = _OUTPUT_DIR / output_filename
    else:
        output_path = _new_output_path(prompt, _OUTPUT_DIR, DEFAULT_PREFIX)

    # Route VideoSession to local ltx-ws endpoint configured at startup.
    import videofentanyl as vf_mod

    previous = vf_mod._SERVER_OVERRIDE
    vf_mod._SERVER_OVERRIDE = _SERVER_URL
    try:
        result = await _run_job(params=params, output_path=output_path)
    finally:
        vf_mod._SERVER_OVERRIDE = previous

    return {
        "ok": True,
        "server": _SERVER_URL,
        **result,
    }


@mcp.tool()
async def ltx_generate_sequence(
    prompts: list[str],
    mode: str = "generate",
    autocontinue: bool = True,
    chain_method: str = CHAIN_METHOD_AUTOCONTINUE,
    autoconcat: bool = False,
    image: str | None = None,
    audio: str | None = None,
    video: str | None = None,
    seed: int | None = None,
    num_frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
    num_steps: int | None = None,
    retake_start: int | None = None,
    retake_end: int | None = None,
    extend_frames: int | None = None,
    extend_direction: str | None = None,
    lora_specs: list[list[Any]] | None = None,
    video_conditioning: list[list[Any]] | None = None,
    end_image: str | None = None,
    enhance_prompt: bool = False,
    pipeline_profile: str = "distilled",
    cfg_scale: float | None = None,
    stg_scale: float | None = None,
    stage2_steps: int | None = None,
    no_regen_audio: bool = False,
    reference_strength: float | None = None,
    skip_stage_2: bool = False,
    upsample_only: bool = False,
    modality_scale: float | None = None,
    audio_cfg_scale: float | None = None,
    identity_guidance_scale: float | None = None,
    output_prefix: str = DEFAULT_PREFIX,
) -> dict[str, Any]:
    """
    Generate multiple clips sequentially. Chain clip 2+ via autocontinue (last frame)
    or native_extend (ltx-2-mlx extend_from_video on the prior MP4).
    """
    clean_prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
    if not clean_prompts:
        raise ValueError("prompts must contain at least one non-empty prompt")

    normalized_mode = _normalize_mode(mode)
    if normalized_mode == "a2v" and not audio:
        raise ValueError("mode=a2v requires audio")
    if normalized_mode == "retake":
        if not video:
            raise ValueError("mode=retake requires video")
        if retake_start is None or retake_end is None:
            raise ValueError("mode=retake requires retake_start and retake_end")
        if int(retake_end) <= int(retake_start):
            raise ValueError("mode=retake requires retake_end > retake_start (end is exclusive)")
    if normalized_mode == "extend":
        if not video:
            raise ValueError("mode=extend requires video")
        if extend_frames is None:
            raise ValueError("mode=extend requires extend_frames")
    if normalized_mode == "ic_lora":
        if not lora_specs:
            raise ValueError("mode=ic_lora requires lora_specs")
    if normalized_mode == "keyframe":
        if not image or not end_image:
            raise ValueError("mode=keyframe requires image and end_image")
    if normalized_mode == "lipdub":
        if not video:
            raise ValueError("mode=lipdub requires video (reference video)")
        if not lora_specs or len(lora_specs) != 1:
            raise ValueError("mode=lipdub requires exactly one lora_specs entry")
    if normalized_mode == "face_swap":
        if not video:
            raise ValueError("mode=face_swap requires video (reference video)")
        if not image:
            raise ValueError("mode=face_swap requires image (face identity)")
        if not lora_specs or len(lora_specs) != 1:
            raise ValueError("mode=face_swap requires exactly one lora_specs entry (head swap LoRA)")
    if normalized_mode == "id_lora":
        if not image:
            raise ValueError("mode=id_lora requires image (first-frame identity)")
        if not audio:
            raise ValueError("mode=id_lora requires audio (identity reference, ~5s)")
        if lora_specs is not None and len(lora_specs) != 1:
            raise ValueError("mode=id_lora accepts at most one lora_specs entry (ID-LoRA adapter)")

    method = (chain_method or CHAIN_METHOD_AUTOCONTINUE).strip().lower()
    if method not in VALID_CHAIN_METHODS:
        method = CHAIN_METHOD_AUTOCONTINUE
    if method == CHAIN_METHOD_NATIVE_EXTEND:
        if not autocontinue:
            raise ValueError("chain_method=native_extend requires autocontinue=true")
        if normalized_mode != "generate" and not image:
            raise ValueError(
                "chain_method=native_extend supports mode=generate (optionally with image on clip 1)"
            )

    image_payload = load_image_payload(image) if image else None
    end_image_payload = load_image_payload(end_image) if end_image else None
    audio_payload = load_media_payload(audio, kind="audio") if audio else None
    video_payload = load_media_payload(video, kind="video") if video else None

    parsed_loras: list[tuple[str, float]] = []
    for item in lora_specs or []:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each lora_specs item must be [path_or_repo_or_url, scale]")
        parsed_loras.append((str(item[0]).strip(), float(item[1])))

    parsed_vcond: list[tuple[dict, float]] = []
    for item in video_conditioning or []:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each video_conditioning item must be [video_path_or_url, scale]")
        payload = load_media_payload(str(item[0]).strip(), kind="video")
        parsed_vcond.append((payload, float(item[1])))

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs: list[Job] = []
    for i, p in enumerate(clean_prompts, start=1):
        clip_slug = sanitize_filename(p) or "clip"
        output_path = _OUTPUT_DIR / f"{output_prefix}_{i:03d}_{clip_slug}_{_ts_slug()}.mp4"
        jobs.append(
            _build_multi_job(
                job_id=i,
                prompt=p,
                mode=normalized_mode,
                image_payload=image_payload,
                audio_payload=audio_payload,
                video_payload=video_payload,
                seed=seed,
                num_frames=num_frames,
                height=height,
                width=width,
                num_steps=num_steps,
                retake_start=retake_start,
                retake_end=retake_end,
                extend_frames=extend_frames,
                extend_direction=extend_direction,
                lora_specs=parsed_loras,
                video_conditioning_specs=parsed_vcond,
                output_path=output_path,
                skip_stage_2=skip_stage_2,
                upsample_only=upsample_only,
                modality_scale=modality_scale,
                end_image_payload=end_image_payload,
                no_regen_audio=no_regen_audio,
                reference_strength=reference_strength,
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                audio_cfg_scale=audio_cfg_scale,
                identity_guidance_scale=identity_guidance_scale,
            )
        )

    import videofentanyl as vf_mod

    previous = vf_mod._SERVER_OVERRIDE
    vf_mod._SERVER_OVERRIDE = _SERVER_URL
    started = time.time()
    try:
        results: list[dict[str, Any]] = []
        for idx, job in enumerate(jobs):
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            session = VideoSession(job=job, mode="ltx", verbose=_VERBOSE)
            ok = await session.run(idle_timeout=None)
            job.finished_at = time.time()
            job.status = JobStatus.DONE if ok else JobStatus.FAILED
            if not ok:
                raise RuntimeError(f"sequence failed at clip {idx + 1}: {job.error or 'generation failed'}")

            results.append(
                {
                    "index": idx + 1,
                    "prompt": job.params.prompt,
                    "output_path": str(job.output_path.resolve()),
                    "bytes": int(job.file_bytes),
                    "elapsed_s": round(job.elapsed, 3),
                    "ttff_ms": job.ttff_ms,
                    "generation_ms": job.gen_latency_ms,
                    "e2e_ms": job.e2e_latency_ms,
                }
            )

            if autocontinue and idx + 1 < len(jobs):
                nxt = jobs[idx + 1].params
                if method == CHAIN_METHOD_NATIVE_EXTEND:
                    nxt.generation_mode = "extend"
                    nxt.initial_image = None
                    nxt.source_video = load_media_payload(
                        str(job.output_path),
                        kind="video",
                    )
                    if nxt.extend_frames is None:
                        from videofentanyl import (
                            extend_latent_frames_for_video,
                            resolve_extend_latent_frames,
                        )

                        probed = extend_latent_frames_for_video(job.output_path)
                        if probed is not None:
                            nxt.extend_frames = probed
                        else:
                            nf = nxt.num_frames if nxt.num_frames is not None else jobs[idx].params.num_frames
                            nxt.extend_frames = resolve_extend_latent_frames(num_frames=nf)
                    if not nxt.extend_direction:
                        nxt.extend_direction = extend_direction or "after"
                    nxt.seed = int(time.time_ns() % (2**31 - 1)) or 1
                else:
                    next_frame = extract_last_frame(job.output_path)
                    if not next_frame:
                        raise RuntimeError(
                            f"autocontinue failed to extract last frame from clip {idx + 1}"
                        )
                    nxt.initial_image = next_frame
                    prev_mode = (job.params.generation_mode or "").strip().lower()
                    if prev_mode == "a2v":
                        nxt.a2v_visual_i2v_continue = True
                    nxt.seed = int(time.time_ns() % (2**31 - 1)) or 1
        if autoconcat:
            if method == CHAIN_METHOD_NATIVE_EXTEND:
                await asyncio.to_thread(
                    try_finalize_native_extend_chain,
                    jobs,
                    output_prefix,
                    "mp4",
                    _VERBOSE,
                )
            else:
                await asyncio.to_thread(
                    try_autoconcat_clips,
                    jobs,
                    output_prefix,
                    "mp4",
                    _VERBOSE,
                    False,
                )
            merged_candidates = sorted(_OUTPUT_DIR.glob(f"{output_prefix}_merged_*.mp4"))
            merged_path = str(merged_candidates[-1].resolve()) if merged_candidates else None
        else:
            merged_path = None
    finally:
        vf_mod._SERVER_OVERRIDE = previous

    return {
        "ok": True,
        "server": _SERVER_URL,
        "count": len(results),
        "autocontinue": bool(autocontinue),
        "chain_method": method,
        "autoconcat": bool(autoconcat),
        "merged_output_path": merged_path,
        "total_elapsed_s": round(time.time() - started, 3),
        "clips": results,
    }


@mcp.tool()
async def ltx_server_healthcheck() -> dict[str, Any]:
    """
    Verify that the configured ltx-ws endpoint accepts a WebSocket connection.
    """
    import websockets

    started = time.time()
    try:
        async with websockets.connect(_SERVER_URL, open_timeout=10, close_timeout=5):
            pass
    except Exception as exc:
        return {
            "ok": False,
            "server": _SERVER_URL,
            "error": str(exc),
        }
    return {
        "ok": True,
        "server": _SERVER_URL,
        "latency_ms": int((time.time() - started) * 1000),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mcp_server",
        description="MCP server exposing ltx-ws generation tools",
    )
    p.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="ltx-ws WebSocket URL")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="directory for generated videos")
    p.add_argument("--verbose", action="store_true", help="verbose WebSocket session logs")
    return p


def main() -> None:
    global _SERVER_URL, _OUTPUT_DIR, _VERBOSE
    args = _build_parser().parse_args()
    _SERVER_URL = args.server_url.strip()
    _OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()
    _VERBOSE = bool(args.verbose)

    # FastMCP handles stdio transport by default.
    mcp.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
