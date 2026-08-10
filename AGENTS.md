# AGENTS.md

Canonical guide for AI agents using **ltx-ws** to generate video for directors, editors, and creative workflows.

**Read [`DIRECTOR.md`](DIRECTOR.md) first** when the user provides a prompt or asks for video—you act as **Assistant Director** (interview, gold prompts, shot plan, then generate).

Read **this file** for MCP tools, MLX weights, and pipeline mechanics before calling generation APIs.

## What this stack does

| Piece | Role |
|-------|------|
| `server.py` | Local LTX-2.3 inference (MLX on Apple Silicon). One job at a time; streams MP4 over WebSocket. Embeds Web UI by default. |
| `videofentanyl.py` | CLI + protocol client (`--server ws://…/ws`). Implements session handshake, generation, and **autocontinue** frame chaining. |
| `web_ui.py` + `web/` | **LTX-WS Videofentanyl** browser UI: library, multi-clip autocontinue/autoconcat, LoRA picker, SSE progress. |
| `mcp_server.py` | MCP adapter exposing the same capabilities as tools for Cursor, Claude, etc. |

**Default endpoint:** `ws://127.0.0.1:8765/ws`  
**Default Web UI:** `http://127.0.0.1:8765/` (same host/port; use machine IP when remote)  
**Default MCP outputs:** `./mcp_outputs/` (override with `mcp_server.py --output-dir`)  
**Web UI outputs:** `./web_outputs/` (override with `server.py --web-output-dir`)

This stack does **not** run cloud GPT prompt expansion. Prompts you send are what the model sees (`enhancement_enabled=false` in MCP). **Transform user ideas via [`DIRECTOR.md`](DIRECTOR.md)** before submission—especially for chained sequences.

---

## MLX model weights only (mandatory)

**Only ever use MLX weights built for [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx).**  
`server.py` loads checkpoints through **`ltx_pipelines_mlx`** on Apple Silicon (Metal / MLX). It **cannot** run standard upstream LTX 2.3 PyTorch checkpoints.

### Allowed (use these)

| Source | Examples |
|--------|----------|
| Hugging Face MLX repos (`dgrauet`) | `dgrauet/ltx-2.3-mlx`, `dgrauet/ltx-2.3-mlx-q8`, `dgrauet/ltx-2.3-mlx-q4` |
| Local MLX snapshot dirs | `./models/dgrauet__ltx-2.3-mlx-q8/`, paths from `snapshot_download` of the repos above |
| Server default | `--model auto` (RAM → one of the **dgrauet** MLX variants) |

Weights are **MLX-converted** (via [mlx-forge](https://github.com/dgrauet/mlx-forge) / the ltx-2-mlx ecosystem). Repo ids usually include **`mlx`** in the name (e.g. `ltx-2.3-mlx-q8`).

### Forbidden (never suggest or configure)

| Do **not** use | Why |
|----------------|-----|
| `Lightricks/LTX-2.3` | Standard upstream weights — **not** MLX layout for `ltx_pipelines_mlx` |
| `Lightricks/LTX-2` or other non-MLX LTX Hub repos | Wrong runtime / tensor layout |
| Raw ComfyUI / diffusers / CUDA LTX checkpoints | Not compatible with this WebSocket server |
| Any `--model` id **without** a matching MLX conversion for ltx-2-mlx | Load or generation will fail |

If a user asks for “standard LTX 2.3”, “Lightricks weights”, or “the normal Hugging Face model”, **do not** point them at `Lightricks/LTX-2.3`. Tell them this project **only** supports **MLX** weights from the **dgrauet/ltx-2.3-mlx** family (or `auto`). On Apple Silicon with `ltx-ws`, MLX is the only valid path.

**Agents:** never recommend downloading or passing `--model Lightricks/LTX-2.3` (or similar non-MLX repos) to `server.py`.

---

## Primary job: serve directors with longer, continuous video

Directors usually want **one coherent piece** (15–30s+), not a single ~4s clip.

**Default agent strategy:**

1. **Do not** default to `ltx_generate_video` for narrative, reel, ad, or “make me a video” requests.
2. **Use** `ltx_generate_sequence` with:
   - `autocontinue: true` — clip N’s **last frame** becomes clip N+1’s start image (visual continuity).
   - `autoconcat: true` — merge successful clips into one deliverable MP4 (PyAV via `pip install av`).
3. **Plan in ~5 second segments** using `num_frames: 121` (~5.0s at 24 fps).
4. **Write one prompt per segment** — establish in clip 1; continuation language in clips 2+.

### Duration math (24 fps nominal)

LTX requires frame counts of **`8k + 1`** (e.g. 49, 97, 121, 193). Server snaps invalid values.

| `num_frames` | Approx. duration | Typical use |
|--------------|------------------|-------------|
| 49 | ~2.0s | quick test, transition beat |
| 97 | ~4.0s | default server clip; short beat |
| **121** | **~5.0s** | **recommended segment length for director chains** |
| 193 | ~8.0s | longer single segment (more RAM) |

**Target length → segment count (at 121 frames):**

| Desired runtime | Segments (`prompts` length) |
|-----------------|-----------------------------|
| ~15s | 3 |
| ~20s | 4 |
| ~25s | 5 |
| ~30s | 6 |

Example: a 25s hero spot → **5 prompts**, `num_frames: 121`, `autocontinue: true`, `autoconcat: true`.

### CLI equivalent (for humans / debugging)

```bash
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "Establishing wide shot of canyon at sunrise" \
  --prompt "Camera drifts forward along the river" \
  --prompt "Low angle as raft enters frame" \
  --prompt "Close pass alongside splashing oars" \
  --prompt "Pull back revealing full canyon scale" \
  --num-frames 121 --autocontinue --autoconcat
```

MCP `ltx_generate_sequence` is the programmatic equivalent of that workflow.

### Web UI (LTX-WS Videofentanyl)

For directors who prefer a browser over MCP/CLI:

1. `python server.py` (Web UI on by default; build `web/` once: `cd web && npm run build`).
2. Open `http://<host>:8765/` — header shows **LTX-WS Videofentanyl**.
3. Set **Clips (× duration)** for segment count; ×N > 1 forces **autocontinue + autoconcat** (matches CLI `--count N --autocontinue --autoconcat`).
4. **LoRA** dropdown defaults to **OmniNFT RL** (`DEFAULT_LORA_URL`); auto-downloads via `/api/loras/ensure`. Choose **None** to disable LoRA for one job.
5. Library retains prior generations; merged autoconcat output appears as a single **MERGED** clip.

Embedded mode runs multi-clip chains **in-process** (same MLX generator as the server)—do not route Web UI autocontinue through a separate WebSocket loopback.

---

## MCP tools (full reference)

Start MCP:

```bash
python mcp_server.py --server-url ws://127.0.0.1:8765/ws
```

Always ensure `server.py` is running first.

### `ltx_server_healthcheck`

Verifies the WebSocket endpoint accepts a connection.

**When to call:** before any generation; after user reports failures.

**Returns:** `{ ok, server, latency_ms }` or `{ ok: false, error }`.

If `ok=false`, tell the user to start `python server.py` or fix `--server-url`.

---

### `ltx_generate_video`

**One clip, one prompt.** Use for isolated tests, thumbnails, or a single beat—not for director-length work.

| Parameter | Type | Notes |
|-----------|------|--------|
| `prompt` | string | **Required.** Model-ready visual description. |
| `mode` | string | `generate` (default), `a2v`, `retake`, `extend`, `ic_lora`, `keyframe`, `lipdub`, `face_swap`, `id_lora` |
| `image` | string? | Path or URL — image-to-video / keyframe start / ID-LoRA first frame |
| `end_image` | string? | Keyframe end frame |
| `audio` | string? | Required for `a2v` / `id_lora` (ID-LoRA: identity reference, not soundtrack) |
| `video` | string? | Required for `retake` / `extend` |
| `seed` | int? | Override seed (`-1` random on server) |
| `num_frames` | int? | Default from server (~97). Use **121** for ~5s. |
| `height`, `width` | int? | Multiples of 32 (server snaps) |
| `num_steps` | int? | Denoising steps (server default ~8 distilled) |
| `retake_start`, `retake_end` | int? | `retake` mode latent frame range |
| `extend_frames` | int? | `extend` mode |
| `extend_direction` | string? | `before` or `after` |
| `lora_specs` | list? | `[[path_or_url, scale], …]` |
| `video_conditioning` | list? | `[[video_path_or_url, scale], …]` for `ic_lora` |
| `output_filename` | string? | Under `mcp_outputs/` |

**Returns:** `{ ok, server, output_path, bytes, elapsed_s, ttff_ms, generation_ms, e2e_ms, … }`

---

### `ltx_generate_sequence` ⭐ (director default)

**Multiple clips, sequential jobs, optional visual chaining and merge.**

| Parameter | Type | Default | Notes |
|-----------|------|---------|--------|
| `prompts` | list[string] | **required** | One prompt per segment; order = timeline order |
| `autocontinue` | bool | **`true`** | **Keep true** for director continuity |
| `chain_method` | string | `autocontinue` | `autocontinue` (last frame → i2v) or **`native_extend`** (ltx-2-mlx `extend_from_video` on prior MP4 for clip 2+) |
| `autoconcat` | bool | `false` | Set **`true`** when delivering one merged file |
| `mode` | string | `generate` | `generate`, `a2v`, `retake`, `extend`, `ic_lora`, `keyframe`, `lipdub`, `face_swap`, `id_lora` |
| `image` | string? | — | Start image for **clip 1 only** (unless you override per-clip via separate calls) |
| `end_image` | string? | — | Keyframe mode: end frame |
| `enhance_prompt` | bool | `false` | Gemma prompt rewrite via ltx-2-mlx |
| `pipeline_profile` | string | `distilled` | `distilled`, `two_stage`, `hq`, `one_stage` |
| `cfg_scale`, `stg_scale`, `stage2_steps` | | — | Optional generate/HQ kwargs |
| `no_regen_audio`, `reference_strength` | | — | Retake/extend/lipdub |
| `audio`, `video` | string? | — | Mode-specific (see single-clip tool) |
| `seed`, `num_frames`, `height`, `width`, `num_steps` | | — | Applied to **every** clip in the sequence |
| `retake_*`, `extend_*` | | — | Mode-specific |
| `lora_specs`, `video_conditioning` | | — | Mode-specific |
| `output_prefix` | string | `ltx_mcp` | Clip files + `{prefix}_merged_*.mp4` when autoconcat |

**Autocontinue behavior (critical):**

- **`chain_method: autocontinue`** (default): after each successful clip, extract the **last full frame** and inject as `initial_image` on the next job.
- **`chain_method: native_extend`**: clip 1 is `generate`/i2v; clips 2+ run **`mode: extend`** with the prior clip MP4 as `source_video` (ltx-2-mlx `RetakePipeline.extend_from_video`, dev model + CFG; **same `num_steps` as clip 1**, cfg=3.0, stg=0.0 unless overridden). Each extend output **includes prior footage**; with `autoconcat` the **last extend** is promoted as merged (not ffmpeg-concat of overlapping segments). Incompatible with `audiocontinue`.
- Each chained segment gets a **new seed** on the client path.
- If chaining fails, the sequence aborts—do not silently continue.

**Autoconcat behavior:**

- After all clips succeed, runs PyAV concat (stream copy).
- Deletes fragment MP4s on success.
- Requires PyAV (`pip install av`); if missing, fragments remain and `merged_output_path` is null.

**Returns:**

```json
{
  "ok": true,
  "count": 5,
  "autocontinue": true,
  "autoconcat": true,
  "merged_output_path": "/path/to/ltx_mcp_merged_….mp4",
  "total_elapsed_s": …,
  "clips": [ { "index", "prompt", "output_path", "bytes", "elapsed_s", … } ]
}
```

Deliver **`merged_output_path`** to the director when `autoconcat=true`.

---

## Decision matrix: which tool?

| User intent | Tool | Settings |
|-------------|------|----------|
| “Make a short test” / one shot | `ltx_generate_video` | optional `num_frames: 49` or `97` |
| Story, ad, reel, scene, “longer video” | `ltx_generate_sequence` | `autocontinue: true`, `autoconcat: true`, `num_frames: 121` |
| Same, but browser / non-technical user | Web UI | ×N clips, LoRA dropdown, watch library + merged output |
| Storyboard with distinct shots (hard cuts) | `ltx_generate_sequence` | `autocontinue: false` (or separate prompts as separate projects) |
| Music video from one track | CLI `audiocontinue` or manual a2v sequence | MCP: `mode: a2v` + sequence (advanced) |
| Talking head from face photo + short voice sample | `ltx_generate_video` | `mode: id_lora` + `image` + `audio` (structured prompt) |
| Fix a section of existing footage | `ltx_generate_video` | `mode: retake` + `video` |
| Extend clip length | `ltx_generate_video` | `mode: extend` + `video` |
| Style from reference video + LoRA | `ltx_generate_video` | `mode: ic_lora` |

**Rule:** If the user is a **director** or asks for **continuity**, **camera motion through a scene**, or **>5s**, use **`ltx_generate_sequence`** with **`autocontinue: true`**.

---

## Prompt engineering for LTX-2.3

**Canonical prompt guide:** [`DIRECTOR.md`](DIRECTOR.md) — LTX-2.3 principles, rewrite workflow, autocontinue segment prompts, i2v/a2v, portrait, and anti-patterns.

The local server uses **LTX-2.3** via **ltx-2-mlx**. There is no automatic GPT rewrite in MCP—prompt quality directly affects output.

### Quick reference (see DIRECTOR.md for full detail)

- **Be visually specific** and **direct the scene** (blocking, left/right, foreground/background).
- **Describe motion with verbs**; avoid static photo captions—especially for i2v.
- **One dominant action per ~5s segment**; chain with establish + continue prompts (`autocontinue: true`).
- **Match aspect ratio** to delivery (native portrait: `height: 1024`, `width: 576`).
- **Describe audio** explicitly in a2v mode.

### Aspect ratio presets (height × width, snapped to 32px)

| Format | `height` | `width` | Notes |
|--------|----------|---------|--------|
| Landscape 16:9 | 576 | 1024 | common cinematic |
| Vertical 9:16 | 1024 | 576 | reels / stories |
| Square 1:1 | 768 | 768 | social loops |
| 4:5 vertical | 960 | 768 | feed posts |

When using a **start image**, match output orientation to the image.

### Chained sequences (`autocontinue: true`)

Full establish/continue rules, examples, and mistakes: **`DIRECTOR.md` § Extending prompts for longer videos**.

Structure prompts as a **timeline**, not duplicate full scene descriptions.

1. **Clip 1 — establish:** full scene, subject, lighting, camera starting position.
2. **Clips 2+ — continue:** short prompts that assume the same world; emphasize **what changes** (camera move, action beat, reveal).

**Good continuation phrases:**

- “Continue forward along the same street as storefront lights flicker on”
- “Continue the drone shot, banking right between towers”
- “Continue closer on the dancer’s footwork, same stage lighting”
- “Continue pulling back to reveal the full crowd”

**Avoid in continuation clips:**

- Re-introducing the entire scene from scratch (“A city at night with…” every time)
- Contradicting wardrobe, location, or time of day
- Sudden genre or lighting changes without a narrative reason

### Image-to-video + autocontinue

- Pass `image` on the **sequence** call → used for **clip 1** only.
- Clips 2+ inherit the **last frame** of the prior clip—do not re-describe a different location unless intentional.

### Audio-to-video (`mode: a2v`)

- Prompt should describe **visual performance** aligned to audio (performance, lip sync context, energy).
- Provide `audio` path or URL.
- For long music-driven pieces, prefer CLI `--audiocontinue` or split audio manually across sequence jobs.

---

## Generation modes (when directors need more than T2V)

| `mode` | Purpose | Required inputs |
|--------|---------|-----------------|
| `generate` | Text-to-video or image-to-video (`image`) | `prompt` |
| `a2v` | Video driven by audio track (audio = soundtrack / timing) | `prompt`, `audio` |
| `retake` | Replace a segment of source video | `prompt`, `video`, `retake_start`, `retake_end` |
| `extend` | Add frames before/after source | `prompt`, `video`, `extend_frames`, `extend_direction` |
| `ic_lora` | Reference-video conditioning + LoRA | `prompt`, `lora_specs`, `video_conditioning` |
| `lipdub` | Lip-sync / dub with IC-LoRA + reference video audio | `prompt`, `video`, exactly one LipDub `lora_specs` |
| `face_swap` | BFS V3 head swap (identity image + reference video) | `prompt`, `image` (face), `video`, exactly one head-swap `lora_specs` |
| `id_lora` | Identity transfer: face still + voice sample → new talking clip | `prompt`, `image`, `audio`; optional one ID-LoRA `lora_specs` (defaults to CelebV-HQ) |

**When to pick which talking-head path**

| Need | Mode | Notes |
|------|------|--------|
| Drive video from a full audio track (lips follow that WAV) | `a2v` | Audio is the deliverable soundtrack / timing driver |
| Swap a face onto existing performance footage | `face_swap` | Needs reference **video** + face image; keeps performance motion |
| Re-dub / lip-sync an existing clip with IC-LoRA | `lipdub` | Needs reference **video** (+ voice audio) |
| New clip from a face photo + short voice sample (identity), new speech from prompt | `id_lora` | Ref audio is **identity context** at negative positions — not remuxed as the soundtrack |

**ID-LoRA (`mode: id_lora`)** — two-stage MLX port of [ID-LoRA 2.3](https://github.com/ID-LoRA/ID-LoRA):

1. Stage 1: **dev** transformer + ID-LoRA, first-frame I2V, reference audio at negative temporal positions (`[ref ‖ target]`), video CFG≈3 / audio CFG≈7 + identity guidance.
2. Stage 2: drop ID-LoRA, 2× spatial upsample, fuse distilled LoRA, freeze stage-1 audio, short distilled refine.
3. Prompt structure: `[VISUAL]: … [SPEECH]: … [SOUNDS]: …`.
4. **Reference audio ≠ soundtrack.** The WAV is **voice identity only** (~5s mono tip); spoken words must be written in `[SPEECH]`. Empty/silent refs and 0-match LoRA adapters fail loudly.
5. **Balanced defaults (speed):** stage-1 steps **20**, `stg_scale=0`, `modality_scale=1` (≈2.5× fewer DiT forwards than the old 30×5 path). **Faithful preset:** `num_steps=30`, `stg_scale=1`, `modality_scale=3`.
6. **Guidance knobs:** `cfg_scale` / video CFG≈**3**, `audio_cfg_scale`≈**7**, `identity_guidance_scale`≈**3** (raise toward **4–6** if voice match is weak). `ref_audio_tokens` is derived from WAV length — not user-set.
7. **Preview escapes:** `skip_stage_2` (stage-1 resolution, no upsample/refine) or `upsample_only` (2× pixels, skip stage-2 denoise). Mutually exclusive.
8. CelebV-HQ LoRA default (`AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K`); alt TalkVid-3K. Requires **dev** MLX weights (`dgrauet/ltx-2.3-mlx` or q8). Output is ~2× stage-1 resolution unless `skip_stage_2` (stage-1 long side ≤512).

**Face swap (`mode: face_swap`)** — Comfy-aligned BFS V3 on MLX (`ltx_ltxv_add_guide` + `FaceSwapPipeline`):

1. Builds composite guide (green side strip + identity face every frame + performance video).
2. `LTXVPreprocess` (CRF 33) + `LTXVAddGuide` full-video append + `LTXVCropGuides`.
3. **Dev** DiT + Comfy V3 LoRA stack fused in one pass (never washed by Stage 2):
   - `ltx-2.3-22b-distilled-lora-dynamic_fro09…` @ 1.0 (auto from Kijai / local cache)
   - head-swap LoRA @ ~0.98
4. Distilled 8-step sigma table, CFG≈1.0 (Comfy V3); output cropped to main panel; reference audio muxed when present.

LoRA preset: `Alissonerdx/BFS-Best-Face-Swap-Video` → `head_swap_v3_rank_adaptive_fro_098.safetensors` @ 0.98.  
Prompt trigger: `head_swap:` with `FACE:` / `ACTION:` sections (identity from image, not face description in text).  
Requires **dev** MLX weights (`dgrauet/ltx-2.3-mlx` or q8). Skip distilled-dynamic with `LTX_WS_FACE_SWAP_NO_DISTILLED_LORA=1`.

For most **director narrative** work, stay on `mode: generate` with **autocontinue sequences**.

### LoRA (OmniNFT default)

- **Default artifact:** `https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors` (`DEFAULT_LORA_URL` in `server.py`).
- **Web UI:** dropdown applies per-request `lora_specs` (no `--enable-lora` required).
- **MCP / API:** `lora_specs: [["<url or path>", 1.0]]` on `ltx_generate_video` or `ltx_generate_sequence`.
- **Global server default:** `python server.py --enable-lora --lora <url> 1.0` or `LTX_WS_ENABLE_LORA=1` + `LTX_WS_DEFAULT_LORA`.

---

## Standard agent workflow

### 1. Preflight

```text
ltx_server_healthcheck → ok?
```

### 2. Clarify deliverable (briefly)

- Target length (e.g. “~20s spot”)
- Orientation (vertical vs landscape)
- Optional hero still (`image`) for clip 1

### 3. Plan segments

- `segments = ceil(target_seconds / 5)`
- `num_frames = 121` per segment
- Write `prompts` list (establish + continue…)

### 4. Generate

```json
{
  "tool": "ltx_generate_sequence",
  "arguments": {
    "prompts": [ "…", "…", "…" ],
    "mode": "generate",
    "autocontinue": true,
    "autoconcat": true,
    "num_frames": 121,
    "height": 576,
    "width": 1024,
    "num_steps": 8,
    "output_prefix": "director_cut"
  }
}
```

### 5. Return paths

- Primary: `merged_output_path` when autoconcat succeeded
- Fallback: list `clips[].output_path` if merge failed or autoconcat was false

---

## Reference example (neon city, ~15s)

Three ~5s segments, chained and merged:

```json
{
  "tool": "ltx_generate_sequence",
  "arguments": {
    "prompts": [
      "Cinematic drone shot descending toward a neon city at dusk, rain-slick streets, magenta and cyan signage, slow forward motion",
      "Continue forward between glass towers as signs illuminate sequentially, reflections on wet asphalt, same dusk lighting",
      "Continue into a low pass above traffic, headlight streaks and neon reflections, camera glides forward without cutting"
    ],
    "mode": "generate",
    "autocontinue": true,
    "autoconcat": true,
    "num_frames": 121,
    "height": 576,
    "width": 1024,
    "num_steps": 8,
    "output_prefix": "neon_city_15s"
  }
}
```

---

## Technical constraints agents should respect

- **Frames:** `num_frames` must be `8k+1`; server coerces if needed.
- **Resolution:** `height` and `width` multiples of **32**.
- **Queue:** one MLX generation at a time; sequences run **serially** (correct for autocontinue).
- **RAM:** longer clips / higher resolution need more unified memory; q8/q4 weights help on smaller Macs.
- **No GPT rewrite** on this MLX server—optimize prompts yourself.
- **ltx-2-mlx** version pinned in repo (see `requirements.txt` / `ltx_mlx_backend.py`); install matching MLX packages after pulls.
- **Weights:** **MLX only** — `dgrauet/ltx-2.3-mlx*` repos or local MLX snapshots; **never** `Lightricks/LTX-2.3` or other standard LTX checkpoints (see [MLX model weights only](#mlx-model-weights-only-mandatory)).

---

## Minimal local setup (for user troubleshooting)

```bash
# 1. Inference server (+ Web UI)
python server.py

# 2. MCP adapter (optional)
python mcp_server.py --server-url ws://127.0.0.1:8765/ws

# 3. Agent calls
#    ltx_server_healthcheck → ltx_generate_sequence (preferred) or ltx_generate_video
#    Or direct users to http://127.0.0.1:8765/ for LTX-WS Videofentanyl
```

---

## Summary for agents

1. **Directors → `ltx_generate_sequence`**, not single-clip by default.
2. **`autocontinue: true`** always for continuous camera / scene flow.
3. **`autoconcat: true`** when delivering one file.
4. **`num_frames: 121`** (~5s) per segment; plan segment count for target runtime.
5. **Prompt clip 1 to establish; clips 2+ to continue** with motion and detail, not full re-description.
6. **Match aspect ratio** to platform; use `image` only for clip 1 when doing I2V chains.
7. **Models:** **only MLX** weights for ltx-2-mlx (`dgrauet/ltx-2.3-mlx*`) — never standard `Lightricks/LTX-2.3` checkpoints.
8. **Web UI** for human directors; **MCP sequence** for programmatic agents—both support autocontinue/autoconcat and per-request LoRA.

`mcp_server.py` reuses `videofentanyl.py` session/protocol code—behavior matches CLI `--autocontinue` / `--autoconcat`.
