# <img width="488" height="58" alt="image" src="https://github.com/user-attachments/assets/7065ad1b-020e-42f2-862d-5c0c3654dd65" />

A unified command-line tool for [FastVideo](https://fastvideo.org/) — supporting both the **1080p** and **Dreamverse** backends in a single script.

> Reverse-engineered from the FastVideo web app's WebSocket protocol _(LTX-2 backend)_

### Requirements

- Python 3.10+
- `websockets`, `av`, `Pillow` libraries
- **ffmpeg** (optional): only needed on your machine if you use `--autoconcat` to merge autocontinue clips after generation

### Install

Grab the unified script and install dependencies:

```bash
curl -O https://raw.githubusercontent.com/lmangani/videofentanyl/main/videofentanyl.py
curl -O https://raw.githubusercontent.com/lmangani/videofentanyl/main/requirements.txt
pip install -r requirements.txt
```

---

### Usage

Select the backend with `--mode` (default: `fastvideo`).

#### FastVideo 1080p — short clips (~5–10s)

```bash
# Single video
python videofentanyl.py --prompt "a fox running through a snowy forest"

# 5 videos from the same prompt
python videofentanyl.py --prompt "sunset over the ocean" --count 5

# Multiple prompts, one video each
python videofentanyl.py --prompt "forest rain" --prompt "city lights" --prompt "volcano"

# Image-to-video (local file or https URL — URLs are downloaded to a temp file, then removed)
python videofentanyl.py --prompt "the scene comes alive" --image photo.jpg
python videofentanyl.py --prompt "the scene comes alive" --image https://example.com/still.png

# With --server, run scripts/fastvideo_install so local LTX2 image-to-video matches the start image
# (see "Local Server (Apple MPS)").

# AI prompt enhancement, custom output folder
python videofentanyl.py --prompt "girl walking in rain" --enhance --output-dir ./videos

# Seamless multi-clip continuation (last frame of each clip → first frame of next)
python videofentanyl.py --prompt "a river flowing through a canyon" --count 5 --autocontinue

# Same, then merge successful clips into one file with ffmpeg and delete the fragments
# (requires --autocontinue; if ffmpeg is not on PATH, fragments are left in place and a log is printed)
python videofentanyl.py --prompt "a river flowing through a canyon" --count 5 --autocontinue --autoconcat
```

#### Dreamverse — long-form videos (~30s, 6 segments)

```bash
# Single ~30s video (GPT expands prompt into 6 segments automatically)
python videofentanyl.py --mode dreamverse --prompt "a kid burps into a tunnel, with a huge echo"

# Queue of 3 videos
python videofentanyl.py --mode dreamverse --prompt "sunset over the ocean" --count 3

# Multiple prompts
python videofentanyl.py --mode dreamverse \
    --prompt "dog learns to skateboard" \
    --prompt "volcano at night"

# Skip GPT prompt expansion, use prompt as-is
python videofentanyl.py --mode dreamverse --prompt "detailed scene description..." --no-enhance
```

#### Quick examples

```bash
# Preview the queue without connecting (dry-run)
python videofentanyl.py --prompt "test" --count 5 --dry-run

# Verbose protocol output (useful for debugging)
python videofentanyl.py --mode dreamverse --prompt "test" --verbose

# Save to a folder with a custom prefix
python videofentanyl.py --prompt "test" --count 3 --output-dir ./videos --prefix clip

# Aggressive retry, multiple prompts x multiple videos each (6 total)
python videofentanyl.py \
    --prompt "coral reef" \
    --prompt "arctic tundra" \
    --count 3 \
    --retries 2
```

---

### Options

#### Mode

| Flag | Short | Default | Description |
|---|---|---|---|
| `--mode {fastvideo,dreamverse}` | `-m` | `fastvideo` | Generation backend. |

#### Generation

| Flag | Short | Default | Description |
|---|---|---|---|
| `--prompt TEXT` | `-p` | — | Prompt text. Repeat for multiple prompts. |
| `--count N` | `-n` | `1` | Videos to generate per prompt. |
| `--enhance` | `-e` | off (fastvideo) / on (dreamverse) | Enable AI prompt enhancement (GPT rewrite). |
| `--no-enhance` | | — | Disable GPT prompt expansion (dreamverse only). |
| `--image PATH_OR_URL` | `-i` | — | Input image for image-to-video: local path or `http(s)` URL (downloaded via a temp file). Hosted API: works out of the box. **Local server:** use FastVideo installed with `scripts/fastvideo_install` (LTX2 i2v patches). |
| `--preset-id ID` | | mode default | Override the session preset ID. |
| `--preset-label STR` | | mode default | Override the preset label (dreamverse). |
| `--auto-extension` | | off | Enable server-side segment auto-extension. |
| `--loop` | | off | Enable loop generation. |

#### Queue & network

| Flag | Short | Default | Description |
|---|---|---|---|
| `--idle-timeout SECS` | | `120` (hosted); **unlimited** with `--server` | If set, no application message for this long triggers a WebSocket ping; only a failed pong ends the session. With `--server`, omit this flag to wait indefinitely for the next server frame (keepalives + `generation_status` traffic still flow). |
| `--delay SECS` | `-d` | `1.0` / `2.0` | Pause between consecutive jobs. |
| `--retries N` | `-r` | `1` | Max attempts per job (exponential backoff). |

#### Output

| Flag | Short | Default | Description |
|---|---|---|---|
| `--output-dir DIR` | `-o` | `.` | Directory to save videos. |
| `--prefix STR` | | `video` / `dreamverse` | Filename prefix (per-mode default). |
| `--ext EXT` | | `mp4` | File extension. |

#### Misc

| Flag | Description |
|---|---|
| `--verbose` / `-v` | Print full WebSocket protocol trace. |
| `--dry-run` | Show the job queue and exit without connecting. |
| `--autocontinue` | Extract the last frame of each clip and feed it as the first frame of the next one. Ideal for seamless multi-clip runs in 1080p mode. |
| `--autoconcat` | After the queue finishes, merge **successful** autocontinue clips with **ffmpeg** (`-c copy`), then **delete** the fragment files. **Requires `--autocontinue`.** If `ffmpeg` is not on your PATH, the tool logs details and leaves all fragments unchanged. |
| `--server URL` | Override the WebSocket endpoint. Use `ws://localhost:8765/ws` to route all generation through a local `server.py` instance. |

---

### Output filenames

Files are named automatically:

```
{prefix}_{job_number}_{prompt_slug}_{timestamp}.{ext}
```

After **`--autoconcat`** succeeds, fragments are removed and a single merged file is written:

```
{prefix}_merged_{timestamp}.{ext}
```

Examples:
- `video_001_sunset_over_the_ocean_20260401_183000.mp4`
- `dreamverse_001_a_dog_learns_to_fly_20260401_191248.mp4`
- `video_merged_20260402_120000.mp4` (merged run)

---

### Full example (Dreamverse)

```
% python videofentanyl.py --mode dreamverse --prompt "a dog learns to fly" --verbose

════════════════════════════════════════════════════════════
  Dreamverse Queue — 1 job(s)
  Endpoint : wss://dreamverse.fastvideo.org/ws
  Wall limit : none (until idle fails or server closes)  |  idle 120s max silence between messages
  Delay    : 2.0s between jobs
════════════════════════════════════════════════════════════

  ┌─ Job 01/1  dreamverse_001_a_dog_learns_to_fly_20260401_191248.mp4
  │  prompt: 'a dog learns to fly'
  │  enhance: ON
  └──────────────────────────────────────────────────
    [  0.61s] connected ✓
    [  0.61s] → session_init_v2  prompt='a dog learns to fly'
    ...
    [ 33.06s] saved → dreamverse_001_a_dog_learns_to_fly_20260401_191248.mp4  (37140.8 KB)
  ✓ saved  dreamverse_001_a_dog_learns_to_fly_20260401_191248.mp4  (37141 KB, 6 segments, 33.1s)
```

### Full example (FastVideo 1080p)

```
% python videofentanyl.py --prompt "a mouse running through a dark pipe"

════════════════════════════════════════════════════════════
  FastVideo Queue — 1 job(s)
  Endpoint : wss://1080p.fastvideo.org/ws
  Wall limit : none (until idle fails or server closes)  |  idle 120s max silence between messages
  Delay    : 1.0s between jobs
════════════════════════════════════════════════════════════

  ┌─ Job 01/1  video_001_a_mouse_running_through_a_dark_pipe_20260401_184637.mp4
  │  prompt: 'a mouse running through a dark pipe'
  └──────────────────────────────────────────────────
    [  0.65s] connected ✓
    [  0.65s] → simple_generate  prompt='a mouse running through a dark pipe'
    ...
    [  6.64s] saved → video_001_a_mouse_running_through_a_dark_pipe_20260401_184637.mp4  (3892.7 KB)
  ✓ saved  video_001_a_mouse_running_through_a_dark_pipe_20260401_184637.mp4  (3893 KB, 6.6s)
```

---

### Notes

**Sequential generation** — each video opens a fresh WebSocket connection. The server enforces one active session per IP, so jobs run one at a time.

**Output format** — files are raw MediaSource chunks (fMP4/WebM). Most players handle them fine. If playback fails, remux with ffmpeg:

```bash
ffmpeg -i input.mp4 -c copy fixed.mp4
```

**`--autoconcat`** — merged output is named `{prefix}_merged_{timestamp}.{ext}` next to the clips. Concat uses stream copy; if ffmpeg cannot merge your fragments, stderr is printed and the individual files are kept.

**IP session limit** — if you see `ip_session_limit`, another session is active on your IP (e.g. a browser tab). The client detects this and retries automatically after 15 seconds.

**Generation time**
- *FastVideo 1080p*: typically 5–10 seconds per clip once a GPU is assigned.
- *Dreamverse*: typically ~30 seconds per video (6 segments × ~5s each).

---

### Local Server (Apple MPS)

`server.py` runs a fully local WebSocket server that implements the
same protocol as `wss://1080p.fastvideo.org/ws`, using
[FastVideo](https://github.com/hao-ai-lab/FastVideo)'s **LTX2-Distilled** model
on **Apple Silicon (MPS)**.  No internet connection is needed for generation.

#### Install FastVideo (Apple MPS)

Upstream `FastVideo` declares a hard dependency on **`fastvideo-kernel`**, which
pulls **Triton**. Triton and the published `fastvideo-kernel` wheels target
**Linux x86_64 + CUDA**, not **macOS / Apple Silicon**. Resolver errors mentioning
`triton` and `macosx_*_arm64` are expected from a plain `uv pip install -e .` on a Mac.

The **MPS** code path in FastVideo uses **Torch SDPA** only and does not need
`fastvideo-kernel` for LTX2 inference in `server.py`. Use
`scripts/fastvideo_install` so the submodule, **pyproject** workaround (gate
`fastvideo-kernel` to Linux x86_64), **patches FastVideo sources** for Apple Silicon:
LTX2 denoising uses **`torch.autocast` on the active device** (MPS/CUDA, not
hard-coded `cuda`), optional **`FV_PROGRESS_JSON`** lines for local-server
keepalives, **decoded frames on CPU** for multiprocessing IPC, and **worker
`pipe.send` sanitization** so MPS tensors never hit `_share_filename_: only
available on CPU`. **Image-to-video (`--image`):** the pipeline passes the VAE into
latent preparation, encodes the start image into the first latent temporal slice, and
during denoising **re-applies that slice each step** while setting **per-token timestep
0** on first-frame tokens—matching the FastVideo **`LTX2TrainingPipeline`** first-frame
conditioning (same idea as official LTX-2 keyframe conditioning: the first frame is
held as conditioning, not treated as fully noised). Workers prepend
**`VIDEOFENTANYL_FASTVIDEO_SRC`** so spawn processes load the patched FastVideo tree from this repo.
The script then runs **editable install**.

After `git pull` inside your FastVideo clone, re-run
`python scripts/fastvideo_install --no-install` (from this repo) to re-apply patches.

```bash
# From the videofentanyl repo root (recommended: uses third_party/FastVideo submodule)
uv venv --python 3.12 --seed && source .venv/bin/activate   # or your own env
python scripts/fastvideo_install

# Optional: install flash-attn (CUDA only; skip on Apple MPS)
# uv pip install flash-attn --no-build-isolation -v

# Server / client extras for this repo
uv pip install websockets av Pillow huggingface_hub
```

**Standalone FastVideo clone** (same workaround, your directory):

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git ~/src/FastVideo
python scripts/fastvideo_install --no-submodule --path ~/src/FastVideo
```

**Script flags** (for CI or split steps): `--no-submodule`, `--no-install`,
`--no-patch`, `--path DIR`. Run `python scripts/fastvideo_install --help`.

#### Download the server script

```bash
curl -O https://raw.githubusercontent.com/lmangani/videofentanyl/main/server.py
```

The local WebSocket entrypoint was formerly `videofentanylserver.py`; old bookmarks or scripts can be updated to `server.py`.

#### Start the server

```bash
# Default: downloads/caches the model in ~/.cache/huggingface/hub on first run
python server.py

# Download to a specific local folder (downloaded once, reused on subsequent starts)
python server.py --model-dir ./models/LTX2-Distilled-Diffusers

# Use a model folder that is already fully downloaded (no network access needed)
python server.py --model ./models/LTX2-Distilled-Diffusers

# Custom resolution / port
python server.py --port 9000 --height 720 --width 1280 --num-frames 65
```

**Model weights** (~9 GB) are resolved in this priority order:

1. **Local directory** (`--model ./path/to/model`) — loaded directly from disk, no network required.
2. **HF model ID + custom dir** (`--model-dir ./models/…`) — downloads to that folder on first run, then loads from it on subsequent starts.
3. **HF model ID only** (default) — downloaded and cached in `~/.cache/huggingface/hub` on first run.

In all cases the model is always loaded from a local path before being passed to FastVideo, which avoids a registry identification warning that occurs when passing a raw HuggingFace repo ID directly.

```bash
# One-time download to a custom folder, then use it offline
pip install huggingface_hub
python server.py --model-dir ./models/LTX2-Distilled-Diffusers
# On subsequent runs point directly at the folder:
python server.py --model ./models/LTX2-Distilled-Diffusers
```

#### Generate videos locally

Use the `--server` flag to point **`videofentanyl.py`** at your local server.

While the model runs, `server.py` emits **`generation_keepalive`** JSON
about every 15 seconds (with optional **`model_progress`**: denoise step, total steps,
rolling average seconds per step, and ETA parsed from FastVideo worker logs), accepts
**`generation_status`** from the client, and replies with **`generation_status_ack`**
including the same **`model_progress`** when available. The client has **no wall-clock session limit**; with
`--server` the default **idle** limit is **unlimited** (use `--idle-timeout` only if you
want a finite silence cap). On the hosted API the default idle is **120 s** (then a
WebSocket ping probe). If the client disconnects after the MP4 is ready, the server
copies the file into **`fvserver_completed/`** (override with `--spill-dir`).

```bash
# Text-to-video
python videofentanyl.py --server ws://localhost:8765/ws \
    --prompt "a fox running through a snowy forest"

# Image-to-video (first frame matches the start image when FastVideo was installed with
# scripts/fastvideo_install — see "Install FastVideo (Apple MPS)" above)
python videofentanyl.py --server ws://localhost:8765/ws \
    --prompt "the scene comes alive" --image photo.jpg

# Multiple prompts, 3 videos each, with autocontinue
python videofentanyl.py --server ws://localhost:8765/ws \
    --prompt "coral reef" --prompt "arctic tundra" \
    --count 3 --autocontinue --autoconcat
```

#### Server options

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address. |
| `--port` | `8765` | Port. |
| `--model` | `FastVideo/LTX2-Distilled-Diffusers` | HuggingFace model ID **or** path to a local model directory. When a local path is given it is loaded directly (no network). |
| `--model-dir` | _(HF cache)_ | Directory to download/cache the model into when `--model` is a HuggingFace ID. Ignored when `--model` is already a local path. |
| `--num-gpus` | `1` | Device count. |
| `--num-frames` | `97` | Frames to generate (`(8k+1)` required by LTX2: 9, 17, 25 … 97). |
| `--height` | `480` | Output height in pixels (snapped to a multiple of 32 for LTX2). |
| `--width` | `832` | Output width in pixels (snapped to a multiple of 32 for LTX2). |
| `--fps` | `24` | Frames per second. |
| `--guidance-scale` | `1.0` | CFG scale (LTX2-Distilled uses 1.0). |
| `--infer-steps` | `12` | Denoising steps (distilled LTX2 is tuned for few steps; raise e.g. to `50` for quality on fast GPUs). |
| `--stage-verification` | off | Enable FastVideo per-stage checks (slower). |
| `--torch-compile` | off | Enable `torch.compile` in FastVideo (experimental on MPS; needs a recent PyTorch). |
| `--attention-backend` | `TORCH_SDPA` | Attention backend: `TORCH_SDPA` (MPS/CPU) or `FLASH_ATTN` (CUDA). |
| `--chunk-size` | `65536` | Binary chunk size in bytes. |
| `--spill-dir` | `fvserver_completed` | If the client disconnects after the MP4 is encoded (or during download), copy the finished file here. On encode failure, the server also tries to salvage `*_ENCODE_FAIL*.mp4` from the temp workdir. |
| `--mac-ipc-safe-offload` | off | **Rare escape hatch:** enables DiT/VAE/text-encoder CPU offload so more tensors are CPU at FastVideo worker pipe boundaries. **Very slow** on MPS; only if you hit errors like `_share_filename_: only available on CPU`. Default keeps offload off for normal throughput. |
| `--verbose` / `-v` | off | Per-connection protocol logging. |

On **Apple MPS**, use a current **PyTorch** build (2.5+ recommended) with Metal support; the server logs the active version at startup and applies light runtime tuning (`float32_matmul_precision`, bf16 accumulation where supported). FastVideo still runs inference in a **multiprocess** worker, so wall time will not match a minimal single-process diffusers script — lowering `--infer-steps` is the main quality/speed trade-off for distilled weights.

---

### Protocol

#### FastVideo (1080p)
```
connect → session_init_v2    (handshake)
        → simple_generate    (trigger generation)
        → generation_status  (optional client poll while waiting)
        ← gpu_assigned
        ← ltx2_segment_start / ltx2_segment_complete
        ← generation_keepalive / generation_status_ack  (local server)
        ← [binary chunks]    (raw video data)
        ← ltx2_stream_complete
```

#### Dreamverse
```
connect → session_init_v2            (prompt in initial_rollout_prompt)
        ← queue_status
        ← gpu_assigned
        ← rewrite_seed_prompts_started
        ← seed_prompts_updated        (6 GPT-expanded segment prompts)
        ← rewrite_seed_prompts_complete
        ← ltx2_stream_start
        ← ltx2_segment_start … ltx2_segment_complete  (×6)
        ← [binary chunks]
        ← ltx2_stream_complete
```

---

> See [README_CMD.md](README_CMD.md) for the original per-script documentation.
