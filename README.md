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

The original single-backend scripts are still available:

```bash
curl -O https://raw.githubusercontent.com/lmangani/videofentanyl/main/legacy/fastvideo.py
curl -O https://raw.githubusercontent.com/lmangani/videofentanyl/main/legacy/dreamverse.py
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
| `--image PATH_OR_URL` | `-i` | — | Input image for image-to-video: local path or `http(s)` URL (downloaded via a temp file). |
| `--preset-id ID` | | mode default | Override the session preset ID. |
| `--preset-label STR` | | mode default | Override the preset label (dreamverse). |
| `--auto-extension` | | off | Enable server-side segment auto-extension. |
| `--loop` | | off | Enable loop generation. |

#### Queue & network

| Flag | Short | Default | Description |
|---|---|---|---|
| `--timeout SECS` | `-t` | `240` / `480` | Per-video timeout (fastvideo / dreamverse). |
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
| `--server URL` | Override the WebSocket endpoint. Use `ws://localhost:8765/ws` to route all generation through a local `videofentanylserver.py` instance. |

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
  Timeout  : 480s per video
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
  Timeout  : 240s per video
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

`videofentanylserver.py` runs a fully local WebSocket server that implements the
same protocol as `wss://1080p.fastvideo.org/ws`, using
[FastVideo](https://github.com/hao-ai-lab/FastVideo)'s **LTX2-Distilled** model
on **Apple Silicon (MPS)**.  No internet connection is needed for generation.

#### Install FastVideo (Apple MPS)

Upstream `FastVideo` declares a hard dependency on **`fastvideo-kernel`**, which
pulls **Triton**. Triton and the published `fastvideo-kernel` wheels target
**Linux x86_64 + CUDA**, not **macOS / Apple Silicon**. Resolver errors mentioning
`triton` and `macosx_*_arm64` are expected from a plain `uv pip install -e .` on a Mac.

The **MPS** code path in FastVideo uses **Torch SDPA** only and does not need
`fastvideo-kernel` for LTX2 inference in `videofentanylserver.py`. Use the small
patch script in this repo to gate that dependency to Linux x86_64 before installing.

**Option A — submodule (stays on latest upstream)**

```bash
# From the videofentanyl repo root
git submodule update --init --recursive

# One-time per submodule update: relax fastvideo-kernel for macOS
python scripts/patch_fastvideo_pyproject_for_apple_silicon.py

cd third_party/FastVideo
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e .
cd ../..

# Optional: install flash-attn (CUDA only; skip on Apple MPS)
# uv pip install flash-attn --no-build-isolation -v

# Install videofentanyl server/client deps (if not already global)
uv pip install websockets av Pillow huggingface_hub
```

**Option B — standalone clone** (same patch idea, any directory)

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
python /path/to/videofentanyl/scripts/patch_fastvideo_pyproject_for_apple_silicon.py \
    FastVideo/pyproject.toml
cd FastVideo
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e .
```

#### Download the server script

```bash
curl -O https://raw.githubusercontent.com/lmangani/videofentanyl/main/videofentanylserver.py
```

#### Start the server

```bash
# Default: downloads/caches the model in ~/.cache/huggingface/hub on first run
python videofentanylserver.py

# Download to a specific local folder (downloaded once, reused on subsequent starts)
python videofentanylserver.py --model-dir ./models/LTX2-Distilled-Diffusers

# Use a model folder that is already fully downloaded (no network access needed)
python videofentanylserver.py --model ./models/LTX2-Distilled-Diffusers

# Custom resolution / port
python videofentanylserver.py --port 9000 --height 720 --width 1280 --num-frames 65
```

**Model weights** (~9 GB) are resolved in this priority order:

1. **Local directory** (`--model ./path/to/model`) — loaded directly from disk, no network required.
2. **HF model ID + custom dir** (`--model-dir ./models/…`) — downloads to that folder on first run, then loads from it on subsequent starts.
3. **HF model ID only** (default) — downloaded and cached in `~/.cache/huggingface/hub` on first run.

In all cases the model is always loaded from a local path before being passed to FastVideo, which avoids a registry identification warning that occurs when passing a raw HuggingFace repo ID directly.

```bash
# One-time download to a custom folder, then use it offline
pip install huggingface_hub
python videofentanylserver.py --model-dir ./models/LTX2-Distilled-Diffusers
# On subsequent runs point directly at the folder:
python videofentanylserver.py --model ./models/LTX2-Distilled-Diffusers
```

#### Generate videos locally

Use the `--server` flag to point the client at your local server:

```bash
# Text-to-video
python videofentanyl.py --server ws://localhost:8765/ws \
    --prompt "a fox running through a snowy forest"

# Image-to-video
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
| `--height` | `480` | Output height in pixels. |
| `--width` | `848` | Output width in pixels. |
| `--fps` | `24` | Frames per second. |
| `--guidance-scale` | `1.0` | CFG scale (LTX2-Distilled uses 1.0). |
| `--attention-backend` | `TORCH_SDPA` | Attention backend: `TORCH_SDPA` (MPS/CPU) or `FLASH_ATTN` (CUDA). |
| `--chunk-size` | `65536` | Binary chunk size in bytes. |
| `--verbose` / `-v` | off | Per-connection protocol logging. |

---

### Protocol

#### FastVideo (1080p)
```
connect → session_init_v2    (handshake)
        → simple_generate    (trigger generation)
        ← gpu_assigned
        ← ltx2_segment_start / ltx2_segment_complete
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
