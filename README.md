# <img width="488" height="58" alt="image" src="https://github.com/user-attachments/assets/7065ad1b-020e-42f2-862d-5c0c3654dd65" />

A unified command-line tool for **local LTX-2.3 (MLX)** and **Dreamverse** (hosted) in a single script.

> Default `ltx` mode targets `server.py` on Apple Silicon via [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx). Dreamverse still uses the FastVideo-hosted WebSocket API.

### Requirements

- Python 3.10+ for the client; **3.11+** recommended for `server.py` + ltx-2-mlx
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

Select the backend with `--mode` (default: `ltx`; **`ltx` requires `--server`**).

#### LTX local (MLX) — short clips (duration depends on model, resolution, and steps)

```bash
# Single video (local server must be running)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
    --prompt "a fox running through a snowy forest"

# 5 videos from the same prompt
python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
    --prompt "sunset over the ocean" --count 5

# Multiple prompts, one video each
python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
    --prompt "forest rain" --prompt "city lights" --prompt "volcano"

# Image-to-video (local file or https URL — URLs are downloaded to a temp file, then removed)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
    --prompt "the scene comes alive" --image photo.jpg
python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
    --prompt "the scene comes alive" --image https://example.com/still.png

# AI prompt enhancement, custom output folder
python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
    --prompt "girl walking in rain" --enhance --output-dir ./videos

# Seamless multi-clip continuation (last frame of each clip → first frame of next)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
    --prompt "a river flowing through a canyon" --count 5 --autocontinue

# Same, then merge successful clips into one file with ffmpeg and delete the fragments
# (requires --autocontinue; if ffmpeg is not on PATH, fragments are left in place and a log is printed)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
    --prompt "a river flowing through a canyon" --count 5 --autocontinue --autoconcat
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
python videofentanyl.py --server ws://127.0.0.1:8765/ws --prompt "test" --count 5 --dry-run

# Verbose protocol output (useful for debugging)
python videofentanyl.py --mode dreamverse --prompt "test" --verbose

# Save to a folder with a custom prefix
python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
    --prompt "test" --count 3 --output-dir ./videos --prefix clip

# Aggressive retry, multiple prompts x multiple videos each (6 total)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
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
| `--mode {ltx,dreamverse}` | `-m` | `ltx` | Generation backend (`ltx` needs `--server`). |

#### Generation

| Flag | Short | Default | Description |
|---|---|---|---|
| `--prompt TEXT` | `-p` | — | Prompt text. Repeat for multiple prompts. |
| `--count N` | `-n` | `1` | Videos to generate per prompt. |
| `--enhance` | `-e` | off (ltx) / on (dreamverse) | Enable AI prompt enhancement (GPT rewrite). |
| `--no-enhance` | | — | Disable GPT prompt expansion (dreamverse only). |
| `--image PATH_OR_URL` | `-i` | — | Input image for image-to-video: local path or `http(s)` URL (downloaded via a temp file). **Local server:** `server.py` + ltx-2-mlx. |
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
| `--prefix STR` | | `ltx` / `dreamverse` | Filename prefix (per-mode default). |
| `--ext EXT` | | `mp4` | File extension. |

#### Misc

| Flag | Description |
|---|---|
| `--verbose` / `-v` | Print full WebSocket protocol trace. |
| `--dry-run` | Show the job queue and exit without connecting. |
| `--autocontinue` | Extract the last frame of each clip and feed it as the first frame of the next one. Ideal for seamless multi-clip runs when chaining short clips (e.g. local LTX). |
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

### Full example (LTX local)

```
% python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
    --prompt "a mouse running through a dark pipe"

════════════════════════════════════════════════════════════
  LTX (MLX local) Queue — 1 job(s)
  Endpoint : ws://127.0.0.1:8765/ws
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
- *LTX local (MLX)*: depends on model variant, resolution, and steps; use ``scripts/benchmark_local_generation.py`` on your machine.
- *Dreamverse*: typically ~30 seconds per video (6 segments × ~5s each).

---

### Local server (Apple Silicon / MLX)

[`server.py`](server.py) keeps the **same WebSocket JSON + MP4 binary** protocol used with ``videofentanyl.py --server``, but runs **[ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)** (pure MLX on Metal).

#### Install

Python **3.11+** recommended. From the repo root:

```bash
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install \
  "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git#subdirectory=packages/ltx-core-mlx" \
  "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git#subdirectory=packages/ltx-pipelines-mlx"
```

See the upstream repo for **q4 / q8 / bf16** weights and RAM notes.

**Weights download** — when ``--model`` is a Hugging Face repo id (default [dgrauet/ltx-2.3-mlx](https://huggingface.co/dgrauet/ltx-2.3-mlx)), the server runs ``huggingface_hub.snapshot_download`` on startup (same file set as ``huggingface-cli download <repo>``). Weights go under ``./models/<org>__<name>/`` unless you set ``--model-dir`` or ``$VIDEOFENTANYL_MODELS``. The default bf16 repo is large (~90GB+ class); use ``--model dgrauet/ltx-2.3-mlx-q8`` or ``-q4`` if you need a smaller footprint.

#### Start the server

```bash
python server.py
python server.py --model dgrauet/ltx-2.3-mlx-q4 --model-dir ./models/ltx-q4
python server.py --port 9000 --height 480 --width 704 --num-frames 65
```

#### Client

```bash
python videofentanyl.py --server ws://127.0.0.1:8765/ws --prompt "a fox in snow"
python videofentanyl.py --server ws://127.0.0.1:8765/ws --prompt "scene" --image photo.jpg
```

Keepalives and ``generation_status`` / ``generation_status_ack`` behave as before; ``model_progress`` is reserved until wired from MLX. Use ``--spill-dir`` if clients may disconnect mid-download.

#### ``server.py`` flags (summary)

| Flag | Default | Description |
|------|---------|-------------|
| ``--host`` | ``0.0.0.0`` | Bind address |
| ``--port`` | ``8765`` | TCP port |
| ``--model`` | ``dgrauet/ltx-2.3-mlx`` | HF repo id or local weights directory (bf16; large). |
| ``--model-dir`` | — | Override download dir for HF ids; default is ``./models/<org>__<name>/`` (or ``$VIDEOFENTANYL_MODELS``). Missing files are pulled with ``huggingface_hub.snapshot_download`` (same as ``huggingface-cli download``). |
| ``--num-frames`` | ``97`` | ``(8k+1)`` frame counts |
| ``--height`` / ``--width`` | ``480`` / ``704`` | Multiples of 32 |
| ``--infer-steps`` | ``8`` | Distilled one-stage steps |
| ``--mlx-low-memory`` | off | Pass ``low_memory=True`` into ltx-2-mlx |
| ``--chunk-size`` | ``65536`` | WS binary chunk size |
| ``--spill-dir`` | ``fvserver_completed`` | Salvage path on disconnect |
| ``--verbose`` | off | Per-connection logs |


### Protocol

#### LTX local (WebSocket)
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
