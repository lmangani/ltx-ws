---
name: videofentanyl
description: "Use this skill whenever the user wants to generate a video, create a video from a prompt, generate a short clip, generate a long video, use local LTX (MLX) or Dreamverse, batch generate videos, or save videos to disk. Trigger on phrases like 'generate a video', 'make a video', 'create a clip', 'generate videos from prompts', 'use ltx', 'use dreamverse', 'batch video generation', 'long video', 'short video clip', or any request to synthesize video from a text prompt or image."
---

# videofentanyl.py — Video Generation Skill

Single script for **local LTX** (WebSocket to `server.py` + [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) on Apple Silicon) and **Dreamverse** (hosted). Select the backend with `--mode` (default: `ltx`). `ltx` requires `--server`. No API key for local; Dreamverse follows hosted limits. One active session per IP where the host enforces it.

| Mode | Use for | Output | Typical time |
|---|---|---|---|
| `ltx` (default) | Short clips, local MLX | Single segment, resolution/steps dependent | machine-dependent |
| `dreamverse` | Long videos | ~30s, 6 segments, GPT-expanded prompt | 60–120s |

---

## Environment Setup

Resolve `VIDEOFENTANYL_DIR` — the directory containing the script.

```
$VIDEOFENTANYL_DIR/videofentanyl.py
$VIDEOFENTANYL_DIR/server.py
```

**How to resolve:**
1. Check if the user mentioned the path.
2. Try: `ls videofentanyl.py` (may be in cwd).
3. If unknown, ask once:
   > "Where is your videofentanyl.py directory?"

**Install client dependencies (once):**
```bash
pip install -r requirements.txt
# or individually:
pip install websockets av Pillow huggingface_hub
```

For **local `server.py`**, also install MLX packages (see `requirements.txt` comments and README *Local server (Apple Silicon / MLX)*).

---

## ltx mode — Short clips (default, requires `--server`)

One prompt → one WebSocket session → one video file saved to disk. Start `server.py` in another terminal first.

### Basic usage

```bash
# Single video (local server on default port)
python videofentanyl.py --server ws://localhost:8765/ws --prompt "a fox running through a snowy forest"

# Multiple videos from same prompt
python videofentanyl.py --server ws://localhost:8765/ws --prompt "sunset over the ocean" --count 5

# Multiple prompts, one video each
python videofentanyl.py --server ws://localhost:8765/ws \
  --prompt "forest rain timelapse" \
  --prompt "city lights at night"

# Multiple prompts × N videos each
python videofentanyl.py --server ws://localhost:8765/ws \
  --prompt "coral reef" \
  --prompt "arctic tundra" \
  --count 3

# Image-to-video
python videofentanyl.py --server ws://localhost:8765/ws \
  --prompt "the scene comes alive" --image ./photo.jpg

# With AI prompt enhancement (GPT rewrite), if you use it
python videofentanyl.py --server ws://localhost:8765/ws \
  --prompt "girl walking in rain" --enhance

# Custom output folder and prefix
python videofentanyl.py --server ws://localhost:8765/ws --prompt "test" --count 3 \
  --output-dir ./videos --prefix clip

# Seamless multi-clip continuation (last frame of each → first frame of next)
python videofentanyl.py --server ws://localhost:8765/ws \
  --prompt "ocean waves" --count 5 --autocontinue

# Preview queue without connecting
python videofentanyl.py --server ws://localhost:8765/ws --prompt "test" --count 5 --dry-run

# Full protocol trace
python videofentanyl.py --server ws://localhost:8765/ws --prompt "test" --verbose
```

---

## dreamverse mode — Long Videos (~30s)

One prompt → GPT expands into 6 segment descriptions → 6-segment ~30s video.
Enhancement is **on by default** for this mode.

### Basic usage

```bash
# Single long video
python videofentanyl.py --mode dreamverse --prompt "a kid burps into a tunnel, with a huge echo"

# Queue of videos
python videofentanyl.py --mode dreamverse --prompt "volcano eruption at night" --count 3

# Multiple prompts
python videofentanyl.py --mode dreamverse \
  --prompt "dog learns to skateboard" \
  --prompt "timelapse of a city waking up"

# Skip GPT expansion (use prompt as-is)
python videofentanyl.py --mode dreamverse --prompt "very detailed scene description..." --no-enhance

# Custom output, verbose
python videofentanyl.py --mode dreamverse --prompt "test" --output-dir ./videos --verbose
```

---

## All flags

| Flag | Short | Default | Description |
|---|---|---|---|
| `--mode {ltx,dreamverse}` | `-m` | `ltx` | Generation backend (`ltx` needs `--server`). |
| `--prompt TEXT` | `-p` | — | Prompt. Repeat for multiple. |
| `--count N` | `-n` | `1` | Videos per prompt. |
| `--enhance` | `-e` | off (ltx) / on (dreamverse) | GPT prompt rewrite before generation. |
| `--no-enhance` | | — | Disable GPT expansion (dreamverse only). |
| `--image PATH` | `-i` | — | Input image for image-to-video. |
| `--server URL` | | _(required for ltx)_ | WebSocket endpoint (e.g. `ws://localhost:8765/ws` with local `server.py`). |
| `--preset-id ID` | | mode default | Override session preset ID. |
| `--preset-label STR` | | mode default | Override preset label (dreamverse). |
| `--auto-extension` | | off | Server-side segment auto-extension. |
| `--loop` | | off | Loop generation. |
| `--output-dir DIR` | `-o` | `.` | Save directory. |
| `--prefix STR` | | `ltx` / `dreamverse` | Filename prefix (per-mode default). |
| `--ext EXT` | | `mp4` | File extension. |
| `--idle-timeout SECS` | | `120` (dreamverse); **unlimited** with `--server` | If set, no application message this long → WebSocket ping probe. With `--server`, omit for unlimited recv wait (keepalives + optional `generation_status` traffic). |
| `--delay SECS` | `-d` | `1.0` (ltx) / `2.0` (dreamverse) | Pause between jobs. |
| `--retries N` | `-r` | `1` | Max attempts (exponential backoff). |
| `--verbose` | `-v` | off | Full WebSocket protocol trace. |
| `--dry-run` | | off | Show queue, don't connect. |

---

## Local server (`server.py`)

Fully local generation uses **`server.py`** with **ltx-2-mlx** on Apple Silicon. Install MLX packages from the repo README / `requirements.txt`, run **`python server.py`**, then point the client at **`--server ws://localhost:8765/ws`**.

---

## Output Files

Each successful job saves one video file:

```
ltx_001_a_fox_running_20260401_183000.mp4
dreamverse_001_a_dog_learns_to_fly_20260401_191248.mp4
```

Filename pattern: `{prefix}_{N:03d}_{prompt_slug}_{timestamp}.{ext}`

---

## Protocol Reference

### ltx mode flow (local `server.py`)

```
connect  →  session_init_v2    (preset: simple_custom_prompt, single_clip_mode: true)
         →  simple_generate    (prompt sent here)
         ←  gpu_assigned
         ←  ltx2_segment_start
         ←  [binary fMP4 chunks]
         ←  ltx2_stream_complete
```

### dreamverse mode flow

```
connect  →  session_init_v2    (preset: custom_editable, single_clip_mode: false,
                                initial_rollout_prompt: "<prompt>")
         ←  gpu_assigned
         ←  rewrite_seed_prompts_started   (GPT expanding prompt, ~6s)
         ←  seed_prompts_updated           (6 segment prompts)
         ←  ltx2_stream_start              (total_segments: 6)
         ←  ltx2_segment_start × 6
         ←  [binary fMP4 chunks]
         ←  ltx2_stream_complete
```

Key difference: dreamverse sends **no** `simple_generate` — the server starts
automatically after the GPT rewrite completes.

---

## Common Patterns

### Generate N videos from a list of prompts
```bash
python videofentanyl.py --server ws://localhost:8765/ws \
  --prompt "prompt one" \
  --prompt "prompt two" \
  --prompt "prompt three" \
  --count 2 \
  --output-dir ./out
```
Produces 6 videos total (3 prompts × 2 each), sequentially.

### Generate a long video with verbose logging
```bash
python videofentanyl.py --mode dreamverse \
  --prompt "a scientist discovers something extraordinary" \
  --verbose \
  --output-dir ./out
```

### Retry flaky jobs automatically
```bash
python videofentanyl.py --server ws://localhost:8765/ws \
  --prompt "test" --count 10 --retries 3 --delay 2
```

### Check what would be generated without connecting
```bash
python videofentanyl.py --mode dreamverse --prompt "test" --count 5 --dry-run
```

---

## Troubleshooting

**`ip_session_limit` error** — another session is active on your IP (e.g. a
browser tab on `dreamverse.fastvideo.org`). The client
detects this and retries automatically after 15 seconds. Close the browser tab
to resolve immediately.

**Video won't play** — output is raw fMP4/WebM MediaSource chunks. Remux:
```bash
ffmpeg -i input.mp4 -c copy fixed.mp4
```

**Stall / hung session** — there is no client wall-clock limit. Dreamverse default
`--idle-timeout` is 120 s of silence (then ping). With `--server`, idle defaults to
unlimited unless you set `--idle-timeout` yourself.

**Queue gets stuck on retry** — check `--retries` is > 1. Default is 1 attempt
(no retry). Use `--retries 3` for resilience.
