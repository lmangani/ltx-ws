# fastvideo.py

A command-line queue manager for [FastVideo](https://1080p.fastvideo.org/) 

> Reverse-engineered from the FastVideo web app's WebSocket protocol (`src/App.svelte`, LTX-2 backend).


### Requirements

- Python 3.10+
- `websockets` library (auto-installed on first run)

### Install

Just grab the latest script directly:

```bash
curl -O https://raw.githubusercontent.com/lmangani/fastervideo.py/main/fastvideo.py
pip install websockets
```

### Usage

```
python fastvideo.py --prompt "your prompt here" [options]
```

#### Quick examples

```bash
# Single video
python fastvideo.py --prompt "a fox running through a snowy forest"

# 5 videos from the same prompt
python fastvideo.py --prompt "sunset over the ocean" --count 5

# Multiple prompts, one video each
python fastvideo.py \
  --prompt "forest rain timelapse" \
  --prompt "city lights at night" \
  --prompt "volcano eruption aerial"

# Multiple prompts × multiple videos each (6 total, sequential)
python fastvideo.py \
  --prompt "coral reef" \
  --prompt "arctic tundra" \
  --count 3

# Image-to-video
python fastvideo.py --prompt "the scene comes alive" --image photo.jpg

# AI prompt enhancement (GPT rewrite before generation)
python fastvideo.py --prompt "girl walking in rain" --enhance

# Save to a folder with a custom prefix
python fastvideo.py --prompt "test" --count 3 --output-dir ./videos --prefix clip

# Preview the queue without connecting
python fastvideo.py --prompt "test" --count 5 --dry-run

# Verbose protocol output (useful for debugging)
python fastvideo.py --prompt "test" --verbose
```

#### Full Example

```bash
% python fastvideo.py --prompt "a mouse running through a dark pipe" --count 1

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
    [  0.83s] ← gpu_assigned ✓
    [  0.83s] ← stream started — receiving frames…
    [  0.95s] ← stream started — receiving frames…
    [  0.95s] ← segment 1/1 started
    [  5.39s] first chunk  TTFF=5.39s
      ↓ 41 chunks  3892.7 KB   
    [  6.44s] ← ltx2_stream_complete  6.4s  41 chunks
    [  6.64s] saved → video_001_a_mouse_running_through_a_dark_pipe_20260401_184637.mp4  (3892.7 KB)
  ✓ saved  video_001_a_mouse_running_through_a_dark_pipe_20260401_184637.mp4  (3893 KB, 6.6s)

════════════════════════════════════════════════════════════
  Summary: 1 done  0 failed  (1 total)
════════════════════════════════════════════════════════════
  ✓ [01] video_001_a_mouse_running_through_a_dark_pipe_20260401_184637.mp4  3893 KB  6.6s  41 chunks
```


### Options

#### Generation

| Flag | Short | Default | Description |
|---|---|---|---|
| `--prompt TEXT` | `-p` | — | Prompt text. Repeat for multiple prompts. |
| `--count N` | `-n` | `1` | Videos to generate per prompt. |
| `--enhance` | `-e` | off | Enable AI prompt enhancement (GPT rewrite). |
| `--image PATH` | `-i` | — | Input image for image-to-video mode. |
| `--auto-extension` | | off | Enable server-side segment auto-extension. |
| `--loop` | | off | Enable loop generation. |
| `--preset-id ID` | | `simple_custom_prompt` | Override the session preset ID. |

#### Queue & network

| Flag | Short | Default | Description |
|---|---|---|---|
| `--timeout SECS` | `-t` | `240` | Per-video timeout in seconds. |
| `--delay SECS` | `-d` | `1.0` | Pause between consecutive jobs. |
| `--retries N` | `-r` | `1` | Max attempts per job (with exponential backoff). |

#### Output

| Flag | Short | Default | Description |
|---|---|---|---|
| `--output-dir DIR` | `-o` | `.` | Directory to save videos. |
| `--prefix STR` | | `video` | Filename prefix. |
| `--ext EXT` | | `mp4` | File extension. |

#### Misc

| Flag | Description |
|---|---|
| `--verbose` / `-v` | Print full WebSocket protocol trace. |
| `--dry-run` | Show the job queue and exit without connecting. |


### Output filenames

Files are named automatically:

```
{prefix}_{job_number}_{prompt_slug}_{timestamp}.{ext}
```

Example: `video_001_sunset_over_the_ocean_20260401_183000.mp4`

### Notes

**Sequential generation** — each video opens a fresh WebSocket connection. The server enforces one active session per IP, so jobs run one at a time.

**Output format** — files are raw MediaSource chunks (fMP4/WebM). Most players handle them fine. If playback fails, remux with ffmpeg:

```bash
ffmpeg -i input.mp4 -c copy fixed.mp4
```

**IP session limit** — if you see `ip_session_limit`, another session is active on your IP (e.g. a browser tab). The client detects this automatically and retries after 15 seconds.

**Generation time** — typical generation is 5–10 seconds per video once a GPU is assigned. Queue wait time varies by server load.


### Protocol

The client speaks directly to `wss://1080p.fastvideo.org/ws` using the same WebSocket protocol as the web app.

```
connect  →  session_init_v2    (handshake)
         →  simple_generate    (trigger generation)
         ←  gpu_assigned
         ←  ltx2_segment_start / ltx2_segment_complete
         ←  [binary chunks]    (raw video data)
         ←  ltx2_stream_complete
```


