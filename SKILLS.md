---
name: videofentanyl
description: "Use this skill whenever the user wants to generate a video, create a video from a prompt, generate a short clip, generate a long video, use FastVideo or Dreamverse, batch generate videos, or save videos to disk. Trigger on phrases like 'generate a video', 'make a video', 'create a clip', 'generate videos from prompts', 'use fastvideo', 'use dreamverse', 'batch video generation', 'long video', 'short video clip', or any request to synthesize video from a text prompt or image."
---

# fastervideo.py — Video Generation Skill

Two scripts for generating AI videos via the FastVideo WebSocket API.
No API key required. One active session per IP at a time.

| Script | Use for | Output | Typical time |
|---|---|---|---|
| `fastvideo.py` | Short clips | ~5s, single segment, 1080p | 5–10s |
| `dreamverse.py` | Long videos | ~30s, 6 segments, GPT-expanded prompt | 60–120s |

---

## Environment Setup

Resolve `FASTERVIDEO_DIR` — the repo root. All scripts are there.

```
$FASTERVIDEO_DIR/fastvideo.py
$FASTERVIDEO_DIR/dreamverse.py
```

**How to resolve:**
1. Check if the user mentioned the path.
2. Try: `ls fastvideo.py dreamverse.py` (may be in cwd).
3. If unknown, ask once:
   > "Where is your fastervideo.py directory?"

**Install dependency (once):**
```bash
pip install websockets
```

---

## fastvideo.py — Short Clips

One prompt → one WebSocket session → one video file + metadata sidecar.

### Basic usage

```bash
# Single video
python fastvideo.py --prompt "a fox running through a snowy forest"

# Multiple videos from same prompt
python fastvideo.py --prompt "sunset over the ocean" --count 5

# Multiple prompts, one video each
python fastvideo.py \
  --prompt "forest rain timelapse" \
  --prompt "city lights at night"

# Multiple prompts × N videos each
python fastvideo.py \
  --prompt "coral reef" \
  --prompt "arctic tundra" \
  --count 3

# Image-to-video
python fastvideo.py --prompt "the scene comes alive" --image ./photo.jpg

# With AI prompt enhancement (GPT rewrite)
python fastvideo.py --prompt "girl walking in rain" --enhance

# Custom output folder and prefix
python fastvideo.py --prompt "test" --count 3 \
  --output-dir ./videos --prefix clip

# Preview queue without connecting
python fastvideo.py --prompt "test" --count 5 --dry-run

# Full protocol trace
python fastvideo.py --prompt "test" --verbose
```

### All flags

| Flag | Short | Default | Description |
|---|---|---|---|
| `--prompt TEXT` | `-p` | — | Prompt. Repeat for multiple. |
| `--count N` | `-n` | `1` | Videos per prompt. |
| `--enhance` | `-e` | off | GPT prompt rewrite before generation. |
| `--image PATH` | `-i` | — | Input image for image-to-video. |
| `--auto-extension` | | off | Server-side segment auto-extension. |
| `--loop` | | off | Loop generation. |
| `--output-dir DIR` | `-o` | `.` | Save directory. |
| `--prefix STR` | | `video` | Filename prefix. |
| `--ext EXT` | | `mp4` | File extension. |
| `--timeout SECS` | `-t` | `240` | Per-video timeout. |
| `--delay SECS` | `-d` | `1.0` | Pause between jobs. |
| `--retries N` | `-r` | `1` | Max attempts (exponential backoff). |
| `--verbose` | `-v` | off | Full WebSocket protocol trace. |
| `--dry-run` | | off | Show queue, don't connect. |

---

## dreamverse.py — Long Videos (~30s)

One prompt → GPT expands into 6 segment descriptions → 6-segment video.
`enhancement_enabled` is **on by default** — this is what makes it long-form.

### Basic usage

```bash
# Single long video
python dreamverse.py --prompt "a kid burps into a tunnel, with a huge echo"

# Queue of videos
python dreamverse.py --prompt "volcano eruption at night" --count 3

# Multiple prompts
python dreamverse.py \
  --prompt "dog learns to skateboard" \
  --prompt "timelapse of a city waking up"

# Skip GPT expansion (use prompt as-is)
python dreamverse.py --prompt "very detailed scene description..." --no-enhance

# Custom output
python dreamverse.py --prompt "test" --output-dir ./videos --verbose
```

### Dreamverse-specific flags

| Flag | Default | Description |
|---|---|---|
| `--no-enhance` | off | Disable GPT prompt expansion (on by default). |
| `--preset-id ID` | `custom_editable` | Session preset ID. |
| `--preset-label STR` | `Custom rollout` | Session preset label. |
| `--timeout SECS` | `480` | Longer default — 6 segments take ~2 min. |
| `--delay SECS` | `2.0` | Slightly longer inter-job pause. |

All other flags (`--prompt`, `--count`, `--image`, `--output-dir`, `--prefix`, `--retries`, `--verbose`, `--dry-run`) work identically to fastvideo.py.

---

## Output Files

Every successful job produces two files:

```
video_001_a_fox_running_20260401_183000.mp4   ← video
video_001_a_fox_running_20260401_183000.txt   ← metadata sidecar
```

**fastvideo.py sidecar** contains: prompt, endpoint, preset, enhance flag,
timestamp, elapsed time, TTFF, latency, chunk count, bytes, server notices.

**dreamverse.py sidecar** contains all of the above plus the **6 GPT-expanded
segment prompts** — useful for re-using or editing individual segments.

Filename pattern: `{prefix}_{N:03d}_{prompt_slug}_{timestamp}.{ext}`

---

## Protocol Reference

### fastvideo.py flow
```
connect  →  session_init_v2    (preset: simple_custom_prompt, single_clip_mode: true)
         →  simple_generate    (prompt sent here)
         ←  gpu_assigned
         ←  ltx2_segment_start
         ←  [binary fMP4 chunks]
         ←  ltx2_stream_complete
```

### dreamverse.py flow
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
python fastvideo.py \
  --prompt "prompt one" \
  --prompt "prompt two" \
  --prompt "prompt three" \
  --count 2 \
  --output-dir ./out
```
Produces 6 videos total (3 prompts × 2 each), sequentially.

### Generate a long video with verbose logging
```bash
python dreamverse.py \
  --prompt "a scientist discovers something extraordinary" \
  --verbose \
  --output-dir ./out
```

### Retry flaky jobs automatically
```bash
python fastvideo.py --prompt "test" --count 10 --retries 3 --delay 2
```

### Check what would be generated without connecting
```bash
python dreamverse.py --prompt "test" --count 5 --dry-run
```

---

## Troubleshooting

**Stuck after `gpu_assigned` with no chunks** — this was a known bug (fixed).
Ensure you have the latest script. The issue was `simple_generate` not being
sent on socket open.

**`ip_session_limit` error** — another session is active on your IP (e.g. a
browser tab on `1080p.fastvideo.org` or `dreamverse.fastvideo.org`). The client
detects this and retries automatically after 15 seconds. Close the browser tab
to resolve immediately.

**Video won't play** — output is raw fMP4/WebM MediaSource chunks. Remux:
```bash
ffmpeg -i input.mp4 -c copy fixed.mp4
```

**Timeout on dreamverse** — long videos take 60–120s. Default timeout is 480s.
If you're hitting it, increase: `--timeout 600`

**Queue gets stuck on retry** — check `--retries` is > 1. Default is 1 attempt
(no retry). Use `--retries 3` for resilience.
