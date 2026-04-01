# <img width="488" height="58" alt="image" src="https://github.com/user-attachments/assets/7065ad1b-020e-42f2-862d-5c0c3654dd65" />

A command-line tool for [FastVideo](https://fastvideo.org/) 

> Reverse-engineered from the FastVideo web app's WebSocket protocol _(LTX-2 backend)_


### Requirements

- Python 3.10+
- `websockets` library

### Install

Just grab the latest script directly:

```bash
curl -O https://raw.githubusercontent.com/lmangani/videofentanyl/main/fastvideo.py
curl -O https://raw.githubusercontent.com/lmangani/videofentanyl/main/dreamverse.py
```

### Usage

#### Dreamverse (6 scenes, 30s)
```
python dreamverse.py --prompt "a dog learns to fly" --verbose
```

#### 1080p 5s

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

#### Full Example (dreamverse)

```
% python dreamverse.py --prompt "a dog learns to fly" --verbose

════════════════════════════════════════════════════════════
  Dreamverse Queue — 1 job(s)
  Endpoint : wss://dreamverse.fastvideo.org/ws
  Timeout  : 480s per video
  Delay    : 2.0s between jobs
════════════════════════════════════════════════════════════

  ┌─ Job 01/1  dreamverse_001_a_dog_learns_to_fly_20260401_191248.mp4
  │  prompt: 'a dog learns to fly'
  └──────────────────────────────────────────────────
    [  0.61s] connected ✓
    [  0.61s] → session_init_v2  prompt='a dog learns to fly'
    [  0.61s] ← queue_status  position=0  gpus=2/4
    [  0.78s] ← gpu_assigned  gpu=2  session_timeout=300s
    [  0.78s] ← loop_generation_updated  enabled=False
    [  0.78s] ← generation_paused_updated  paused=True
    [  0.78s] ← rewrite started  model=gpt-oss-120b  (expanding prompt into segments…)
    [  5.84s] ← generation_paused_updated  paused=False
    [  5.84s] ← seed_prompts_updated  6 segments  reason=rewrite  fallback=False
      seg 1: A sunny park glows with green grass and dappled light as a golden‑retriever name…
      seg 2: A gust lifts the kite higher, tugging the string and pulling Buddy forward. He d…
      seg 3: Buddy flaps his ears and paws, trying to stay aloft as the kite steadies him a f…
      seg 4: The camera pans to a colorful hot‑air balloon drifting nearby, its burner flicke…
      seg 5: The balloon lifts, and Buddy feels the lift of warm air beneath him. He gasps, t…
      seg 6: The balloon gently descends, and Buddy’s harness releases, letting him glide bac…
    [  5.84s] ← rewrite complete  fallback=False
    [  5.85s] ← stream start  segments=6  mode=av_fmp4
    [  5.85s] ← seed_prompts_reset_applied  reason=initial_rewrite
    [  5.85s] ← segment 1 source=curated
    [  5.85s] ← segment 1/6 started  prompt='A sunny park glows with green grass and dappled light as a g'
    [  9.83s] ← media_init  mime=video/mp4; codecs="avc1.42E01E,mp4a.40.2"  stream_id=seg001-611d6ad2
    [  9.88s] first chunk  TTFF=9.88s
      ↓ 46 chunks  7977.4 KB  seg 0/6       [ 11.30s] ← media_segment_complete
    [ 11.30s] ← step_complete
    [ 11.30s] ← segment 1/6 complete
    [ 11.30s] ← segment 2 source=curated
    [ 11.30s] ← segment 2/6 started  prompt='A gust lifts the kite higher, tugging the string and pulling'
    [ 14.27s] ← media_init  mime=video/mp4; codecs="avc1.42E01E,mp4a.40.2"  stream_id=seg002-a810b25d
      ↓ 90 chunks  14719.3 KB  seg 1/6       [ 14.94s] ← media_segment_complete
    [ 14.94s] ← step_complete
    [ 14.94s] ← segment 2/6 complete
    [ 14.94s] ← segment 3 source=curated
    [ 14.94s] ← segment 3/6 started  prompt='Buddy flaps his ears and paws, trying to stay aloft as the k'
    [ 18.79s] ← media_init  mime=video/mp4; codecs="avc1.42E01E,mp4a.40.2"  stream_id=seg003-ef3aeb1f
      ↓ 132 chunks  21596.2 KB  seg 2/6       [ 19.46s] ← media_segment_complete
    [ 19.46s] ← step_complete
    [ 19.46s] ← segment 3/6 complete
    [ 19.46s] ← segment 4 source=curated
    [ 19.46s] ← segment 4/6 started  prompt='The camera pans to a colorful hot‑air balloon drifting nearb'
    [ 23.25s] ← media_init  mime=video/mp4; codecs="avc1.42E01E,mp4a.40.2"  stream_id=seg004-acaae5e3
      ↓ 180 chunks  29262.0 KB  seg 3/6       [ 23.99s] ← media_segment_complete
    [ 23.99s] ← step_complete
    [ 23.99s] ← segment 4/6 complete
    [ 23.99s] ← segment 5 source=curated
    [ 23.99s] ← segment 5/6 started  prompt='The balloon lifts, and Buddy feels the lift of warm air bene'
    [ 27.68s] ← media_init  mime=video/mp4; codecs="avc1.42E01E,mp4a.40.2"  stream_id=seg005-b20cb0eb
      ↓ 219 chunks  32688.7 KB  seg 4/6       [ 28.40s] ← media_segment_complete
    [ 28.40s] ← step_complete
    [ 28.40s] ← segment 5/6 complete
    [ 28.40s] ← segment 6 source=curated
    [ 28.40s] ← segment 6/6 started  prompt='The balloon gently descends, and Buddy’s harness releases, l'
    [ 32.09s] ← media_init  mime=video/mp4; codecs="avc1.42E01E,mp4a.40.2"  stream_id=seg006-0048cc93
      ↓ 261 chunks  37140.8 KB  seg 5/6       [ 32.54s] ← media_segment_complete
    [ 32.54s] ← step_complete
    [ 32.54s] ← segment 6/6 complete

    [ 32.54s] ← ltx2_stream_complete  32.5s  6 segments  261 chunks
    [ 33.06s] saved → dreamverse_001_a_dog_learns_to_fly_20260401_191248.mp4  (37140.8 KB)
  ✓ saved  dreamverse_001_a_dog_learns_to_fly_20260401_191248.mp4  (37141 KB, 6 segments, 33.1s)

════════════════════════════════════════════════════════════
  Summary: 1 done  0 failed  (1 total)
════════════════════════════════════════════════════════════
  ✓ [01] dreamverse_001_a_dog_learns_to_fly_20260401_191248.mp4  37141 KB  33.1s  6 segments  261 chunks
```

#### Full Example (1080p)

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


