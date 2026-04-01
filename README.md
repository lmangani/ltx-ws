# fastervideo.py
Hackish Fastvideo Client in Py

## Install
```
...
```

## Usage Example

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
