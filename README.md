# ltx-ws

**Local LTX-2.3 video over WebSocket on Apple Silicon (MLX).** This repository provides:

| Component | Role |
|-----------|------|
| [`server.py`](server.py) | WebSocket server: loads [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx), runs **T2V/I2V/A2V/retake/extend**, streams **MP4** to clients. |
| [`videofentanyl.py`](videofentanyl.py) | CLI client: queues jobs, speaks the same JSON + binary protocol (`--mode ltx` + `--server`). |
| [`ltx_mlx_backend.py`](ltx_mlx_backend.py) | MLX pipeline adapter, Hugging Face weight resolution, frame/spatial alignment. |
| [`scripts/benchmark_local_generation.py`](scripts/benchmark_local_generation.py) | Spawns (or attaches to) `server.py`, runs one client job, prints timings + `BENCHMARK_JSON:…`. |

Everything below is **local-only**: your Mac, Metal / MLX, and optional Hugging Face downloads. No hosted inference is required.

---

## Features

- **MLX on Metal** — Inference via [`ltx_pipelines_mlx`](https://github.com/dgrauet/ltx-2-mlx): `TextToVideoPipeline`, `ImageToVideoPipeline`, `AudioToVideoPipeline`, `RetakePipeline`, `ExtendPipeline`.
- **Automatic weight download** — For a Hugging Face repo id (`org/model`), the server calls [`huggingface_hub.snapshot_download`](https://huggingface.co/docs/huggingface_hub/guides/download) on load (equivalent to `huggingface-cli download`). Resumes partial downloads.
- **Default weights** — [`dgrauet/ltx-2.3-mlx`](https://huggingface.co/dgrauet/ltx-2.3-mlx) (full MLX bf16; very large). Use [`ltx-2.3-mlx-q8`](https://huggingface.co/dgrauet/ltx-2.3-mlx-q8) or [`-q4`](https://huggingface.co/dgrauet/ltx-2.3-mlx-q4) for less RAM/disk.
- **Weight paths** — `./models/<org>__<name>/` by default, or `--model-dir`, or base directory `$VIDEOFENTANYL_MODELS`.
- **Per-job overrides** — Client `simple_generate` may send `seed`, `num_frames`, `height`, `width`, `num_steps` (server snaps frames to **8k+1** and resolution to **multiples of 32**).
- **Image/audio/video inputs** — Session / generate payloads support image keys plus `audio_input` and `source_video`; client supports `--image`, `--audio`, `--video` (path or `http(s)` URL).
- **Cross-machine safe media upload** — Client serializes `--image`, `--audio`, and `--video` as payload data, so server and client can run on different hosts without shared filesystem paths.
- **Operation routing** — `--generation-mode generate|a2v|retake|extend` maps to matching MLX pipelines, including `--retake-start`, `--retake-end`, `--extend-frames`, `--extend-direction`.
- **Long runs without stalling** — Server emits `generation_keepalive` JSON during inference; client may send `generation_status` and receive `generation_status_ack` (with reserved `model_progress` for future use).
- **Disconnect safety** — Finished MP4s are copied to `--spill-dir` if the client drops while streaming (`fvserver_completed` by default).
- **Single-flight generation** — One active MLX job at a time; extra clients wait in a fair queue (`queue_status` / `gpu_assigned`).
- **Client batching** — Multiple `--prompt`s, `--count`, `--delay`, `--retries`, `--dry-run`, `--verbose`.
- **Autocontinue / autoconcat** — Chain clips using the last frame of each as the next start image; optional `ffmpeg` stream-copy merge into one file (`--autoconcat`).
- **Audiocontinue** — `--audiocontinue` auto-enables `--autocontinue --autoconcat --autocompact`, splits one input audio track into clip-length chunks, and feeds each chunk sequentially in `a2v` mode.

---

## Requirements

| | |
|--|--|
| **Hardware** | Apple Silicon (M1 / M2 / M3 / M4 …). |
| **OS** | macOS (Metal). |
| **Python** | **3.11+** recommended for `server.py` and ltx-2-mlx; 3.10+ may work for the client only. |
| **ffmpeg** | Optional; required on the **client** machine if you use `--autoconcat`. |
| **Disk / RAM** | Depends on model (bf16 ≫ q8 ≫ q4). Plan tens of GB disk for full bf16 weights; see [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) model table. |

Python packages: see [`requirements.txt`](requirements.txt) (`websockets`, `av`, `Pillow`, `huggingface_hub`). **MLX** packages are installed separately from the ltx-2-mlx monorepo (comments in `requirements.txt`).

---

## Install

```bash
git clone https://github.com/lmangani/ltx-ws.git
cd ltx-ws

uv venv --python 3.12 --seed
source .venv/bin/activate   # or: source .venv/bin/activate.fish

uv pip install -r requirements.txt
uv pip install \
  "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git#subdirectory=packages/ltx-core-mlx" \
  "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git#subdirectory=packages/ltx-pipelines-mlx"
```

Use `pip` instead of `uv pip` if you prefer.

**Gated or private Hub repos:** set [`HF_TOKEN`](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) or run `huggingface-cli login`.

---

## Model weights

1. **Hugging Face repo id** — On first `server.py` startup, weights are downloaded under `./models/<org>__<name>/` (unless `--model-dir` or `$VIDEOFENTANYL_MODELS` applies).
2. **Local directory** — Pass an existing folder path to `--model` instead of `org/name`.

**Single-folder names (e.g. `./models/ltx-2.3-mlx/`)**

Hugging Face ids look like `author/model` (with a **slash**). A bare name like `ltx-2.3-mlx` is **not** a Hub id; if it is passed through unchanged, `ltx_pipelines_mlx` may try `https://huggingface.co/ltx-2.3-mlx` and fail with **404**.

The server resolves local weights in this order before any Hub download:

1. `--model` is an existing directory path (relative or absolute). For **relative** paths it checks **the current working directory first**, then the **repository root** (the folder that contains `ltx_mlx_backend.py`), so you can start the server from another directory and still use `./models/...` next to the checkout.
2. **`--model-dir`** uses the same rule (cwd, then repo root) when the path is relative.
3. **`<repo>/models/<name>/`** and **`./models/<name>/` from cwd** for a shorthand leaf name (no `/` in `<name>`) — then `--model <name>` alone is enough if one of those folders exists.

**RAM-based default (`--model` omitted or `auto`)**

If you do not pass `--model`, the server defaults to **`auto`**: it reads total physical RAM (on macOS, `sysctl hw.memsize`; Apple Silicon uses **unified memory**, so this is the same pool MLX uses—there is no separate VRAM to probe) and picks a pre-converted MLX repo:

| Variant | Hugging Face repo | Approx. weights | Auto when RAM is |
|--------|-------------------|-----------------|------------------|
| bf16 | `dgrauet/ltx-2.3-mlx` | ~42 GB | **≥ 64 GiB** |
| int8 | `dgrauet/ltx-2.3-mlx-q8` | ~21 GiB | **≥ 32 GiB** and under 64 GiB |
| int4 | `dgrauet/ltx-2.3-mlx-q4` | ~12 GiB | **under 32 GiB** (still chosen if RAM is below 16 GiB, with a startup warning) |

Pass an explicit **`--model <repo or path>`** to skip auto-selection. You can also set **`LTX_WS_MODEL`** to the default when the flag is omitted (e.g. `LTX_WS_MODEL=dgrauet/ltx-2.3-mlx-q8` or `LTX_WS_MODEL=auto`).

**Practical defaults**

```bash
# Same as omitting --model: resolve from installed RAM
python server.py --model auto

# Smaller quantised model (recommended for many machines)
python server.py --model dgrauet/ltx-2.3-mlx-q8

# Explicit download directory
python server.py --model dgrauet/ltx-2.3-mlx-q4 --model-dir "$HOME/mlx-weights/ltx-q4"

# Custom snapshot folder under ./models/ (any of these work if the directory exists)
python server.py --model ./models/ltx-2.3-mlx
python server.py --model ltx-2.3-mlx
python server.py --model ltx-2.3-mlx --model-dir ./models/ltx-2.3-mlx
```

---

## Run the server

```bash
python server.py
```

Listens on **`ws://0.0.0.0:8765/ws`** by default. Model path and pipeline registry are resolved at startup; the first use of each pipeline (`t2v`/`i2v`/`a2v`/`retake`/`extend`) is lazy-loaded.

Useful variants:

```bash
python server.py --port 9000
python server.py --model dgrauet/ltx-2.3-mlx-q8 --infer-steps 8 --num-frames 65
python server.py --height 512 --width 768 --mlx-low-memory
python server.py --enable-lora --lora Kijai/LTX2.3_comfy 1.0
# multiple LoRAs (repeat --lora)
python server.py --enable-lora \
  --lora Kijai/LTX2.3_comfy 1.0 \
  --lora /path/to/another_lora.safetensors 0.6
# enable default LoRA via env
LTX_WS_ENABLE_LORA=1 python server.py
# override default LoRA via env (still requires enable)
LTX_WS_DEFAULT_LORA="https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors" \
LTX_WS_DEFAULT_LORA_SCALE="1.0" LTX_WS_ENABLE_LORA=1 python server.py
# multi-default via env (comma-separated path:scale)
LTX_WS_DEFAULT_LORAS="Kijai/LTX2.3_comfy:1.0,/path/to/another_lora.safetensors:0.6" \
LTX_WS_ENABLE_LORA=1 python server.py
```

---

## Run the client (local MLX)

`videofentanyl.py` must use **`--mode ltx`** (default) and **`--server`** pointing at your `server.py` WebSocket URL.

```bash
python videofentanyl.py --server ws://127.0.0.1:8765/ws --prompt "a fox running through snow"

python videofentanyl.py --server ws://127.0.0.1:8765/ws --prompt "sunset" --count 3

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "animate gently" --image ./photo.jpg

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --generation-mode a2v --audio ./music.wav --prompt "a musician performing"

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --generation-mode retake --video ./source.mp4 --retake-start 1 --retake-end 3 \
  --prompt "A different scene"

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --generation-mode extend --video ./source.mp4 --extend-frames 2 --extend-direction after \
  --prompt "Continue the motion"

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --generation-mode a2v --audio ./song.wav --count 5 \
  --num-frames 121 --audiocontinue --prompt "music video"

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "a river in a canyon" --count 4 --autocontinue

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "timelapse" --count 3 --autocontinue --autoconcat
```

### Duration and aspect-ratio examples

`--num-frames` controls clip duration (`seconds ≈ frames / 24`).  
Use `8k+1` frame counts (for example: `49`, `97`, `121`, `193`).

```bash
# ~2.0s clip (49 / 24)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "cinematic close-up of a singer" --num-frames 49

# ~4.0s clip (97 / 24)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "a fox running through snow" --num-frames 97

# ~5.0s clip (121 / 24)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "street dance performance at night" --num-frames 121
```

Social/vertical outputs are controlled with `--height` and `--width` (server snaps to multiples of 32):

```bash
# 9:16 vertical (stories/reels style), text-to-video
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "fashion model walking through neon alley" \
  --height 1024 --width 576 --num-frames 97

# 9:16 vertical with starting image (image-to-video)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "slow cinematic movement, subtle wind in hair" \
  --image ./vertical_start.jpg \
  --height 1024 --width 576 --num-frames 121

# 4:5 vertical post format with starting image
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "product reveal with soft studio lighting" \
  --image ./post_4x5.jpg \
  --height 960 --width 768 --num-frames 97

# 1:1 square format
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "loopable abstract animation" \
  --height 768 --width 768 --num-frames 97
```

Tip: when using a starting image, match `--height/--width` to the image orientation for best framing (portrait source image + portrait output).

**Dry-run** (print queue, no network):

```bash
python videofentanyl.py --server ws://127.0.0.1:8765/ws --prompt "test" --count 5 --dry-run
```

---

## Benchmark

From the repo root (uses `.venv/bin/python3` when present):

```bash
./scripts/benchmark_local_generation.py
./scripts/benchmark_local_generation.py --port 9000 --no-server --server-url ws://studio.local:9000/ws
```

The last line of output is **`BENCHMARK_JSON:{...}`** for scripts. Outputs go under `benchmark_runs/` by default.

---

## Repository layout

```
server.py                 # WebSocket server (MLX)
videofentanyl.py          # CLI client
ltx_mlx_backend.py        # MLX generator + HF snapshot paths
requirements.txt
scripts/benchmark_local_generation.py
models/                   # default HF snapshot dir (gitignored)
third_party/LTX-2/        # optional submodule: upstream Lightricks LTX-2 reference
```

---

## WebSocket protocol (local server)

JSON messages use a `type` field. After `simple_generate`, the server streams **raw MP4 bytes** in chunks (`--chunk-size` on the server). Typical sequence:

```
client →  session_init_v2          (session + optional initial image for i2v / autocontinue)
client →  simple_generate         (prompt + mode; optional seed/frames/size/steps + image/audio/video keys)
server ←  queue_status             (while waiting for the single MLX slot)
server ←  gpu_assigned             (generation_id, gpu_id reports mlx:0)
server ←  ltx2_stream_start       (single-segment stream)
server ←  ltx2_segment_start
server ↔  generation_keepalive     (periodic JSON during inference)
client →  generation_status       (optional)
server ←  generation_status_ack   (phase, elapsed_s, generation_id)
server ←  [binary MP4 chunks]
server ←  ltx2_segment_complete
server ←  ltx2_stream_complete
server ←  latency                 (timing metadata)
```

The client implements this flow for `--mode ltx` when `--server` is set.

---

## `server.py` CLI reference

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address. |
| `--port` | `8765` | TCP port; path **`/ws`**. |
| `--model` | `auto` (or `$LTX_WS_MODEL`) | HF repo id, local weights directory, or **`auto`** (RAM → bf16 / q8 / q4; see [Model weights](#model-weights)). |
| `--model-dir` | *(see Models)* | Store HF snapshot here; overrides default `./models/<org>__<name>/`. |
| `--enable-lora` | off | Enable global LoRA handling on the server. |
| `--lora <path_or_repo_or_url> <scale>` | off unless enabled | Global LoRA(s) applied to all requests; **repeat flag** to stack multiple LoRAs. |
| `--num-frames` | `97` | Target length; adjusted to **8k+1** (e.g. 9, 25, 49, 97). |
| `--height` | `480` | Snapped to multiple of **32**. |
| `--width` | `704` | Snapped to multiple of **32**. |
| `--fps` | `24` | Nominal rate (mux behaviour follows pipeline). |
| `--infer-steps` | `8` | One-stage distilled step count (minimum 1). |
| `--mlx-low-memory` | off | `low_memory=True` in ltx-2-mlx (slower, less RAM). |
| `--chunk-size` | `65536` | Max bytes per WebSocket binary frame. |
| `--spill-dir` | `fvserver_completed` | Salvage directory on client disconnect. |
| `--verbose` | off | Extra per-connection logs. |

Default global LoRA is **disabled unless enabled** with `--enable-lora` (or env below).  
When enabled, LoRA defaults can be configured in `server.py` constants and overridden via env:
- `LTX_WS_ENABLE_LORA`
- `LTX_WS_DEFAULT_LORA`
- `LTX_WS_DEFAULT_LORA_SCALE`
- `LTX_WS_DEFAULT_LORAS` (comma-separated `path:scale,path:scale`) for multiple defaults

LoRA artifacts are resolved from local path, URL, or Hugging Face repo id. Downloaded LoRAs are cached under `./loras/` by default; override with:
- `VIDEOFENTANYL_LORA_DIR`
When LoRA is enabled, server startup now pre-resolves/downloads global LoRAs (fail-fast), matching the main model preflight behavior.

---

## `videofentanyl.py` CLI reference (local `--server`)

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `ltx` | Use `ltx` for this stack (**requires `--server`**). |
| `--server` | — | WebSocket URL, e.g. `ws://127.0.0.1:8765/ws`. |
| `--prompt` / `-p` | *(built-in demo)* | Repeat for multiple prompts. |
| `--count` / `-n` | `1` | Videos per prompt. |
| `--seed`, `--num-frames`, `--height`, `--width`, `--num-steps` | — | Per-job generation overrides for local server. |
| `--generation-mode` | `generate` | Local route: `generate`, `a2v`, `retake`, `extend`. |
| `--image` / `-i` | — | Image-to-video: path or `http(s)` URL. |
| `--audio` | — | Audio-to-video input for `--generation-mode a2v`. |
| `--video` | — | Source video for `--generation-mode retake|extend`. |
| `--retake-start`, `--retake-end` | — | Retake frame range for `--generation-mode retake`. |
| `--extend-frames`, `--extend-direction` | — | Extend parameters for `--generation-mode extend`. |
| `--enhance` / `-e` | off | Sets `enhancement_enabled` in the client handshake; **this MLX server does not run GPT rewrite** — generation uses the prompt you send. |
| `--preset-id`, `--preset-label` | — | Override `session_init_v2` preset fields. |
| `--auto-extension`, `--loop` | off | Forwarded session flags. |
| `--output-dir` / `-o` | `.` | Save directory. |
| `--prefix` | `ltx` | Filename prefix. |
| `--ext` | `mp4` | Extension. |
| `--delay` / `-d` | `1.0` | Seconds between jobs. |
| `--retries` / `-r` | `1` | Retries per job. |
| `--idle-timeout` | unlimited with `--server` | Seconds of silence before a WebSocket ping probe. |
| `--verbose` / `-v` | off | Full protocol trace. |
| `--dry-run` | off | Print plan, exit. |
| `--autocontinue` | off | Last frame → next job’s start image. |
| `--autoconcat` | off | After queue: `ffmpeg -c copy` merge (**requires** `--autocontinue` + ffmpeg). |
| `--audiocontinue` | off | Music-video helper for `a2v`: implies `--autocontinue --autoconcat --autocompact`, splits `--audio` per clip and assigns one segment per job. |

Saved files look like: **`{prefix}_{NNN}_{slug}_{timestamp}.mp4`**. After a successful **`--autoconcat`**, fragments are removed and **`{prefix}_merged_{timestamp}.mp4`** is written.

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| **Player won’t open MP4** | Fragments are progressive / fMP4-style; remux: `ffmpeg -i in.mp4 -c copy out.mp4`. |
| **`Missing ltx_pipelines_mlx`** | Install the two `uv pip install …ltx-2-mlx.git#subdirectory=…` lines from `requirements.txt`. |
| **`huggingface_hub` errors** | Install deps; check `HF_TOKEN` for gated models; ensure enough free disk. |
| **OOM / slow** | Use `--model dgrauet/ltx-2.3-mlx-q8` or `-q4`, lower `--num-frames` / resolution, or `--mlx-low-memory`. |
| **Need exact runtime params for debugging** | Check server logs for `Generation effective params:` — it prints mode, seed, final `size/frames/steps`, original requested values, input types, retake/extend args, LoRA/video-conditioning counts, and resolved model path. |
| **Video looks fine for ~5s then turns into garbage (any resolution)** | Often **not** a WebSocket bug: the server reads the full MP4 from disk and sends it in order; the client concatenates chunks verbatim. First check **(1)** temporal length — at 24 fps, **121 frames ≈ 5.0s**; corruption starting right after that may track **model / distilled sampler** behaviour more than transport. **(2)** Compare **with vs without** global LoRA. **(3)** Try **fewer frames** (e.g. 97) vs more (121+) with the same prompt. **(4)** Inspect the file **on the server** before download (`ffprobe` on the temp output or a spill copy) to see whether corruption exists **before** streaming. The backend now passes the server **`--fps`** into `generate_and_save` (mapped to `frame_rate` when required by ltx-2-mlx) so mux timing matches CLI defaults. |
| **Port already in use** | `--port` on server and matching URL on client. |
| **`autoconcat` failed** | Install `ffmpeg` on the client host; fragments are kept if merge fails. |

---

## References

- Repo: [github.com/lmangani/ltx-ws](https://github.com/lmangani/ltx-ws)  
- Inference stack: [github.com/dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)  
- Default weights: [huggingface.co/dgrauet/ltx-2.3-mlx](https://huggingface.co/dgrauet/ltx-2.3-mlx)  
- Hub downloads: [huggingface.co/docs/huggingface_hub/guides/download](https://huggingface.co/docs/huggingface_hub/guides/download)
