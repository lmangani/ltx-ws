# ltx-ws

**Local LTX-2.3 video generation on Apple Silicon** — Web UI, CLI, and MCP over a single WebSocket server.

<img width="800" height="776" alt="LTX-WS Web UI" src="https://github.com/user-attachments/assets/4b9c967d-cc42-44a6-964d-3f24be3aa173" />

Generate text-to-video, image-to-video, audio-to-video, retake, extend, and multi-clip chains on your Mac. Outputs are saved locally; no cloud inference required.

---

## What you can do

| Capability | Description |
|------------|-------------|
| **Text / image / audio to video** | Standard generation modes from prompt, still image, or audio track |
| **Multi-clip chains** | Build longer videos with **autocontinue** (last frame → next clip) or **native extend** (extend prior clip in-place) |
| **Retake & extend** | Edit a segment of existing footage or append/prepend new motion |
| **Face swap (BFS V3)** | Replace face in reference video using identity image + head-swap LoRA (Comfy-aligned MLX port) |
| **ID-LoRA (face + voice)** | New talking clip from a face still + ~5s identity audio; structured `[VISUAL]/[SPEECH]/[SOUNDS]` prompt |
| **Web UI** | Browser library, progress, LoRA picker, duration presets, clip multiplier |
| **CLI client** | Scriptable batch runs, autocontinue, PyAV merge (`autoconcat`) |
| **MCP tools** | Drive generation from Cursor, Claude, or other MCP clients |
| **LoRA** | Per-request or server-wide style adapters (optional) |

**Agent docs:** [`DIRECTOR.md`](DIRECTOR.md) (prompting & shot planning), [`AGENTS.md`](AGENTS.md) (MCP & pipelines), [`CLAUDE.md`](CLAUDE.md) (pointer to agent guides).

---

## Requirements

- **Apple Silicon Mac** with macOS (Metal / MLX)
- **Python 3.11+** (3.12 recommended)
- **Node.js 18+** — only to build the Web UI (`web/dist/`)
- **PyAV (`av`)** — bundled via `requirements.txt`; powers audio trim, a2v loading, clip merge (`autoconcat`), and mux (no system ffmpeg install)
- **Disk / RAM** — depends on model variant (q4 ≈ 12 GB weights, q8 ≈ 21 GB, bf16 ≈ 42 GB). Use a [quantized MLX model](https://huggingface.co/dgrauet/ltx-2.3-mlx-q8) unless you have plenty of unified memory.

Weights must be **MLX-converted** checkpoints from the [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) ecosystem — not standard PyTorch `Lightricks/LTX-2.3` weights.

---

## Install

```bash
git clone https://github.com/lmangani/ltx-ws.git
cd ltx-ws

# Python environment (uv recommended; use venv + pip if you prefer)
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install \
  "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@v0.14.19#subdirectory=packages/ltx-core-mlx" \
  "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@v0.14.19#subdirectory=packages/ltx-pipelines-mlx"

# Web UI (first time, or after editing web/)
cd web && npm install && npm run build && cd ..
```

For gated Hugging Face models, set [`HF_TOKEN`](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) or run `huggingface-cli login`.

Classic **venv + pip**: use `python3.12 -m venv .venv`, `pip install -r requirements.txt`, and the same two `ltx-2-mlx` package lines above.

---

## Quick start

```bash
source .venv/bin/activate

# Start server + Web UI (downloads weights on first run if needed)
python server.py --model dgrauet/ltx-2.3-mlx-q8
```

| Service | URL |
|---------|-----|
| Web UI | http://127.0.0.1:8765/ |
| WebSocket | ws://127.0.0.1:8765/ws |

Open the Web UI, enter a prompt, and generate. Clips are saved under `./web_outputs/`.

**CLI** (second terminal, same venv):

```bash
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "a fox running through snow"
```

**MCP** (with `server.py` already running):

```bash
python mcp_server.py --server-url ws://127.0.0.1:8765/ws
```

---

## Web UI

Served automatically by `server.py` on the same port as the WebSocket API.

- **Generate** — modes, resolution, duration, steps, seed, LoRA presets
- **Multi-clip** — set clip count (e.g. 5s × 2); choose **autocontinue** (frame chain) or **extend video** (native extend on prior MP4)
- **Library** — browse past clips; reuse settings or pick a library clip as **source video** for retake / extend / lip sync
- **Progress** — denoising steps and ETA over SSE

Outputs: `./web_outputs/` (override with `--web-output-dir`).

Disable the UI and run WebSocket-only:

```bash
python server.py --no-web-ui --model dgrauet/ltx-2.3-mlx-q8
```

---

## CLI examples

Point `videofentanyl.py` at your server with `--server ws://HOST:PORT/ws`.

```bash
# Basic text-to-video
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "sunset over the ocean"

# Image-to-video
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "gentle camera push-in" --image ./photo.jpg

# ~5 second clip (121 frames @ 24 fps)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "street dance at night" --num-frames 121

# Chained clips + merged output (PyAV — included in requirements.txt)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "drone over coastline" --count 3 --autocontinue --autoconcat

# Retake / extend on an existing file
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --generation-mode retake --video ./clip.mp4 \
  --retake-start 1 --retake-end 3 --prompt "new action in the middle"

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --generation-mode extend --video ./clip.mp4 \
  --extend-frames 8 --extend-direction after --prompt "continue forward"
```

**Duration:** `seconds ≈ num_frames / 24`. Use frame counts of the form **8k+1** (e.g. 49 ≈ 2s, 97 ≈ 4s, 121 ≈ 5s). The server snaps invalid values.

**Vertical / square:** set `--height` and `--width` (multiples of 32), e.g. `--height 1024 --width 576` for 9:16.

Run `python videofentanyl.py --help` and `python server.py --help` for all flags.

---

## Models & weights

| Variant | Hugging Face repo | Rough size | When to use |
|---------|-------------------|------------|-------------|
| q4 | `dgrauet/ltx-2.3-mlx-q4` | ~12 GB | Tight RAM / disk |
| q8 | `dgrauet/ltx-2.3-mlx-q8` | ~21 GB | **Good default** for most Macs |
| bf16 | `dgrauet/ltx-2.3-mlx` | ~42 GB | Best quality; 64 GB+ RAM |

```bash
# Explicit model
python server.py --model dgrauet/ltx-2.3-mlx-q8

# Auto-pick from installed RAM
python server.py --model auto

# Reuse weights already on disk
ls models/
python server.py --model ./models/ltx-2.3-mlx-q8
```

Downloads land in `./models/` (gitignored — safe across `git pull`). Override location with `--model-dir` or `$VIDEOFENTANYL_MODELS`.

---

## Chaining longer videos

| Method | Best for | Behavior |
|--------|----------|----------|
| **autocontinue** | New scenes, camera moves | Each clip starts from the **last frame** of the previous clip |
| **native extend** | Same shot, more duration | Clip 1 generates; clip 2+ **extends** the prior MP4 (cumulative length) |
| **autoconcat** | Deliver one file | Merges successful clips with PyAV (stream copy) |

In the Web UI, set **Clips × duration** and pick the chain method. In the CLI, use `--count N --autocontinue` and optionally `--autoconcat`.

For agent-driven workflows, see [`AGENTS.md`](AGENTS.md) and [`DIRECTOR.md`](DIRECTOR.md).

---

## LoRA (optional)

- **Web UI** — select presets in the LoRA dropdown (downloads to `./loras/` on first use)
- **Server-wide** — `python server.py --enable-lora --lora <url-or-path> 1.0`
- **Env** — `LTX_WS_ENABLE_LORA=1`, `LTX_WS_DEFAULT_LORA`, `VIDEOFENTANYL_LORA_DIR`

Default catalog includes the OmniNFT RL LoRA; choose **None** in the UI to disable for a job.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Missing `ltx_pipelines_mlx` | Re-run the two `ltx-2-mlx` install lines from [Install](#install) |
| Hub / auth errors | `HF_TOKEN` or `huggingface-cli login`; check free disk |
| Out of memory | `--model dgrauet/ltx-2.3-mlx-q4`, lower resolution or `--num-frames`, or `--mlx-low-memory` |
| `autoconcat` failed | `pip install av` (PyAV); fragment files are kept if merge fails |
| Player won't open MP4 | Re-mux with PyAV or any standard MP4 tool |
| Generated audio is hiss / near-silent (T2V/I2V); A2V OK | Known **mlx 0.31.2** Metal vocoder bug. Use `ltx-core-mlx` **≥ v0.14.19** (README pin). Latest `ltx-ws` also applies a runtime workaround. Diagnose: `python scripts/diagnose_audio_mlx.py`. Rebuild any packaged `.app` still on `0.14.9`. |
| ID-LoRA / face swap / keyframe fails to load | Needs **dev** MLX weights (`transformer-dev.safetensors` + distilled LoRA) — use `dgrauet/ltx-2.3-mlx` or `ltx-2.3-mlx-q8`, not distilled-only. |
| ID-LoRA voice doesn't match / ignores reference WAV as soundtrack | Expected: reference audio is **identity context** (~5s tip from upstream), not the final track. Put spoken lines in `[SPEECH]: …` of the prompt. |

---

## Development

**Frontend hot reload** while the server runs:

```bash
python server.py --model dgrauet/ltx-2.3-mlx-q8
cd web && npm run dev   # :5299, proxies API/WS to :8765
```

**Standalone Web UI** (attach to a remote server):

```bash
python web_server.py --server-url ws://127.0.0.1:8765/ws
```

**Benchmark script:**

```bash
./scripts/benchmark_local_generation.py
```

---

## Project layout

```
server.py           WebSocket server + embedded Web UI
videofentanyl.py    CLI client
mcp_server.py       MCP adapter
web_ui.py           Web API & job orchestration
web/                React UI → web/dist/
ltx_mlx_backend.py  MLX generation backend
models/             Local weights (gitignored)
web_outputs/        Generated clips (gitignored)
```

---

## Internals (for integrators)

**Stack:** [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) on Apple Silicon via `ltx_pipelines_mlx`. One generation job at a time; additional clients queue fairly.

**WebSocket flow:** `session_init_v2` → `simple_generate` → binary MP4 chunks → completion events. Optional `generation_keepalive` JSON during long runs. Full tool contracts are documented in [`AGENTS.md`](AGENTS.md).

**Common server flags**

| Flag | Default | Notes |
|------|---------|-------|
| `--host` / `--port` | `0.0.0.0` / `8765` | Web UI + `/ws` on same port |
| `--model` | `auto` | HF repo, local path, or RAM-based pick |
| `--num-frames` | `97` | Snapped to 8k+1 |
| `--height` / `--width` | `480` / `704` | Snapped to multiples of 32 |
| `--infer-steps` | `8` | Distilled pipeline steps |
| `--web-output-dir` | `./web_outputs` | Web UI clip storage |
| `--enable-lora` | off | Global LoRA for all requests |

**MCP tools:** `ltx_server_healthcheck`, `ltx_generate_video`, `ltx_generate_sequence` — see [`AGENTS.md`](AGENTS.md).

---

## Links

- [ltx-ws](https://github.com/lmangani/ltx-ws)
- [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)
- [dgrauet/ltx-2.3-mlx weights](https://huggingface.co/dgrauet/ltx-2.3-mlx)
