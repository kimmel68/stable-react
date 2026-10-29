# Stable React — the music is a grammar, not a volume meter

**Stable React** (code name `dmk`) turns a photo or a video plus a music track (or just a BPM) into a Deforum-style journey that **glides through
the picture without a single cut** and **plays the bar**: every count of a 4/4 bar does something different, and on
count 1 the whole world changes — coast → arctic → lava → marble — while space, shapes and the flow of motion stay
intact. Everything runs locally through ComfyUI and the **Comfy SDK**.

Two apps, one code base, the same settings:

| App | Start | What draws the worlds | Category |
|---|---|---|---|
| **Stable React · SD 1.5** | `app.bat` → http://127.0.0.1:7860 | Stable Diffusion 1.5 in a depth-guided feedback loop | Open weights only |
| **Stable React · Flux Hybrid** | `app_flux.bat` → http://127.0.0.1:7861 | **FLUX.1 Kontext [dev]** redraws the footage on the beat, the SD 1.5 loop makes it flow | Built with FLUX |

> Demo video (≤ 2 min): *link follows*

## What is new

- **A 4/4 beat grammar.** Not "pulse on every beat": count 1 = zoom push and world change, count 2 = blur that snaps
  back to sharp in the next frame, count 3 = the source pulls back in, count 4 = a transformation impulse. Every
  effect is a curve (track → features per frame → bindings → per-frame parameters) and can be previewed before
  rendering.
- **On the hit, not near it.** Beats are refined to the actual onset (±40 ms window, 3 ms resolution) and mapped to
  the first frame that starts with the sound: the picture changes 7–13 ms after the onset, never before it. A sample
  is forced onto every beat frame (78 % of the transformation lands in the beat frame, 100 % in the Flux Hybrid).
- **World change on count 1.** Every 1, 2 or 4 bars the prompt jumps to the next world; on that frame the loop redraws
  harder and releases its memory, so the new world stands at once instead of fading in.
- **Endless glide, never a cut.** The camera zooms and drifts through the picture along a path; *Follow the depth*
  steers towards the farthest, structured, bright region (no flying into walls or sky). When the camera has used up
  the source pixels, its own generated frame becomes the new source (anchor + new depth estimate) — the journey
  continues as long as the music does, without a restart or jump.
- **Flow instead of boiling.** Cadence 3 with an optical-flow morph, motion-compensated smoothing between beats, a
  moving memory image and the **DMK Reference Attention** node (the model "sees" its warped previous output):
  boiling between beats dropped from 14.9 to 3.3 while the hit got stronger (measured, see `docs/`).
- **A look that is not "AI".** Photographic prompts, low CFG, colours from the footage and an output-only film finish
  (halation, highlight shoulder, vignette, grain).
- **BPM only.** No track at hand? Enter a tempo: dmk synthesizes a click track (kick on count 1) and the same
  analysis runs on it — the grid is exact. Keep the click in the result or render silent and add music later.
- **Flux Hybrid.** FLUX.1 Kontext gets an edit instruction per beat ("change everything in this image to *world*,
  keeping the same camera view, framing and outline of every shape"); optical flow carries each keyframe to the next
  beat; then the dmk loop runs over this Flux video with less redraw and more pull towards it. Flux brings precise
  content, the loop brings the swelling Deforum flow. One schedule drives both stages, so every world change sits on
  the same frame in Flux and in the loop.
- **Reusable as nodes.** The beat grammar is also a ComfyUI node pack, model-agnostic (SD, Flux, video models).

## The node pack `comfyui_dmk` (GPL-3.0)

| Node | Does |
|---|---|
| DMK Audio to Beats (4/4) | track → beats on the onset, bar counts 1–4, features per frame |
| DMK Beat Grammar | bindings (JSON) → per-frame curves for zoom, denoise, blur, source pull, memory, cfg … |
| DMK Curve to Schedule | curve → keyframe schedule `0:(v), 1:(v), …` (Deforum / FizzNodes style) |
| DMK Prompts on the Bar | prompt list → prompt schedule that switches on every n-th count 1 |
| DMK Plot Beat Curves | preview image of the curves with the bar counts |
| DMK Reference Attention | SD 1.5 attends to a reference latent (the warped previous frame) |

Example workflow: `comfyui_dmk/example_workflows/dmk_beat_grammar.json` (beats identical to the app's).
Flux keyframe workflow (core nodes only): `deforum_mosh/dmk/workflows/flux_kontext_keyframe_api.json`.

## Using the apps

1. **Start services** (launches ComfyUI and the Comfy API proxy; the status line checks versions and models).
2. **Source:** an image or a video (drone footage and landscapes work best; the SD app also takes Blender passes).
3. **Beat source:** *Track* (any audio file, choose the start second) or *BPM only*.
4. **Worlds:** pick a world set (Coast, Mountain lake, Forest seasons, North sea, Desert, Clouds, Fields,
   Architecture, Architecture II) — it fills an editable list, one prompt per line — and how often the world changes
   on count 1.
5. **Render.** *Preview curves* shows how the music maps onto every parameter before you spend GPU time.

| Setting | Default | Notes |
|---|---|---|
| Length | 10 s | 2–60 s; the endless glide has no length limit by design |
| World change on count 1 | every 2 bars | off / 1 / 2 / 4 bars |
| Mode | Camera move | *Rigid frame*: transformation in place, no camera move |
| Glide speed | 2 (image) / 0 (video) | 0 = zoom pulses only, 5 = wild; a video brings its own motion |
| Camera path | Follow the depth | Direction (8 directions + sideways drift), Sweep, Straight in |
| Flux draws a new image … | every beat | Flux app only: every bar or world change only ≈ 4× faster |
| Look | film colour, film finish 1.0, palette auto | black and white; palette from the footage or per world |
| Resolution | 512 | 640 / 768 (showcase clips: 768) |
| Advanced | | beat recipe, strength of counts 1–4, snap to a fixed grid, picture delay (e.g. Bluetooth), zoom pulse strength, vanishing point, mask |

## Hardware and time

Tested on Windows 11, RTX 3070 Ti (8 GB VRAM), 32 GB RAM.

| | 10 s at 512 px | 15 s at 768 px (measured) |
|---|---|---|
| SD 1.5 app | ≈ 5 min | 12–17 min |
| Flux Hybrid, every beat | ≈ 15 min | 21–32 min (≈ 35 s per Flux image) |

## Setup (Windows, NVIDIA GPU with 8 GB)

Requirements: git, Python 3.12 (or [uv](https://docs.astral.sh/uv)), ffmpeg in PATH (`winget install Gyan.FFmpeg`).

```bash
powershell -ExecutionPolicy Bypass -File install.ps1
```

Add `-Flux` for the Flux Hybrid models (about 19 GB). The installer creates `.venv`, installs PyTorch with CUDA
(`-TorchIndex` for another CUDA build), clones ComfyUI **v0.39.1** into `ComfyUI/`, installs the requirements, copies
the node pack into `ComfyUI/custom_nodes` and downloads the models from Hugging Face (`-SkipModels` if you have them).
Check the install without a GPU in about a minute: `.venv\Scripts\python tests\smoke.py`.

### Comfy Developer Platform, self-hosted

All generation runs through the official **Comfy SDK** against the **Comfy API v2**; locally the v2 API is served by
`comfy-api-proxy` in front of ComfyUI (https://docs.comfy.org/development/api-development/sdks). *Start services*
launches both (ComfyUI without a console window, log in `out/logs/`). Pointing the SDK at another deployment uses
the same code path.

## Models

| Model | Role | License |
|---|---|---|
| Stable Diffusion 1.5 — DreamShaper 8 | image model | CreativeML OpenRAIL-M |
| ControlNet v1.1 depth (`control_v11f1p_sd15_depth`) | keeps the space | CreativeML OpenRAIL-M |
| Depth-Anything-V2-Small | depth for photos, video and anchors | Apache-2.0 |
| RAFT (torchvision) | optical flow for video | BSD-3-Clause |
| FLUX.1 Kontext [dev] fp8 (Comfy-Org), T5-XXL fp8, CLIP-L, Flux VAE | Flux Hybrid only | **FLUX.1 [dev] Non-Commercial License** (Kontext, VAE); Apache-2.0 / MIT (encoders) |

ComfyUI core nodes plus the dmk node pack; no partner nodes, no closed APIs. Model weights are not part of this
repository and keep their own licenses (including the CreativeML OpenRAIL-M use restrictions). The SD 1.5 app does
not depend on any Flux weight. Read the FLUX.1 [dev] license before you use Flux Hybrid results.

## How it works

```
track or BPM ──► beats on the onset, bar counts 1–4, features ──► bindings ──► per-frame parameters + world schedule
                                                                                     │
source (image | video | passes) ──► motion (virtual camera + path | RAFT + depth | Blender vectors)
   │                                                                                 │
   └─ Flux Hybrid only: FLUX.1 Kontext keyframes on the beat ──► carried along the video flow ──► new source
                                                                                     │
previous output ──advect──► blend with source ──► img2img SD 1.5 + depth ControlNet + reference attention
       ▲                                              (Comfy SDK → comfy-api-proxy → ComfyUI)
       └── memory ◄── sharpness hold ◄── motion-compensated smoothing ◄── anchors for the endless glide ◄──┘
cadence 3 with optical-flow morph in between, forced sample on every beat, film finish on the output only
```

Driver: `deforum_mosh/dmk/method_a.py`; audio: `audio.py`; Flux stage: `hybrid.py`; app: `app/dmk_app.py`.

## Repository

| Path | Content |
|---|---|
| `app/` | Gradio app (both engines) |
| `comfyui_dmk/` | ComfyUI node pack, GPL-3.0 |
| `deforum_mosh/dmk/` | driver, audio analysis, Flux stage, motion, metrics, Blender pass reader, research scripts |
| `configs/` | beat recipes (`recipe_*.json`), world sets (`weltensets.json`), worlds (`welten.json`) |
| `examples/` | neutral example material (generated with the open model from a procedural corridor) |
| `tests/` | smoke tests (no GPU), node tests |
| `docs/` | German working notes: measurements, night runs, decisions |

## Credits and license

Code: MIT, © 2026 Kimmel (see `LICENSE`), except `comfyui_dmk/`, which is GPL-3.0 because it runs inside ComfyUI.
Music in the examples and the demo: by the author. Drone footage in the demo video: [Pexels](https://www.pexels.com)
(Pexels license; sources listed in the video description) — not part of this repository. Architecture in the demo:
the author's own Unreal Engine renders.
