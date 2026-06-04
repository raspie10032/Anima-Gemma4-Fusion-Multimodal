# GemmAnima Engine

[Korean README](README.ko.md) | [Hugging Face adapter bundle](https://huggingface.co/raspie/gemmanima-adapter-bundle)

GemmAnima is a prototype local-first chat and image orchestration app that
connects three parts:

- **Gemma Core** for resident chat, routing, intent planning, and Gemma vision tagging.
- **Anima Image Core** for local image generation.
- **HiddenStage Bridge** for mapping Gemma-side planning into Anima-side
  conditioning.

The v0.1.0 release is a public prototype. It is meant to be tested, inspected,
and improved. It is not a production model release and it is not safety-rated.

## What This Repository Contains

This repository contains the standalone app source, runtime configs, schemas,
tests, and prototype training/evaluation utilities.

It does **not** store large model weights. On first run, the app is expected to
download required assets from their original model pages or from the GemmAnima
prototype adapter bundle on Hugging Face.

Normal runtime is local. Chat, tag routing, image-request planning,
bridge-profile selection, sampler/scheduler selection, and render orchestration
do not depend on external AI APIs once the model assets are present.

## Current Prototype Features

- Local web GUI for chat and image requests.
- Resident Gemma runtime design for fast chat once initialized.
- Strong language harnessing for Korean or English chat.
- English tag contract for tag requests, even when the user speaks Korean.
- Automatic routing between normal chat, image generation, and image tagging.
- Drag-and-drop or attach-image input in the chat UI.
- Chat-to-image generation with visible pending/loading state.
- Preset generation controls for resolution, sampler, scheduler, steps, CFG,
  seed, and Anima LoRA usage.
- Local-worker, in-process, external-script, and dry-run renderer modes.
- First-run model asset plan and download-progress APIs.
- Run manifests for generated jobs.
- Unit tests for routing, contracts, model downloads, manifests, render plans,
  and the browser GUI surface.

## Model Layout

GemmAnima is split into named model groups so users do not need duplicate
copies of the same base model.

| Group | Purpose |
| --- | --- |
| Gemma Core | Base GGUF plus LoRA/mmproj adapters for chat, planning, and Gemma vision tagging. |
| Anima Image Core | Diffusion model and VAE used by the local image renderer. |
| HiddenStage Bridge | Prototype bridge/profile checkpoints used to translate Gemma-side intent into Anima conditioning. |

The expected v0.1.0 asset sources are:

| Asset group | Source policy |
| --- | --- |
| Gemma base GGUF | Download from the original GGUF model page. |
| Anima diffusion and VAE | Download from the original Anima model page. |
| GemmAnima adapters and bridge profiles | Download from the GemmAnima adapter bundle. |

The configured sources live in:

- `configs/models.yaml`
- `RTD/configs/models.yaml`
- `RTD/configs/model_sources.json`
- `RTD/asset_manifest.json`

## Reference Index

This GitHub repository is the app/source surface, not a model mirror. Use these
references to audit where each runtime component, model asset, and design idea
comes from.

### Model And Asset Sources

| Component | Reference | Used for |
| --- | --- | --- |
| GemmAnima app source | [raspie10032/Anima-Gemma4-Fusion-Multimodal](https://github.com/raspie10032/Anima-Gemma4-Fusion-Multimodal) | Standalone app source, launchers, schemas, tests, and runtime orchestration. |
| GemmAnima adapter bundle | [raspie/gemmanima-adapter-bundle](https://huggingface.co/raspie/gemmanima-adapter-bundle) | Gemma task adapters, prototype vision projector, HiddenStage bridge checkpoints, metadata, and model card. |
| Gemma Core base GGUF | [mradermacher/gemma-4-E2B-it-heretic-ara-custom-GGUF](https://huggingface.co/mradermacher/gemma-4-E2B-it-heretic-ara-custom-GGUF) | Shared local GGUF loaded once for chat, planning, and Gemma-side adapters. |
| Anima Image Core | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | Anima diffusion model and VAE assets used by the local image renderer. |

### Runtime And Implementation References

| Reference | Used for |
| --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF runtime conventions, local chat inference, LoRA attachment, and multimodal projector execution model. |
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | Optional in-process Python binding path for resident Gemma runtime experiments. |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Sampler/scheduler naming, compatibility renderer concepts, and local diffusion-runtime expectations. |
| [chopratejas/headroom](https://github.com/chopratejas/headroom) | Design reference for long-chat context compression. GemmAnima embeds its own minimal Headroom-style compressor and does not vendor or require the package. |
| [Danbooru tag groups](https://danbooru.donmai.us/wiki_pages/tag_groups) | Human-readable reference for canonical English Danbooru tag vocabulary used by tag prompts and tagger output contracts. |
| [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | License reference cited by the Anima upstream page where NVIDIA Cosmos-derived terms apply. |

### Project Documentation References

| Path | Reference role |
| --- | --- |
| `RTD/configs/model_sources.json` | Machine-readable source, filename, and license-id map for first-run downloads. |
| `RTD/asset_manifest.json` | Portable RTD asset manifest and expected local model layout. |
| `RTD/LICENSE_NOTICES.md` | Consolidated source/license notices for the composite prototype. |
| `RTD/HF_MODEL_CARD.md` | Hugging Face-facing adapter bundle model-card source. |
| `RTD/HF_MODEL_CARD.ko.md` | Korean Hugging Face-facing adapter bundle model-card source. |
| `docs/architecture_summary.md` | High-level app architecture and data flow. |
| `docs/verification_plan.md` | Test and verification surfaces for release checks. |

Use the CLI to inspect the exact download plan for your environment:

```powershell
python -m gemmanima.cli model-download-plan --json
```

## Installation

The v0.1.0 prototype is distributed as a source checkout. Model weights are not
included in GitHub. Install the Python package first, then let the app download
or locate model assets.

`GemmAnima.bat` is the only user-facing Windows launcher. It provides small
subcommands for GUI startup, health checks, model downloads, dry-run rendering,
image tagging, and tests so first-run users do not need to choose between
multiple executable files.

### 1. Requirements

- Python 3.10 or newer
- Windows or Linux with a CUDA-capable NVIDIA GPU for real local generation
- `git`
- Enough disk space for the base models and adapter bundle
- A Python/CUDA environment capable of running the selected local renderer

The source package itself is intentionally light. Real local chat, vision
tagging, and image generation require the runtime libraries and model assets
reported by the health check.

GemmAnima is a source checkout, so the Windows launcher owns a local `.venv`.
The first run is visible: `GemmAnima.bat` creates or reuses `.venv`, installs
the checkout in editable mode, and then reports which runtime engines are
available. After bootstrap, GUI startup does not perform hidden dependency
installation.

Runtime engines such as `llama-cpp-python`, PyTorch, Pillow, NumPy, and
safetensors are CUDA/runtime choices. The source bootstrap creates `.venv` with
`--system-site-packages` so an existing machine-level CUDA Python stack remains
visible instead of being replaced by surprise downloads. Use:

```powershell
GemmAnima.bat health
```

to inspect what the current source environment can run. Network access in the
app is reserved for visible first-run model asset downloads.

### 2. Clone

```powershell
git clone https://github.com/raspie10032/Anima-Gemma4-Fusion-Multimodal.git
cd Anima-Gemma4-Fusion-Multimodal
```

### 3. Bootstrap The Source Environment

Windows PowerShell:

```powershell
.\GemmAnima.bat bootstrap
```

The launcher uses `.venv\Scripts\python.exe` for all Windows commands after
bootstrap. If `.venv` is missing, `GemmAnima.bat` will run the same visible
bootstrap path before starting the requested command.

Linux/macOS shell, for developers running without the Windows launcher:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

### 4. Choose A Model Asset Folder

By default, first-run model assets are stored under the user's local app data
folder. To choose another location, set `GEMMANIMA_MODEL_ROOT` before running
the download commands or GUI.

Windows PowerShell:

```powershell
$env:GEMMANIMA_MODEL_ROOT = "$HOME\gemmanima-models"
```

Linux/macOS shell:

```bash
export GEMMANIMA_MODEL_ROOT="$HOME/gemmanima-models"
```

### 5. Inspect And Download Assets

Print the configured asset plan:

```powershell
python -m gemmanima.cli model-download-plan --json
```

Download or verify required assets:

```powershell
python -m gemmanima.cli ensure-model-assets --json
```

The GUI also exposes model download progress in its settings panel. GitHub does
not mirror the base model weights; the downloader uses configured upstream
sources and the GemmAnima adapter bundle.

### 6. Run Health And Tests

On Windows, use the single launcher for first checks:

```bat
GemmAnima.bat health
GemmAnima.bat test
```

The direct Python commands are:

```powershell
python -m gemmanima.cli model-download-plan --json
python -m pytest -q
```

For a fresh machine, expect real renderer health to remain blocked until all
external model/runtime dependencies are present.

## Quick Start

After installation and asset setup, launch the local GUI:

Windows launcher:

```bat
GemmAnima.bat
```

Direct Python command:

```powershell
python -m gemmanima.cli gui-command
```

Open the printed URL, usually:

```text
http://127.0.0.1:8765
```

Smoke-test a dry-run generation without invoking the real renderer:

Windows launcher:

```bat
GemmAnima.bat dry-run "draw a bright forest"
```

Direct Python command:

```powershell
python -m gemmanima.cli run "draw a bright forest" --renderer dry-run --json
```

## Running Tests

```powershell
python -m pytest -q
```

The v0.1.0 source snapshot was checked with the full test suite before release.

## API Surface

The local GUI talks to the local HTTP backend:

- `GET /v1/health`
- `POST /v1/chat`
- `POST /v1/uploads`

Useful payload examples are in:

- `RTD/payloads/chat_general.json`
- `RTD/payloads/chat_image_generation.json`
- `RTD/payloads/tag_image_template.json`

## Generation Presets

The GUI exposes prototype-friendly controls rather than requiring users to edit
JSON by hand.

Resolution presets:

- `1024 x 1024`
- `832 x 1216`
- `768 x 1344`
- custom width and height

Supported sampler choices:

- preset default
- `euler`
- `euler_ancestral`
- `dpmpp_2m`
- `dpmpp_2m_sde_gpu`

Supported scheduler choices:

- preset default
- `normal`
- `karras`
- `sgm_uniform`

Renderer choices:

| Renderer | Meaning |
| --- | --- |
| `local-worker` | Preferred standalone local worker path. |
| `in-process` | Repo-native in-process renderer path where the environment supports it. |
| `dry-run` | Writes a dry-run artifact without running real image generation. |
| `external-script` | Compatibility bridge for older local render scripts. |

## Documentation Map

| Path | Purpose |
| --- | --- |
| `RTD/README.md` | Runtime desk for local setup files, payloads, and scripts. |
| `RTD/HF_MODEL_CARD.md` | Hugging Face model-card source for the adapter bundle. |
| `RTD/LICENSE_NOTICES.md` | License and source notices for model assets. |
| `docs/architecture_summary.md` | App architecture overview. |
| `docs/verification_plan.md` | Verification checklist and test surfaces. |
| `docs/version_roadmap.md` | Prototype roadmap. |
| `schemas/` | JSON schemas for protocol and run manifests. |
| `tests/` | Unit tests for core behavior. |

## License And Model Notices

GemmAnima is a composite prototype built around Gemma and Anima base-model
families. The repository source, adapter bundle, bridge checkpoints, downloaded
base weights, and generated runtime stack are not covered by one simple
permissive license.

Use the most restrictive applicable terms across the loaded stack:

- Gemma/GGUF base model terms and required redistribution notices.
- Gemma-dependent routing, harness, prompt-contract, hidden-state bridge, and
  adapter logic where it relies on Gemma or Gemma-derived behavior.
- Anima base model terms, currently presented upstream as the CircleStone Labs
  Non-Commercial License.
- NVIDIA Open Model License terms referenced by the Anima upstream page where
  applicable.
- GemmAnima adapter/checkpoint notices.
- Dataset/source restrictions for any adapter training data.

Because the current image path depends on Anima, treat the v0.1.0 prototype and
its adapter/checkpoint bundle as **non-commercial, non-production,
restricted-use prototype material** unless a separate license review and any
required upstream permissions say otherwise.

Before redistributing model files or using the adapter bundle outside local
testing, read:

- `LICENSE.md`
- `RTD/LICENSE_NOTICES.md`
- the original Gemma/GGUF model page
- the original Anima model page
- the GemmAnima adapter bundle model card

Do not describe the v0.1.0 prototype as production-ready, commercially ready,
promoted, or safety-evaluated.

## Prototype Limitations

- Image quality is still bridge/conditioning limited.
- Pose robustness needs more targeted data and evaluation.
- Real generation requires local GPU/runtime setup.
- First-run model download behavior is still part of the prototype surface.
- External-script rendering remains a compatibility bridge, not the intended
  standalone default.
- The adapter bundle is public prototype material, not a final promoted model.

## Development Notes

Keep GitHub and Hugging Face documentation separate:

- GitHub explains the standalone app, source tree, runtime configs, and tests.
- Hugging Face explains the adapter/checkpoint bundle and upstream model asset
  sources.

Do not commit generated images, run manifests, caches, downloaded models,
checkpoints, or private local paths.
