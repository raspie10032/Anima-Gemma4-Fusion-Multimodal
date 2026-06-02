# GemmAnima Engine

GemmAnima is a prototype local-first chat and image orchestration app that
connects three parts:

- **Gemma Core** for resident chat, routing, tag, and intent planning.
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
| Gemma Core | Base GGUF plus LoRA/mmproj adapters for chat, planning, and vision tagging. |
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

Use the CLI to inspect the exact download plan for your environment:

```powershell
python -m gemmanima.cli model-download-plan --json
```

## Quick Start

Requirements:

- Python 3.10 or newer
- Windows or Linux with a CUDA-capable NVIDIA GPU for real local generation
- `git`
- Enough disk space for the base models and adapter bundle

Clone the repository:

```powershell
git clone https://github.com/raspie10032/Anima-Gemma4-Fusion-Multimodal.git
cd Anima-Gemma4-Fusion-Multimodal
```

Create an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

Inspect and download required model assets:

```powershell
python -m gemmanima.cli model-download-plan --json
python -m gemmanima.cli ensure-model-assets --json
```

Launch the GUI:

```powershell
python -m gemmanima.cli gui-command
```

Open the printed URL, usually:

```text
http://127.0.0.1:8765
```

If you want to store model assets somewhere other than the default app data
location, set:

```powershell
$env:GEMMANIMA_MODEL_ROOT = "$HOME\gemmanima-models"
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
| `external-script` | Compatibility fallback for older local render scripts. |

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

This repository is source code plus configuration. Model assets are governed by
their own upstream licenses and terms.

Before redistributing model files or using the adapter bundle outside local
testing, read:

- `RTD/LICENSE_NOTICES.md`
- the original Gemma/GGUF model page
- the original Anima model page
- the GemmAnima adapter bundle model card

The v0.1.0 prototype should not be described as production-ready, commercially
ready, promoted, or safety-evaluated.

## Prototype Limitations

- Image quality is still bridge/conditioning limited.
- Pose robustness needs more targeted data and evaluation.
- Real generation requires local GPU/runtime setup.
- First-run model download behavior is still part of the prototype surface.
- External-script rendering remains a compatibility fallback, not the intended
  standalone default.
- The adapter bundle is public prototype material, not a final promoted model.

## Development Notes

Keep GitHub and Hugging Face documentation separate:

- GitHub explains the standalone app, source tree, runtime configs, and tests.
- Hugging Face explains the adapter/checkpoint bundle and upstream model asset
  sources.

Do not commit generated images, run manifests, caches, downloaded models,
checkpoints, or private local paths.
