# GemmAnima Engine

GemmAnima is a local Windows backend for routing chat, tag, and image-generation
requests through a Gemma/TIPO planning layer and an Anima hidden-stage renderer.

The current repo supports:

1. Normal chat without touching image components.
2. Strong language and output-contract harnesses for Gemma text chat.
3. TIPO vision tagging with canonical English Danbooru tags.
4. Chat-driven image generation through a parsed JSON contract.
5. Dry-run, local-worker, and in-process renderer paths.
6. Health/preflight reporting for missing local assets.
7. Manifest recording for generated jobs.

## Documentation Surfaces

This GitHub README is for the standalone app and source tree: local setup,
runtime behavior, tests, development notes, and first-run asset wiring.

The Hugging Face model card is separate and should describe only the uploaded
prototype adapter/checkpoint bundle. Its source lives at
`RTD\HF_MODEL_CARD.md`, and the upload staging copy lives at
`runs\hf_upload\gemmanima-adapter-bundle-v0.1\README.md`.

The boundary is intentional: GitHub explains how to run the app; Hugging Face
explains what adapter files were published and what upstream base weights must
be downloaded from original model pages.

## Standalone Runtime Contract

GemmAnima is intended to run as a local standalone app. Chat, tag routing, image
request planning, bridge-profile selection, sampler/scheduler selection, and
render orchestration live in this repository and do not call external AI APIs.

Network access is only a bootstrap path for downloading model assets from their
original pages or from the Hugging Face prototype adapter bundle. Once assets
are present locally, normal app runtime should stay local.

The GUI exposes a model-asset download panel with progress bars for first-run
downloads. Users should be able to see the current asset, current byte progress,
and overall asset count while adapters and base models are fetched.

The preferred image path is the repo-native `local-worker` / `in-process`
renderer stack. `external-script` remains a legacy compatibility backend and is
not the GitHub RTD default.

The default target GPU for local inference/rendering is the RTX 4070 Ti SUPER.
Keep the RTX 5060 out of PyTorch cache/training paths unless the installed
PyTorch build explicitly supports it.

## Quick Start

From the repo root:

```powershell
python -m gemmanima.cli model-download-plan --json
python -m gemmanima.cli ensure-model-assets --json
python -m pytest -q
python -m gemmanima.cli gui-command
```

Open the printed URL, usually:

```text
http://127.0.0.1:8765
```

The local GUI talks to:

- `GET /v1/health`
- `POST /v1/chat`

The GUI can use `dry-run` for smoke tests. For real generation, use the local
worker or in-process renderer. Use `external-script` only as a legacy
compatibility fallback.

On a fresh GitHub checkout, model files are not stored in the repository. The
first-run download path defaults to:

```text
%LOCALAPPDATA%\GemmAnima\models
```

Set `GEMMANIMA_MODEL_ROOT` to override that location. Existing local development
paths are reused when those files already exist.

The same download path is available in the GUI settings panel under
`모델 자산`, where progress bars show the overall queue and current file.

## Python Dependencies

The project metadata is intentionally light:

- Python `>=3.10`
- `pytest` for the test suite

The full local renderer and training stack also depends on the environment that
already exists on this workstation:

- Windows + PowerShell
- NVIDIA driver/CUDA runtime
- CUDA-enabled `llama-cpp-python` for resident Gemma chat startup:
  `python -m pip install --upgrade --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --prefer-binary`
- PyTorch with CUDA support for the RTX 4070 Ti SUPER
- ComfyUI embedded Python at `E:\ComfyUI_sage\python_embeded\python.exe`
- ComfyUI checkout at `E:\ComfyUI_anima_exp`
- Anima/Gemma swap project at `E:\anima_gemma_swap`
- llama.cpp CUDA binaries under `D:\Projects\training\llama_b9209_cuda` for
  vision tagging and CLI fallback paths

The app initializes the Gemma runtime at backend startup. With the default
in-process backend, the base GGUF stays resident for the life of the Python
server process. Text and vision calls attach the needed LoRA adapters, and
vision tagging also attaches the mmproj handler against the same resident base
model. If `llama-cpp-python` reports no GPU offload support while GPU layers are
requested, startup health reports the runtime as failed instead of silently
falling back to CPU.

Use health commands below before assuming a renderer path is available.

## Required Local Model Parts

The standalone runtime is split into three named model parts. Anima text encoder
weights are not part of the required runtime; the in-process path only uses the
Anima tokenizer format for `t5xxl_ids` / `t5xxl_weights` metadata.

### Gemma Core

| Purpose | Path |
| --- | --- |
| Shared base GGUF | `D:\Projects\training\out\gemma-4-E2B-it-heretic-ara-Q4_K_M.gguf` |
| Text/chat LoRA | `D:\Projects\training\out\lora\adapter_model.f16.gguf` |
| Vision/tagger LoRA | `D:\Projects\training\out\gemmanima_v4_vision_tagger\adapter_model.f16.gguf` |
| Vision mmproj | `D:\Projects\training\out\_completed\gemma4-tipo-vision.mmproj-f16.gguf` |

### Anima Image Core

| Purpose | Path |
| --- | --- |
| Diffusion model | `E:\ComfyUI_sage\ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors` |
| VAE | `E:\ComfyUI_sage\ComfyUI\models\vae\qwen_image_vae.safetensors` |

### HiddenStage Bridge

| Purpose | Path |
| --- | --- |
| Planner LoRA adapter | `D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2\adapter_model.safetensors` |
| Planner vision embedding | `D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2\embed_vision.pt` |
| Bridge checkpoint | `E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt` |

First-run source policy:

| Asset group | Source |
| --- | --- |
| Gemma base GGUF | Original GGUF page: `mradermacher/gemma-4-E2B-it-heretic-ara-custom-GGUF`; upstream metadata currently reports `apache-2.0` |
| Anima diffusion + VAE | Original Anima page: `circlestone-labs/Anima`; upstream page currently reports CircleStone Labs Non-Commercial License and references NVIDIA Open Model License terms for derivative use |
| GemmAnima LoRA/mmproj/bridge files | GemmAnima adapter bundle repo, configurable with `GEMMANIMA_ADAPTER_REPO` |

Use this to inspect exact URLs and target paths:

```powershell
python -m gemmanima.cli model-download-plan --json
```

See `RTD\LICENSE_NOTICES.md` before publishing or using the bundle outside local
testing.

Current bridge status:

- Checkpoint: `E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt`
- Validation MSE: `0.001104317136865575`
- Gate: passed

## TIPO / Gemma Chat Assets

`gemmanima.modules.tipo_runtime` uses llama.cpp executables directly.

| Purpose | Path |
| --- | --- |
| Text chat CLI | `D:\Projects\training\llama_b9209_cuda\llama-cli.exe` |
| Vision tag CLI | `D:\Projects\training\llama_b9209_cuda\llama-mtmd-cli.exe` |
| Shared base GGUF | `D:\Projects\training\out\gemma-4-E2B-it-heretic-ara-Q4_K_M.gguf` |
| Text/chat LoRA | `D:\Projects\training\out\lora\adapter_model.f16.gguf` |
| Vision/tagger LoRA | `D:\Projects\training\out\gemmanima_v4_vision_tagger\adapter_model.f16.gguf` |
| Vision mmproj | `D:\Projects\training\out\_completed\gemma4-tipo-vision.mmproj-f16.gguf` |

The text and vision paths now share the same base GGUF and load task behavior
through llama.cpp `--lora`. The older merged GGUFs
`D:\Projects\training\out\gemma4-tipo-ko-v2-Q4_K_M.gguf` and
`D:\Projects\training\out\_completed\gemma4-tipo-vision-Q4_K_M.gguf` are kept as
compatibility/reference artifacts, not the preferred runtime shape.

Default runtime settings:

- `CUDA_VISIBLE_DEVICES=0`
- llama.cpp device: `CUDA0`
- chat template: `gemma`
- text max tokens: `256`
- vision tag max tokens: `140`

Chat modes accepted by the API:

- `general_chat`
- `tag_request`
- `image_generation_request`
- `status_question`
- `file_checkpoint_question`

Language values accepted by the API:

- `ko`
- `en`

For tag requests, the chat harness still requires canonical English Danbooru
tags. Do not translate tag tokens into Korean.

## Renderer Paths

Renderer selection happens through the API payload or GUI.

| Renderer | Meaning |
| --- | --- |
| `dry-run` | Writes a dry-run output artifact without executing real image generation. |
| `local-worker` | Starts the repo-owned local render worker with app-controlled orchestration. |
| `in-process` | Uses repo-native Comfy/Anima modules where available. |
| `real` | Alias path for the in-process renderer in API setup. |
| `external-script` | Legacy compatibility fallback around the old Anima script; not the RTD default. |

Legacy external-script defaults:

- Embedded Python: `E:\ComfyUI_sage\python_embeded\python.exe`
- Script: `E:\anima_gemma_swap\scripts\core\18_hiddenstage_chat_generate.py`
- Reference smoke output:
  `runs\images\nahida_hiddenstage_bridge_real_smoke.png`

In-process bootstrap defaults:

- Comfy root: `E:\ComfyUI_anima_exp`
- Models root: `E:\ComfyUI_sage\ComfyUI\models`
- Project root: `E:\anima_gemma_swap`
- Gemma HF snapshot:
  `C:\Users\seine\.cache\huggingface\hub\models--p-e-w--gemma-4-E2B-it-heretic-ara\snapshots\c9a1d4c031981f14d86eeb0c7d87de7fafd34513`

Renderer profiles are defined in `configs\renderer_profiles.yaml`:

- `anima_fp16_final`: 1024x1024, 28 steps, CFG 4.0
- `anima_int8_draft`: 768x768, 16 steps, CFG 3.5

Chat-to-image requests also apply a generation preset before rendering. The
model supplies the prompt JSON, while the app supplies stable generation
conditions.

Resolution presets:

- `square_1024`: 1024x1024
- `portrait_832_1216`: 832x1216, or 1216x832 with `orientation=landscape`
- `portrait_768_1344`: 768x1344, or 1344x768 with `orientation=landscape`
- `custom`: uses `custom_width` and `custom_height`

Generation presets:

- `anima_draft`: 16 steps, CFG 3.5, `euler_ancestral` / `sgm_uniform`
- `anima_balanced`: 28 steps, CFG 4.0, `euler_ancestral` / `sgm_uniform`
- `anima_final`: 36 steps, CFG 4.0, `euler_ancestral` / `sgm_uniform`
- `anima_lora`: balanced defaults with `lora_stack=["anima_lora"]`

Sampler and scheduler option names are curated from ComfyUI `KSampler`, but the
standalone app only exposes the subset that this checkout smoke-tests and
supports:

- Samplers: `euler`, `euler_ancestral`, `dpmpp_2m`, `dpmpp_2m_sde_gpu`
- Schedulers: `normal`, `karras`, `sgm_uniform`

The API exposes this supported subset from `GET /v1/health` as `samplers` and
`schedulers`. The larger ComfyUI list is kept in code as a reference snapshot,
not as the app contract.

## Health and Preflight

Use these commands before running real generation:

```powershell
python -m gemmanima.cli training-readiness --json
python -m gemmanima.cli renderer-backends --json
python -m gemmanima.cli real-render-health --json
python -m gemmanima.cli gui-command --json
```

The HTTP health endpoint also reports structured preflight data:

```powershell
python -m gemmanima.cli gui-command
# then GET http://127.0.0.1:8765/v1/health
```

Health responses preserve the existing asset-specific sections and also expose:

- `ready`
- `preflight.ready`
- `preflight.blocking`
- `preflight.issues[]`

Each issue includes:

- `code`
- `scope`
- `asset`
- `path`
- `severity`
- `message_ko`
- `message_en`

## Chat to Image Flow

For a chat-driven generation request:

1. The API receives `task="chat"` and `chat_mode="image_generation_request"`.
2. TIPO text chat is forced to return the `image_generation_json` contract.
3. The JSON is parsed into a `GenerationPlan`.
4. The conductor executes that plan directly instead of re-planning the prompt.
5. The selected renderer writes the output and manifest.

If the model returns prose or malformed JSON in this mode, rendering is blocked
with:

```json
{
  "status": "failed",
  "error_code": "image_generation_contract_failed"
}
```

## Useful Commands

```powershell
python -m gemmanima.cli training-readiness --json
python -m gemmanima.cli prepare-teacher-targets --json
python -m gemmanima.cli prepare-gemma-cache --json
python -m gemmanima.cli prepare-bridge-training --json
python -m gemmanima.cli bridge-eval-status --json
python -m gemmanima.cli bridge-smoke-command --json
python -m gemmanima.cli gemma-hidden-smoke-command --json
python -m gemmanima.cli t5-tokenizer-smoke-command --json
python -m gemmanima.cli renderer-backends --json
python -m gemmanima.cli real-render-health --json
python -m gemmanima.cli real-render-command --json
python -m gemmanima.cli gui-command --json
```

Dry-run request:

```powershell
python -m gemmanima.cli run "draw a bright forest" --json
```

External-script render:

```powershell
python -m gemmanima.cli run "draw a bright forest" --renderer external-script --json
```

In-process smoke command:

```powershell
python -m gemmanima.cli in-process-render-smoke-command --json
```

## Reference Documents

- `docs\training_pipeline.md` - bridge/training pipeline notes
- `docs\architecture_summary.md` - architecture overview
- `docs\verification_plan.md` - validation plan
- `docs\superpowers\plans\2026-06-01-chat-mode-contract-harness.md` - chat mode contract implementation plan
- `configs\models.yaml` - local default model paths
- `configs\renderer_profiles.yaml` - renderer presets
- `re-learning\` - re-learning scripts, logs, and evaluation notes

## Current Hold / Evaluation Notes

Do not promote trained or re-learned models without evaluation.

Known held model note:

- `re-learning\eval\vision_tagger_clean_v1_HOLD_EVAL_FAILED.md`

Recent e10k/pose-boost training artifacts may exist under:

- `D:\Projects\training\out\gemmanima_relearning_v1`

Treat checkpoint-only outputs as hold artifacts until merge and evaluation
complete.
