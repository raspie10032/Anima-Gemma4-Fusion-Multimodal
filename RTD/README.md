# GemmAnima RTD Bundle

RTD means "Ready To Drive" for this local checkout. This is a v0.1 prototype
runtime desk: scripts, payload examples, config snapshots, and an external asset
manifest. It intentionally does not copy model files into the repo.

The RTD runtime target is local-first. Chat, tag routing, generation presets,
bridge-profile routing, and render orchestration are app code, not external AI
API calls. Network access is only for first-run model asset downloads.

The local GUI includes a model-asset download panel. First-run users can start
downloads there and watch an overall queue gauge plus a current-file byte gauge.

Use `..\GemmAnima.bat` as the only Windows launcher. It intentionally collects
GUI startup, health checks, downloads, dry-run smoke, tagging, and tests behind
one executable-style entry point.

Because the project is distributed as source, the launcher also owns the local
`.venv`. Run `..\GemmAnima.bat bootstrap` once on a fresh checkout, or let the
first launcher command perform the same visible bootstrap path. The venv is
created with `--system-site-packages` so existing CUDA Python runtimes can be
reused instead of silently replaced.

The chat UI keeps the resident Gemma text runtime at the maximum supported
256k context window by default (`GEMMANIMA_TIPO_TEXT_N_CTX=262144`). It also
includes an optional embedded Headroom-style context compressor. The compressor
follows the local-history compression idea from `chopratejas/headroom`, but the
runtime compressor is built into GemmAnima and does not install or call an
external package or service.

The launcher does not perform hidden package installation after bootstrap. Use
`dependency-audit` to inspect which runtime engines are available; missing
runtime engines should be solved intentionally in the source/runtime
environment, not by surprise install steps at GUI start.

The runtime model set is split into four named parts:

### Gemma Core

One shared base GGUF plus LoRA/adapters:

- Base GGUF:
  `%LOCALAPPDATA%\GemmAnima\models\gemma_core\gemma-4-E2B-it-heretic-ara-custom.Q4_K_M.gguf`
- Text/chat LoRA:
  `%LOCALAPPDATA%\GemmAnima\models\gemma_core\text-adapter-model-f16.gguf`
- Vision/tagger LoRA:
  `%LOCALAPPDATA%\GemmAnima\models\gemma_core\vision-tagger-adapter-model-f16.gguf`
- Vision mmproj:
  `%LOCALAPPDATA%\GemmAnima\models\gemma_core\gemma4-tipo-vision.mmproj-f16.gguf`

The runtime now defaults to base GGUF plus llama.cpp `--lora` loading. Merged
TIPO GGUF files stay in `asset_manifest.json` only as compatibility/reference
artifacts.

### Anima Image Core

- Diffusion model:
  `%LOCALAPPDATA%\GemmAnima\models\anima_image_core\anima-base-v1.0.safetensors`
- VAE:
  `%LOCALAPPDATA%\GemmAnima\models\anima_image_core\qwen_image_vae.safetensors`

Anima text encoder weights are not part of the required standalone runtime. The
in-process renderer keeps only tokenizer-format metadata for Anima conditioning.

### Vision Tagger

- WD SwinV2 ONNX model:
  `%LOCALAPPDATA%\GemmAnima\models\vision_tagger\wd-swinv2-tagger-v3\model.onnx`
- WD tag vocabulary:
  `%LOCALAPPDATA%\GemmAnima\models\vision_tagger\wd-swinv2-tagger-v3\selected_tags.csv`

The default `tag-image` route uses the local WD SwinV2 Danbooru tagger because
live generated-image evaluation showed stronger scene, object, and pose tag
correlation than the prototype Gemma vision LoRA. The Gemma vision LoRA/mmproj
path remains available as a fallback and for future experiments.

### HiddenStage Bridge

- Planner adapter:
  `%LOCALAPPDATA%\GemmAnima\models\hiddenstage_bridge\hiddenstage-planner-adapter.safetensors`
- Planner vision embedding:
  `%LOCALAPPDATA%\GemmAnima\models\hiddenstage_bridge\hiddenstage-planner-embed-vision.pt`
- Bridge checkpoint:
  `%LOCALAPPDATA%\GemmAnima\models\hiddenstage_bridge\kv_proj_hiddenstage_planner_v2.pt`

Runtime bridge profiles:

- `balanced_pose` - default automatic bridge for general image generation,
  composition, and quality-first prompts. In v0.1 it points to
  `kv_proj_text_exact_v27_alpha35.pt` after live render evaluation showed the
  300k text-delta bridge collapsing into abstract textures.
- `style_artist` - selected automatically for style-oriented tags and
  rare surface-token prompts. In v0.1 it also points to
  `kv_proj_text_exact_v27_alpha35.pt`; the earlier 10k style bridge and 300k
  text-delta bridge are retained but are not default routes.
- `text_exact` - selected automatically for signs, labels, logos, captions, or
  prompts that ask for readable text.
- `legacy_mse` - original MSE-gated bridge kept for compatibility and explicit
  override.

These profiles are prototype routing choices, not promoted production models.

## What Is Included

- `asset_manifest.json` - portable relative asset layout for the local model
  root.
- `GITHUB_README.md` - GitHub-facing documentation boundary.
- `LICENSE_NOTICES.md` - source license and attribution checklist.
- `HF_MODEL_CARD.md` - Hugging Face-facing adapter bundle model card.
- `HF_MODEL_CARD.ko.md` - Korean Hugging Face-facing model card.
- `configs\models.yaml` - model path snapshot.
- `configs\model_sources.json` - first-run download source snapshot.
- `configs\renderer_profiles.yaml` - renderer profile snapshot.
- `..\GemmAnima.bat` - single Windows launcher for GUI, health, downloads,
  dry-run smoke, tagging, and tests.
- `scripts\*.ps1` - developer helper scripts used by the launcher and legacy
  smoke paths; not first-run user launchers.
- `payloads\chat_general.json` - text chat request example.
- `payloads\chat_image_generation.json` - chat-to-image request example.
- `payloads\tag_image_template.json` - vision tag request template.

## First Check

From the repo root:

```powershell
.\GemmAnima.bat bootstrap
.\GemmAnima.bat health
```

Expected result:

- Python imports work.
- pytest can run.
- `/v1/health` equivalent CLI checks report asset state.
- Missing model or renderer assets appear as structured preflight issues.

## First-Run Model Download

GitHub does not carry model weights. On a fresh checkout, download required
assets before starting real rendering:

```powershell
.\GemmAnima.bat download
```

Default model root:

```text
%LOCALAPPDATA%\GemmAnima\models
```

Override it with:

```powershell
$env:GEMMANIMA_MODEL_ROOT = "D:\GemmAnima\models"
```

Source policy:

- Gemma base GGUF is downloaded from the original GGUF model page.
- Anima diffusion and VAE are downloaded from the original Anima model page.
- GemmAnima LoRA/mmproj/bridge files are downloaded from the GemmAnima adapter
  bundle repo. Override that repo with `GEMMANIMA_ADAPTER_REPO`.

The GUI downloads the same registry assets and displays progress for both base
models and adapter/checkpoint files.

## Start Local GUI

```powershell
.\GemmAnima.bat
```

Open:

```text
http://127.0.0.1:8765
```

## Dry-Run Smoke

```powershell
.\GemmAnima.bat dry-run "draw a bright forest anime illustration"
```

This does not require real renderer assets.

## External Script Smoke

```powershell
.\RTD\scripts\smoke_external_script.ps1
```

This is a legacy compatibility smoke, not the RTD default. Use it only when the
RTX 4070 Ti SUPER is free and the paths in `asset_manifest.json` are valid.

## Tag One Image

```powershell
.\GemmAnima.bat tag "D:\path\to\image.png"
```

The tagger should output canonical English Danbooru tags.

## API Payload Examples

Start the GUI/backend first, then send a payload:

```powershell
$payload = Get-Content .\RTD\payloads\chat_image_generation.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/v1/chat -ContentType "application/json" -Body $payload
```

The chat-to-image payload uses generation presets instead of letting the model
choose render settings:

- Resolution presets: `square_1024`, `portrait_832_1216`,
  `portrait_768_1344`, `custom`.
- `orientation=landscape` swaps width and height for the portrait presets.
- `custom` reads `custom_width` and `custom_height`.
- Sampler and scheduler names use the app-supported subset curated from ComfyUI
  `KSampler`; leave them blank to use the selected generation preset default.
- Supported samplers: `euler`, `euler_ancestral`, `dpmpp_2m`,
  `dpmpp_2m_sde_gpu`.
- Supported schedulers: `normal`, `karras`, `sgm_uniform`.

## Notes

- Keep RTX 5060 out of PyTorch cache/training unless explicitly re-enabled.
- Do not promote trained or re-learned models without evaluation.
- Do not copy external model files into RTD unless a separate packaging pass
  explicitly asks for an offline bundle.
- Prefer base GGUF plus LoRA/adapters over storing multiple full merged GGUFs.
