# GemmAnima RTD Bundle

RTD means "Ready To Drive" for this local checkout. This is a v0.1 prototype
runtime desk: scripts, payload examples, config snapshots, and an external asset
manifest. It intentionally does not copy model files into the repo.

The RTD runtime target is local-first. Chat, tag routing, generation presets,
bridge-profile routing, and render orchestration are app code, not external AI
API calls. Network access is only for first-run model asset downloads.

The local GUI includes a model-asset download panel. First-run users can start
downloads there and watch an overall queue gauge plus a current-file byte gauge.

The runtime model set is split into three named parts:

### Gemma Core

One shared base GGUF plus LoRA/adapters:

- Base GGUF: `D:\Projects\training\out\gemma-4-E2B-it-heretic-ara-Q4_K_M.gguf`
- Text/chat LoRA: `D:\Projects\training\out\lora\adapter_model.f16.gguf`
- Vision/tagger LoRA:
  `D:\Projects\training\out\gemmanima_v4_vision_tagger\adapter_model.f16.gguf`
- Vision mmproj:
  `D:\Projects\training\out\_completed\gemma4-tipo-vision.mmproj-f16.gguf`

The runtime now defaults to base GGUF plus llama.cpp `--lora` loading. Merged
TIPO GGUF files stay in `asset_manifest.json` only as compatibility/reference
artifacts.

### Anima Image Core

- Diffusion model:
  `E:\ComfyUI_sage\ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors`
- VAE: `E:\ComfyUI_sage\ComfyUI\models\vae\qwen_image_vae.safetensors`

Anima text encoder weights are not part of the required standalone runtime. The
in-process renderer keeps only tokenizer-format metadata for Anima conditioning.

### HiddenStage Bridge

- Planner adapter:
  `D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2\adapter_model.safetensors`
- Planner vision embedding:
  `D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2\embed_vision.pt`
- Bridge checkpoint: `E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt`

Runtime bridge profiles:

- `balanced_pose` - default automatic bridge for general image generation,
  composition, and pose-sensitive prompts.
- `style_artist` - selected automatically for style-oriented tags and
  rare surface-token prompts.
- `text_exact` - selected automatically for signs, labels, logos, captions, or
  prompts that ask for readable text.
- `legacy_mse` - original MSE-gated bridge kept for compatibility and explicit
  override.

These profiles are prototype routing choices, not promoted production models.

## What Is Included

- `asset_manifest.json` - required external files and directories.
- `GITHUB_README.md` - GitHub-facing documentation boundary.
- `LICENSE_NOTICES.md` - source license and attribution checklist.
- `HF_MODEL_CARD.md` - Hugging Face-facing adapter bundle model card.
- `configs\models.yaml` - model path snapshot.
- `configs\model_sources.json` - first-run download source snapshot.
- `configs\renderer_profiles.yaml` - renderer profile snapshot.
- `scripts\health_check.ps1` - quick preflight commands.
- `scripts\run_gui.ps1` - starts the local GUI backend.
- `scripts\smoke_dry_run.ps1` - dry-run generation smoke.
- `scripts\smoke_external_script.ps1` - legacy fallback render smoke, not the
  RTD default.
- `scripts\tag_image.ps1` - TIPO vision tagger helper.
- `payloads\chat_general.json` - text chat request example.
- `payloads\chat_image_generation.json` - chat-to-image request example.
- `payloads\tag_image_template.json` - vision tag request template.

## First Check

From the repo root:

```powershell
python -m gemmanima.cli model-download-plan --json
.\RTD\scripts\health_check.ps1
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
python -m gemmanima.cli ensure-model-assets --json
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
.\RTD\scripts\run_gui.ps1
```

Open:

```text
http://127.0.0.1:8765
```

## Dry-Run Smoke

```powershell
.\RTD\scripts\smoke_dry_run.ps1
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
.\RTD\scripts\tag_image.ps1 -ImagePath "D:\path\to\image.png"
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
