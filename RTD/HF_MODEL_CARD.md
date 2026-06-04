---
language:
- en
- ko
tags:
- gemma
- gguf
- lora
- multimodal
- image-generation
- anime
- danbooru
- local-runtime
license: other
---

# GemmAnima Prototype Adapter Bundle

[Korean model card](HF_MODEL_CARD.ko.md) | [GitHub app source](https://github.com/raspie10032/Anima-Gemma4-Fusion-Multimodal)

## Prototype Notice

This is a **v0.1 prototype adapter bundle** for the standalone GemmAnima app.
It is published so the current local pipeline can be reproduced and tested. It
is not a safety-rated assistant, not a production image model, and not a
commercial-ready release.

The repository intentionally contains GemmAnima adapters, projectors, bridge
checkpoints, metadata, and docs only. Upstream base weights must be downloaded
from their original model pages.

## Quick Summary

| Area | Status |
| --- | --- |
| Release type | Public prototype adapter/checkpoint bundle |
| Base weights | Not mirrored here |
| Runtime shape | One shared Gemma GGUF plus task LoRA/adapters, Anima base weights, and bridge profiles |
| Evaluation | Local smoke tests and bridge checks only |
| Safety | Not safety-rated |
| License posture | Adapter bundle notice plus upstream base-model restrictions |

GemmAnima is a local multimodal adapter bundle that connects a Gemma/TIPO-style
language and vision-tagging core to an Anima image-generation core through a
HiddenStage bridge.

This repository is intended for GemmAnima-owned adapters, projectors, bridge
checkpoints, and runtime metadata for the standalone GemmAnima app. It is not a
single monolithic model file, and it should not mirror upstream base model
weights.

## Status

- Release status: v0.1 public prototype adapter/checkpoint release
- Safety status: not safety-rated
- Evaluation status: partial smoke and bridge checks only
- License status: adapter/checkpoint bundle with upstream dependency licenses;
  see `LICENSE_NOTICES.md`
- Visibility: public adapter-only repository; upstream base model weights are
  not mirrored here

Do not treat this prototype as a promoted production model without additional evaluation.
Do not treat this adapter bundle as relicensing any upstream base model.

## Consumer Download

This Hugging Face repository is the adapter/checkpoint bundle. The GitHub
repository is the app/source-code surface. On a GemmAnima checkout, let the app
plan or download model assets:

```powershell
python -m gemmanima.cli model-download-plan --json
python -m gemmanima.cli ensure-model-assets --json
```

The app/source README owns local setup, GUI, development, and runtime details.
This model card only describes the files published in the adapter bundle.

## Model Parts

This repository should contain only the files produced or adapted by the
GemmAnima project. Original base models should be downloaded from their original
distribution pages and placed locally according to the standalone app
configuration.

### 1. Gemma Core

The Gemma Core handles chat, language-harness behavior, canonical English
Danbooru tag output, and the prototype Gemma vision fallback path.

GemmAnima files:

| File | Role |
| --- | --- |
| `text-adapter-model-f16.gguf` | Text/chat LoRA adapter |
| `vision-tagger-adapter-model-f16.gguf` | Vision/tagger LoRA adapter, refreshed from the mixed-pose-front v2 final prototype |
| `gemma4-tipo-vision.mmproj-f16.gguf` | Vision projector paired with the mixed-pose-front v2 final prototype |

External requirement:

| File | How to obtain |
| --- | --- |
| `gemma-4-E2B-it-heretic-ara-custom.Q4_K_M.gguf` | Download from `mradermacher/gemma-4-E2B-it-heretic-ara-custom-GGUF`; do not mirror it here unless redistribution is explicitly permitted and intentionally chosen. |

Upstream license metadata for the GGUF page currently reports `apache-2.0`.

The preferred runtime shape is an upstream base GGUF loaded with GemmAnima
task-specific LoRA adapters through llama.cpp `--lora`. Older fully merged GGUFs
are not the preferred packaging shape.

### 1.5. Gemma Vision Tagging

The default app `tag-image` route uses the Gemma vision LoRA and vision mmproj
listed above. No separate ONNX tagger is part of the required standalone
runtime.

### 2. Anima Image Core

The Anima Image Core handles diffusion sampling and VAE decoding.

GemmAnima files:

This repository does not need to contain Anima Image Core base weights.

External requirements:

| File | Role |
| --- | --- |
| `split_files/diffusion_models/anima-base-v1.0.safetensors` | Download from `circlestone-labs/Anima` |
| `split_files/vae/qwen_image_vae.safetensors` | Download from `circlestone-labs/Anima` |

The upstream Anima page currently reports
`circlestone-labs-non-commercial-license` and states that Anima is also subject
to the NVIDIA Open Model License Agreement where applicable because it is a
derivative of NVIDIA Cosmos-Predict2-2B-Text2Image.

Anima text encoder weights are not part of the required standalone runtime. The
current in-process renderer uses Anima-compatible tokenizer metadata
(`t5xxl_ids` and `t5xxl_weights`) for conditioning shape compatibility, not the
Anima text encoder weight file.

Supported generation controls in the standalone app:

| Type | Supported values |
| --- | --- |
| Sampler | `euler`, `euler_ancestral`, `dpmpp_2m`, `dpmpp_2m_sde_gpu` |
| Scheduler | `normal`, `karras`, `sgm_uniform` |
| Resolution presets | `1024x1024`, `832x1216`, `768x1344`, custom |

### 3. HiddenStage Bridge

The HiddenStage Bridge connects Gemma hidden-state features to Anima-compatible
conditioning.

GemmAnima files:

| File | Role |
| --- | --- |
| `hiddenstage-planner-adapter.safetensors` | Planner LoRA adapter |
| `hiddenstage-planner-embed-vision.pt` | Planner vision embedding |
| `kv_proj_hiddenstage_planner_v2.pt` | HiddenStage bridge checkpoint |
| `kv_proj_text_delta_300k_from_epoch1_a0p35.pt` | Retained experimental text-delta bridge; not a default image route after live render QA |
| `kv_proj_text_exact_v27_alpha35.pt` | Prototype default quality bridge profile for normal image generation, style-tag prompts, signs, labels, captions, and readable-text prompts |

The standalone app routes bridge profiles automatically:

| Profile | Automatic use |
| --- | --- |
| `balanced_pose` | Normal image-generation prompts; routed to `kv_proj_text_exact_v27_alpha35.pt` |
| `style_artist` | Style-oriented tags and surface-token-heavy prompts; routed to `kv_proj_text_exact_v27_alpha35.pt` |
| `text_exact` | Prompts asking for readable text, signs, labels, captions, or logos |
| `legacy_mse` | Compatibility baseline and explicit override |

Current local bridge metadata:

| Metric | Value |
| --- | --- |
| Bridge validation MSE | `0.001104317136865575` |
| Bridge gate | passed |
| Planner eval loss | `1.0061092711985111` |
| Planner eval threshold | `1.5` |

These are engineering gate metrics and small local smoke tests, not a full
end-user quality evaluation.

## Uploaded Files

Checksums and byte sizes for every uploaded file are recorded in
`adapter_manifest_v0.1.json`. The main uploaded payload is:

| Directory | Files |
| --- | --- |
| `gemma_core/` | Text LoRA, prototype Gemma vision/tagger LoRA, vision mmproj |
| `hiddenstage_bridge/` | Planner adapter, planner vision embedding, legacy bridge, and three prototype bridge profiles |
| repository root | Hugging Face model card, license notices, model source metadata, adapter manifest, version marker |

## Approximate Size

This adapter repository should be much smaller than the full local runtime
because original base weights are expected to come from their original pages:

| Part | Approximate upload size |
| --- | ---: |
| Gemma Core adapters/projector | ~1.06 GB |
| HiddenStage Bridge | ~0.40 GB |
| Anima Image Core base weights | not uploaded here |
| Total uploaded here | ~1.46 GB |

For local runtime planning, the full standalone runtime is about 9 GB in decimal
units after the user downloads the external base weights separately:

| Part | Approximate size |
| --- | ---: |
| Gemma Core | ~4.51 GB |
| Vision Tagger | ~0.47 GB |
| Anima Image Core | ~4.44 GB |
| HiddenStage Bridge | ~0.40 GB |
| Total | ~9.81 GB |

Exact size depends on final filenames and whether source adapters or
compatibility reference models are included.

## Intended Use

This bundle is intended for:

- Local GemmAnima app runtime testing
- Korean or English chat with an explicit language harness
- Canonical English Danbooru tag output for tag requests
- Chat-driven image-generation request planning
- Anima image rendering through the app-controlled preset system

For tag requests, output tags should remain canonical English Danbooru tags even
when the user-facing chat language is Korean.

## Out of Scope

This bundle is not intended as:

- A general-purpose safety-filtered assistant
- A fully evaluated public image-generation model
- A replacement for downloading or licensing upstream base components
- A license override for Gemma, Anima, NVIDIA Cosmos, or any source dataset
- A guarantee of pose, anatomy, text rendering, or prompt fidelity

## Known Limitations

- Safety and content behavior have not been fully evaluated.
- Pose understanding remains an active improvement area.
- Some broad ComfyUI samplers were intentionally not exposed because they were
  not part of the app-supported smoke-tested subset.
- The app currently owns a curated sampler/scheduler contract rather than
  exposing every option from ComfyUI.
- Bridge profile checkpoints are prototype routing choices and should not be
  promoted without separate evaluation.
- Base model files are external dependencies and should remain linked to their
  original distribution pages.

## Runtime Notes

Recommended local runtime:

- Windows + PowerShell
- RTX 4070 Ti SUPER as the primary PyTorch/rendering GPU
- llama.cpp CUDA build for Gemma Core inference
- GemmAnima standalone app for orchestration

Keep RTX 5060 out of PyTorch cache/training paths unless explicitly re-enabled
with a compatible PyTorch build.

## Release Checklist

For this v0.1 public prototype adapter-only release:

- The repository remains adapter/checkpoint-only.
- Upstream base weights are referenced, not mirrored.
- `LICENSE_NOTICES.md` is included.
- SHA256 checksums are recorded in `adapter_manifest_v0.1.json`.

Still required before promotion:

- Run and publish a small reproducible inference smoke.
- Run a safety and content-policy review.
- Add representative example outputs only after evaluation.

## Example File Layout

```text
.
|-- README.md
|-- LICENSE_NOTICES.md
|-- model_sources.json
|-- adapter_manifest_v0.1.json
|-- gemma_core/
|   |-- text-adapter-model-f16.gguf
|   |-- vision-tagger-adapter-model-f16.gguf
|   `-- gemma4-tipo-vision.mmproj-f16.gguf
`-- hiddenstage_bridge/
    |-- hiddenstage-planner-adapter.safetensors
    |-- hiddenstage-planner-embed-vision.pt
    |-- kv_proj_hiddenstage_planner_v2.pt
    |-- kv_proj_text_delta_300k_from_epoch1_a0p35.pt
    `-- kv_proj_text_exact_v27_alpha35.pt
```

External base weights should be downloaded separately from their original model
pages and referenced by the local GemmAnima app configuration.

The standalone app download plan uses:

```powershell
python -m gemmanima.cli model-download-plan --json
python -m gemmanima.cli ensure-model-assets --json
```

## Citation and Attribution

This is a composite adapter/checkpoint bundle. It does not relicense upstream
base models. Add upstream citations, original download links, and license
notices for the base model, image model, VAE, NVIDIA dependency, and any
training datasets before a public release.
