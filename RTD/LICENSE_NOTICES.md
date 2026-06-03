# GemmAnima License Notices

This notice is for the GemmAnima adapter/checkpoint bundle. It is not a legal
opinion. Verify upstream terms before any public or commercial release.

## Bundle Scope

This repository should contain only GemmAnima-owned or GemmAnima-adapted files:

- Gemma Core LoRA adapters and vision projector
- HiddenStage Bridge planner adapter, vision embedding, and bridge checkpoint
- Runtime metadata and documentation

Original base model weights are external dependencies. They should be downloaded
from their original model pages at first run and should not be mirrored in this
repository unless redistribution permission is explicitly documented.

## Source License Summary

| Component | Source | License / notice |
| --- | --- | --- |
| Gemma Core base GGUF | Original Gemma/GGUF model page | Follow the current upstream Gemma/GGUF license, terms, required notices, and use restrictions |
| Gemma-dependent logic and ideas | GemmAnima routing, harness, prompt-contract, hidden-state bridge, and adapter behavior built around Gemma | Treat as Gemma-dependent project material where applicable; do not use this project notice to bypass or relicense Gemma terms |
| Anima diffusion model | `circlestone-labs/Anima` | CircleStone Labs Non-Commercial License according to the upstream model page |
| Anima VAE file | `circlestone-labs/Anima` | Follow the upstream Anima page and the original VAE/model notices |
| Chat context compression reference | `chopratejas/headroom` | Referenced design idea for local context compression; GemmAnima embeds its own minimal compressor and does not vendor Headroom package code |
| GemmAnima adapters/checkpoints | GemmAnima adapter bundle | Composite prototype notice; use is constrained by upstream Gemma, Anima, NVIDIA, and dataset/source restrictions |

The Anima model page states that Anima is licensed under the CircleStone Labs
Non-Commercial License, is usable only for non-commercial purposes, and is a
Derivative Model of NVIDIA Cosmos-Predict2-2B-Text2Image subject to the NVIDIA
Open Model License Agreement where applicable.

## Practical Release Rule

For the adapter bundle, use the most restrictive applicable terms across the
loaded stack:

1. GemmAnima adapter/checkpoint bundle notice
2. Gemma/GGUF upstream license and source model terms
3. Anima upstream license
4. NVIDIA license terms referenced by the Anima upstream page
5. Dataset/source policy for any training data used to create the adapter

Because the current image path depends on Anima, treat the public adapter
release as non-commercial/restricted-use only until a formal review and any
required upstream permissions say otherwise. Do not present it as
production-ready, commercial-ready, or as relicensing any upstream base model.

## Required Attribution in Model Card

Before public release, the model card should include:

- Original Gemma/GGUF repository link and license
- Original Anima repository link and non-commercial license notice
- NVIDIA Open Model License reference as required by Anima
- Statement that original base weights are not mirrored here
- Statement that GemmAnima adapters do not relicense upstream base models
- Dataset/source policy summary for adapter training data

## Files Not To Upload Here

- `gemma-4-E2B-it-heretic-ara-custom.Q4_K_M.gguf`
- `anima-base-v1.0.safetensors`
- `qwen_image_vae.safetensors`
- Anima text encoder weights

These files should be downloaded from their original pages by the first-run
asset downloader.
