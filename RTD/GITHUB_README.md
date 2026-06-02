# GemmAnima GitHub README Guide

This is the GitHub-facing documentation guide for the GemmAnima source
repository. Keep it separate from the Hugging Face model card.

## GitHub Surface

GitHub should explain the standalone app and source tree:

- What the app does
- How to run the local GUI/backend
- How first-run model asset downloads work
- How the Gemma Core, Anima Image Core, and HiddenStage Bridge fit together
- How to run tests
- Where runtime configs, payload examples, and manifests live
- What is still prototype-only

The GitHub-facing app promise is local-first: no external AI API dependency for
chat, tag routing, image planning, bridge-profile routing, or render
orchestration after model assets are present.

GitHub should not pretend to host model weights. It should point model asset
downloads to the original model pages or the Hugging Face adapter bundle.
Those downloads are bootstrap dependencies, not runtime AI service calls.

## Hugging Face Surface

Hugging Face should explain only the uploaded adapter/checkpoint bundle:

- What files are uploaded
- Which upstream base weights are not mirrored
- Which original pages supply those base weights
- License and attribution notices
- SHA256 and file metadata via `adapter_manifest_v0.1.json`
- Prototype limitations and non-promotion status

Do not put app development instructions, local dev path history, or training
workflow details in the Hugging Face model card unless they are required to
consume the adapter bundle.

## Current HF Bundle

- Repo: `raspie/gemmanima-adapter-bundle`
- Local staging folder:
  `runs\hf_upload\gemmanima-adapter-bundle-v0.1`
- HF model card source:
  `RTD\HF_MODEL_CARD.md`

## Current GitHub Docs

- Main source README: `README.md`
- RTD runtime desk: `RTD\README.md`
- License/source notices: `RTD\LICENSE_NOTICES.md`
- Model path snapshot: `RTD\configs\models.yaml`

## Tone

Call this a prototype. Avoid production, finished, promoted, safety-rated, or
commercial-ready language until separate evaluation and licensing review are
complete.
