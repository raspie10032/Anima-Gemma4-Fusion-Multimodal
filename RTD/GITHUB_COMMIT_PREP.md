# GitHub Commit Prep

This checklist is for preparing the GemmAnima v0.1 prototype source commit.

## Commit Scope

Include:

- App/runtime source code
- Local GUI/API code
- GUI-visible model asset download progress
- Generation preset and bridge-profile routing code
- Model asset registry and first-run downloader metadata
- RTD docs, scripts, payloads, and config snapshots
- Tests that verify the local runtime contracts

Exclude:

- Model weights and checkpoints
- Render outputs
- HF upload staging files
- Training/evaluation report dumps
- Long-running cache or training logs

## Runtime Contract

The GitHub app should be local-first:

- Chat runs through the local Gemma/TIPO runtime.
- Tagging runs through the local Gemma/TIPO vision path.
- Image-generation planning, presets, bridge-profile selection, and rendering
  orchestration are implemented in this repository.
- No external AI API is required for normal runtime once model assets are
  present locally.
- Network access is limited to first-run asset downloads from original model
  pages or the Hugging Face prototype adapter bundle.
- First-run downloads must be visible in the GUI through overall and per-file
  progress gauges.

## Documentation Boundary

- GitHub README: app/source usage, local setup, tests, and RTD runtime.
- Hugging Face model card: uploaded adapter/checkpoint bundle only.

## Pre-Commit Verification

Run:

```powershell
python -m pytest -q
python -m json.tool RTD\asset_manifest.json
python -m json.tool RTD\configs\model_sources.json
```

Also verify:

```powershell
git ls-files --others --exclude-standard
git status --short
```

The staged set should not contain `runs/`, `reports/`, model files, or local
HF upload staging payloads.
