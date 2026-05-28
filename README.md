# GemmAnima Engine

Minimal backend scaffold for the Anima-Gemma4 hidden-stage control flow.

The current implementation is a working dry-run control path:

1. Route normal chat without touching image components.
2. Detect image requests.
3. Build a relevant context capsule.
4. Validate a generation plan.
5. Produce a hidden-stage conditioning bundle through an adapter interface.
6. Call a renderer adapter.
7. Record a manifest for reproducibility.

The trained multimodal planner artifacts are registered as planner-side assets:

- `D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2\adapter_model.safetensors`
- `D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2\embed_vision.pt`

The low-level `Gemma hidden [B,S,1536] -> Anima crossattn_emb [B,512,1024]` bridge remains isolated behind `HiddenStageExit` so it can be replaced by the real bridge without changing conductor logic.

Current bridge checkpoint:

- `E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt`
- val MSE: `0.001104317136865575`
- status: passed

## Training Utilities

Useful status commands:

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

The current bridge pipeline is documented in [docs/training_pipeline.md](docs/training_pipeline.md).

## Real Render Smoke

The trained bridge has been smoke-tested through the existing Anima chat-to-image script on the RTX 4070 Ti SUPER:

```powershell
python -m gemmanima.cli real-render-health --json
python -m gemmanima.cli real-render-command --json
```

Reference output:

- `runs\images\nahida_hiddenstage_bridge_real_smoke.png`

The normal backend still defaults to dry-run rendering. To route a request through the real external Anima renderer:

```powershell
python -m gemmanima.cli run "draw a bright forest" --renderer real --json
```

To use the repo-native in-process renderer on the RTX 4070 Ti SUPER:

```powershell
python -m gemmanima.cli in-process-render-smoke-command --json
```

Latest in-process smoke output:

- `runs\images\83722a408c6f4b12ab1ae3295c171b70.png`

In-process renderer migration status:

```powershell
python -m gemmanima.cli renderer-backends --json
```

`external_script` is the current working real renderer. `in_process` reports dependency readiness separately from implementation readiness while the legacy script is being ported into repo-native modules.

Repo-native in-process pieces currently ported:

- `gemmanima.rendering.comfy_bootstrap`
- `gemmanima.rendering.gemma_hidden`
- `gemmanima.rendering.t5_tokenizer`

Gemma hidden provider environment smoke:

```powershell
python -m gemmanima.cli gemma-hidden-smoke-command --json
python -m gemmanima.cli t5-tokenizer-smoke-command --json
```

## Local GUI

The backend includes a small local operations console for health checks and
chat-to-image smoke requests:

```powershell
python -m gemmanima.cli gui-command
```

Open the printed URL, usually `http://127.0.0.1:8765`. The GUI talks to:

- `GET /v1/health`
- `POST /v1/chat`

The GUI defaults to dry-run rendering. Switch the renderer to `in-process` only
when the RTX 4070 Ti SUPER is free and the Comfy/Anima dependencies are ready.
