# HiddenStage Training Pipeline

## Current State

The multimodal planner LoRA is complete and passed evaluation:

- Output: `D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2`
- Eval step: `20500`
- Eval loss: `1.0061092711985111`
- Threshold: `1.5`

The next model training target is the HiddenStage Exit bridge:

```text
Gemma hidden [B,S,1536] -> Anima crossattn_emb [B,512,1024]
```

## Stage 1: Teacher Crossattn Targets

Teacher target extraction is running from:

```text
runs\teacher_targets\hiddenstage_multimodal_planner_anima_v2_teacher_subset.jsonl
```

Output:

```text
E:\anima_gemma_swap\cache_hiddenstage_planner_v2\targets
```

GPU split:

- RTX 4070 Ti SUPER: `shard_[0-9][0-9][0-9][0-9].pt`
- RTX 5060: `shard_5060_*.pt`

Each target shard stores:

- `idx`
- `t5_ids`
- `target`: original Anima LLMAdapter output, shape `[T,1024]`, dtype `float16`

## Stage 2: Gemma Hidden Cache

After target extraction completes, run:

```powershell
.\scripts\run_gemma_hidden_cache_split_4070_ti_super.ps1
.\scripts\run_gemma_hidden_cache_split_5060.ps1
```

Output:

```text
D:\anima_gemma_swap_cache_hiddenstage_planner_v2\gemma
```

The 4070 Ti SUPER runner uses `GEMMA_EMBED_ON_GPU=1`.
The 5060 runner keeps embeddings on CPU with `GEMMA_EMBED_ON_GPU=0`.

## Stage 3: Bridge Training

When every target shard has a matching Gemma shard, run:

```powershell
.\scripts\run_hiddenstage_bridge_train_4070_ti_super.ps1
```

Output:

```text
E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt
```

Training uses the existing Anima adapter bridge objective from `08_train_stream_batched.py`.
Only the Gemma-facing k/v projection path is trainable.

Current result:

- Epoch: `2`
- Val MSE: `0.001104317136865575`
- Gate: `0.004`
- Status: passed

## Stage 4: Backend Bridge Integration

The trained bridge checkpoint is registered in the GemmAnima backend:

```text
gemmanima.modules.hiddenstage_exit.HiddenStageExit
gemmanima.modules.bridge_runtime.TrainedBridgeRuntime
```

Smoke command:

```powershell
E:\ComfyUI_sage\python_embeded\python.exe scripts\smoke_hiddenstage_bridge_forward.py --checkpoint E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt
```

Expected smoke output:

```text
input  [1,16,1536]
t5_ids [1,32]
output [1,32,1024]
finite true
```

The normal backend dry-run now records `trained_hiddenstage_bridge` and the checkpoint audit in the manifest.

## Next Stage: Real Renderer Wiring

The trained bridge has now been validated through the existing Anima chat-to-image renderer:

```powershell
E:\ComfyUI_sage\python_embeded\python.exe E:\anima_gemma_swap\scripts\core\18_hiddenstage_chat_generate.py --request "Draw Nahida from Genshin Impact as a bright forest anime illustration, gentle expression, detailed green-and-white outfit, soft sunlight." --adapter E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt --out C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\images\nahida_hiddenstage_bridge_real_smoke.png --size 512 --steps 12 --cfg 4.5 --unet-dtype fp8_e4m3fn_fast
```

Observed smoke result:

- GPU: `NVIDIA GeForce RTX 4070 Ti SUPER`
- Steps: `12`
- Output: `runs\images\nahida_hiddenstage_bridge_real_smoke.png`

The repo CLI can print the same command:

```powershell
python -m gemmanima.cli renderer-backends --json
python -m gemmanima.cli real-render-health --json
python -m gemmanima.cli real-render-command --json
```

The backend can also route generation requests through the real external renderer:

```powershell
python -m gemmanima.cli run "draw a bright forest" --renderer real --json
```

The current real renderer is a first-class backend adapter around the legacy external script. The in-process backend now has repo-native bootstrap diagnostics in `gemmanima.rendering.comfy_bootstrap`, repo-native Gemma hidden provider support in `gemmanima.rendering.gemma_hidden`, repo-native T5 tokenizer support in `gemmanima.rendering.t5_tokenizer`, and repo-native sampler/VAE decode support in `gemmanima.rendering.anima_sampler`.

The in-process renderer has passed a 4070 Ti SUPER smoke:

```powershell
$env:CUDA_VISIBLE_DEVICES='0'; $env:GEMMA_EMBED_ON_GPU='1'; E:\ComfyUI_sage\python_embeded\python.exe scripts\smoke_in_process_render.py --image-root runs\images --manifest-root runs\manifests --json
```

Observed output:

- Status: `completed`
- Renderer: `dry_run=false`
- Output: `runs\images\83722a408c6f4b12ab1ae3295c171b70.png`

## Status Commands

```powershell
python -m gemmanima.cli training-readiness --json
python -m gemmanima.cli prepare-gemma-cache --json
python -m gemmanima.cli prepare-bridge-training --json
python -m gemmanima.cli bridge-eval-status --json
python -m gemmanima.cli bridge-smoke-command --json
python -m gemmanima.cli gemma-hidden-smoke-command --json
python -m gemmanima.cli t5-tokenizer-smoke-command --json
python -m gemmanima.cli in-process-render-smoke-command --json
python -m gemmanima.cli renderer-backends --json
python -m gemmanima.cli real-render-health --json
python -m gemmanima.cli real-render-command --json
```
