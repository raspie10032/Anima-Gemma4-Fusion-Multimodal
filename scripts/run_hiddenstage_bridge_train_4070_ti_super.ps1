$ErrorActionPreference = "Stop"

$PythonExe = "E:\ComfyUI_sage\python_embeded\python.exe"
$TrainScript = "E:\anima_gemma_swap\scripts\core\08_train_stream_batched.py"
$TargetDir = "E:\anima_gemma_swap\cache_hiddenstage_planner_v2\targets"
$GemmaDir = "D:\anima_gemma_swap_cache_hiddenstage_planner_v2\gemma"
$Out = "E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt"

$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Set-Location "E:\anima_gemma_swap\scripts\core"
& $PythonExe $TrainScript `
  --targets $TargetDir `
  --gemma $GemmaDir `
  --out $Out `
  --epochs 2 `
  --lr 5e-4 `
  --batch-size 2 `
  --accum 2 `
  --val 2000 `
  --prefetch-gb 48 `
  --save-every-shards 25
