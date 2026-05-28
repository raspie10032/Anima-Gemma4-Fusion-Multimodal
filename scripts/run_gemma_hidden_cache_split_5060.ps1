$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "E:\ComfyUI_sage\python_embeded\python.exe"
$CacheScript = "E:\anima_gemma_swap\scripts\core\07_cache_gemma_batched.py"
$Subset = Join-Path $RepoRoot "runs\teacher_targets\hiddenstage_multimodal_planner_anima_v2_teacher_subset.jsonl"
$TargetDir = "E:\anima_gemma_swap\cache_hiddenstage_planner_v2\targets"
$OutDir = "D:\anima_gemma_swap_cache_hiddenstage_planner_v2\gemma"

$env:CUDA_VISIBLE_DEVICES = "1"
$env:GEMMA_EMBED_ON_GPU = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Set-Location "E:\anima_gemma_swap\scripts\core"
& $PythonExe $CacheScript `
  --subset $Subset `
  --target-dir $TargetDir `
  --outdir $OutDir `
  --patterns "shard_5060_*.pt" `
  --batch-size 8 `
  --resume
