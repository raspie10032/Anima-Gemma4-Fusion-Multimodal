$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "E:\ComfyUI_sage\python_embeded\python.exe"
$CacheScript = "E:\anima_gemma_swap\scripts\core\06_cache_targets.py"
$Subset = Join-Path $RepoRoot "runs\teacher_targets\hiddenstage_multimodal_planner_anima_v2_teacher_subset.jsonl"
$OutDir = "E:\anima_gemma_swap\cache_hiddenstage_planner_v2\targets"

$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Set-Location $RepoRoot
& $PythonExe $CacheScript `
  --subset $Subset `
  --outdir $OutDir `
  --shard 2000 `
  --resume
