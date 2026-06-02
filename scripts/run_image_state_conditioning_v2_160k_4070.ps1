$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:GEMMA_EMBED_ON_GPU = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TQDM_DISABLE = "1"

$Stage = "image_state_conditioning_v2_160k"
$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$Manifest = Join-Path $Repo "reports\image_state_conditioning_v2_160k\hiddenstage_multimodal_planner_anima_v2_combined_160k.jsonl"
$RootG = "D:\Projects\danbooru_data_set"
$RootETop = "D:\Projects\danbooru_data_set_e_top"
$Subset = Join-Path $Repo "reports\image_state_conditioning_v2_160k\subset_160k.jsonl"
$TargetDir = Join-Path $Repo "runs\cache\image_state_conditioning_v2_160k\targets"
$Checkpoint = Join-Path $Repo "runs\cache\image_state_conditioning_v2_160k\bridge\image_state_conditioning_v2_160k_image_translator.pt"
$TrainReport = Join-Path $Repo "runs\cache\image_state_conditioning_v2_160k\reports\image_state_conditioning_v2_160k_train_report.json"
$TextAnchor = "E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt"
$EmbeddedPython = "E:\ComfyUI_sage\python_embeded\python.exe"
$MultimodalDatasetScript = "E:\anima_gemma_swap\scripts\core\hiddenstage_multimodal_dataset.py"
$TargetScript = "E:\anima_gemma_swap\scripts\core\06_cache_targets.py"
$TrainScript = Join-Path $Repo "scripts\train_image_state_translator.py"
$LogDir = Join-Path $Repo "reports\image_state_conditioning_v2_160k\logs"
$ManifestLog = Join-Path $LogDir "00_combined_manifest.log"
$SubsetLog = Join-Path $LogDir "01_subset.log"
$CacheLog = Join-Path $LogDir "02_cache_targets.log"
$TrainLog = Join-Path $LogDir "03_train.log"

New-Item -ItemType Directory -Force -Path (Split-Path $Subset), $TargetDir, (Split-Path $Checkpoint), (Split-Path $TrainReport), $LogDir | Out-Null
Set-Location $Repo

& $EmbeddedPython $MultimodalDatasetScript `
  --data $RootG $RootETop `
  --out $Manifest `
  --limit 160000 `
  --prompt-style anima_v2 `
  *> $ManifestLog
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m gemmanima.cli image-state-conditioning-subset `
  --source-manifest $Manifest `
  --output $Subset `
  --limit 160000 `
  --json `
  *> $SubsetLog
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $EmbeddedPython $TargetScript `
  --subset $Subset `
  --outdir $TargetDir `
  --shard 1000 `
  --resume `
  *> $CacheLog
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $EmbeddedPython $TrainScript `
  --subset $Subset `
  --targets $TargetDir `
  --out $Checkpoint `
  --text-translator-anchor $TextAnchor `
  --epochs 1 `
  --batch-size 4 `
  --lr 0.0002 `
  --val 256 `
  --report $TrainReport `
  *> $TrainLog
exit $LASTEXITCODE
