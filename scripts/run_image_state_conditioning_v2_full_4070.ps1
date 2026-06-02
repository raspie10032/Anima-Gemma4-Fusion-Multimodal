$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:GEMMA_EMBED_ON_GPU = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TQDM_DISABLE = "1"

$Stage = "image_state_conditioning_v2_full"
$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$ReportDir = Join-Path $Repo "reports\$Stage"
$CacheRoot = Join-Path $Repo "runs\cache\$Stage"
$Manifest = Join-Path $ReportDir "hiddenstage_multimodal_planner_anima_v2_combined_full.jsonl"
$RootG = "D:\Projects\danbooru_data_set"
$RootETop = "D:\Projects\danbooru_data_set_e_top"
$Subset = Join-Path $ReportDir "subset_full.jsonl"
$TargetDir = Join-Path $CacheRoot "targets"
$Checkpoint = Join-Path $CacheRoot "bridge\image_state_conditioning_v2_full_image_translator.pt"
$TrainReport = Join-Path $CacheRoot "reports\image_state_conditioning_v2_full_train_report.json"
$TextAnchor = "E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt"
$EmbeddedPython = "E:\ComfyUI_sage\python_embeded\python.exe"
$MultimodalDatasetScript = "E:\anima_gemma_swap\scripts\core\hiddenstage_multimodal_dataset.py"
$TargetScript = "E:\anima_gemma_swap\scripts\core\06_cache_targets.py"
$TrainScript = Join-Path $Repo "scripts\train_image_state_translator.py"
$LogDir = Join-Path $ReportDir "logs"
$ManifestLog = Join-Path $LogDir "00_combined_manifest.log"
$SubsetLog = Join-Path $LogDir "01_subset.log"
$CacheLog = Join-Path $LogDir "02_cache_targets.log"
$TrainLog = Join-Path $LogDir "03_train.log"

New-Item -ItemType Directory -Force -Path $ReportDir, $TargetDir, (Split-Path $Checkpoint), (Split-Path $TrainReport), $LogDir | Out-Null
Set-Location $Repo

& $EmbeddedPython $MultimodalDatasetScript `
  --data $RootG $RootETop `
  --out $Manifest `
  --limit 0 `
  --prompt-style anima_v2 `
  *> $ManifestLog
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m gemmanima.cli image-state-conditioning-subset `
  --source-manifest $Manifest `
  --output $Subset `
  --limit 0 `
  --allow-missing-image-embed `
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
  --val 512 `
  --report $TrainReport `
  *> $TrainLog
exit $LASTEXITCODE
