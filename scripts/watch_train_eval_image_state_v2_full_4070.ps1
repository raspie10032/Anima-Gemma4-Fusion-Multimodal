$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:GEMMA_EMBED_ON_GPU = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$Stage = "image_state_conditioning_v2_full"
$Subset = "reports\$Stage\subset_full.jsonl"
$TargetDir = "runs\cache\$Stage\targets"
$Checkpoint = "runs\cache\$Stage\bridge\image_state_conditioning_v2_full_image_translator.pt"
$TrainReport = "runs\cache\$Stage\reports\image_state_conditioning_v2_full_train_report.json"
$TextAnchor = "E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt"
$EmbeddedPython = "E:\ComfyUI_sage\python_embeded\python.exe"
$TrainScript = "scripts\train_image_state_translator.py"
$RenderScript = "scripts\render_image_state_conditioning.py"
$LogDir = Join-Path $Repo "reports\$Stage\logs"
$RenderDir = "runs\images\$Stage\epoch_eval"
$Summary = "reports\$Stage\epoch_eval_summary.json"
$ExpectedShards = 195
$ExpectedCacheWorkers = @("21148", "19420", "10904")

New-Item -ItemType Directory -Force -Path $LogDir, (Join-Path $Repo $RenderDir), (Split-Path (Join-Path $Repo $Summary)) | Out-Null
Set-Location $Repo

while ($true) {
  $aliveWorkers = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*06_cache_targets.py*" -and $_.CommandLine -like "*$Stage*" })
  $shards = @(Get-ChildItem $TargetDir -Filter "*.pt" -ErrorAction SilentlyContinue).Count
  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Add-Content -Path (Join-Path $LogDir "04_watch_train_eval.log") -Value "$stamp cache_shards=$shards workers=$($aliveWorkers.Count)"
  if ($aliveWorkers.Count -eq 0 -and $shards -ge $ExpectedShards) {
    break
  }
  Start-Sleep -Seconds 60
}

& $EmbeddedPython $TrainScript `
  --subset $Subset `
  --targets $TargetDir `
  --out $Checkpoint `
  --text-translator-anchor $TextAnchor `
  --epochs 3 `
  --batch-size 4 `
  --lr 0.0002 `
  --val 512 `
  --save-each-epoch `
  --report $TrainReport `
  *> (Join-Path $LogDir "05_train_3epoch.log")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$renders = @()
foreach ($epoch in 1,2,3) {
  $EpochCheckpoint = "runs\cache\$Stage\bridge\image_state_conditioning_v2_full_image_translator_epoch$epoch.pt"
  foreach ($idx in 0, 93274) {
    $Out = "$RenderDir\epoch${epoch}_idx${idx}.png"
    & $EmbeddedPython $RenderScript `
      --subset $Subset `
      --checkpoint $EpochCheckpoint `
      --idx $idx `
      --out $Out `
      --seed (930001 + $idx + $epoch) `
      --size 512 `
      --steps 16 `
      --cfg 4.5 `
      --unet-dtype default `
      --json `
      *> (Join-Path $LogDir "06_render_epoch${epoch}_idx${idx}.log")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $renders += @{
      epoch = $epoch
      idx = $idx
      output = $Out
      checkpoint = $EpochCheckpoint
    }
  }
}

$Train = Get-Content $TrainReport -Raw | ConvertFrom-Json
[ordered]@{
  stage = $Stage
  train_report = $TrainReport
  checkpoint = $Checkpoint
  history = $Train.history
  renders = $renders
  gpu_policy = @{
    cuda_visible_devices = "0"
    forbidden_gpu = "RTX 5060"
  }
} | ConvertTo-Json -Depth 8 | Set-Content -Path $Summary -Encoding UTF8
