param(
    [int]$Epochs = 3,
    [double]$LearningRate = 0.00015,
    [int]$BatchSize = 2,
    [int]$Accum = 2,
    [int]$ValidationExamples = 512,
    [int]$PrefetchGb = 16,
    [string]$SwapRoot = "E:\anima_gemma_swap",
    [string]$Python = "E:\ComfyUI_sage\python_embeded\python.exe",
    [string]$TargetDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_pose_only_10k\targets",
    [string]$GemmaDir = "D:\anima_gemma_swap_cache_bridge_quality_pose_only_10k\gemma",
    [string]$ResumeKv = "E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt",
    [string]$Out = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_pose_only_10k\bridge\kv_proj_bridge_quality_pose_only_10k_from_a0p35.pt",
    [string]$Log = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\logs\bridge_quality_pose_only_10k\train_4070.log"
)

$ErrorActionPreference = "Stop"

$activeTraining = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "08_train_stream_batched.py" })
if ($activeTraining.Count -gt 0) {
    $ids = ($activeTraining | ForEach-Object { $_.ProcessId }) -join ", "
    throw "refusing duplicate bridge training; active trainer process id(s): $ids"
}

$TrainScript = Join-Path $SwapRoot "scripts\core\08_train_stream_batched.py"
foreach ($path in @($TrainScript, $TargetDir, $GemmaDir, $ResumeKv)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "required path not found: $path"
    }
}

$targetCount = (Get-ChildItem -LiteralPath $TargetDir -Filter "*.pt" -File | Measure-Object).Count
$gemmaCount = (Get-ChildItem -LiteralPath $GemmaDir -Filter "*.pt" -File | Measure-Object).Count
if ($targetCount -le 0 -or $targetCount -ne $gemmaCount) {
    throw "cache shard count mismatch: target=$targetCount gemma=$gemmaCount"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out), (Split-Path -Parent $Log) | Out-Null
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore::FutureWarning"

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python -u $TrainScript `
    --targets $TargetDir `
    --gemma $GemmaDir `
    --out $Out `
    --epochs $Epochs `
    --lr $LearningRate `
    --batch-size $BatchSize `
    --accum $Accum `
    --val $ValidationExamples `
    --prefetch-gb $PrefetchGb `
    --resume-kv $ResumeKv `
    --kv-anchor $ResumeKv `
    --kv-anchor-lambda 0.001 `
    --save-every-shards 2 2>&1 | Tee-Object -FilePath $Log
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference

exit $exitCode
