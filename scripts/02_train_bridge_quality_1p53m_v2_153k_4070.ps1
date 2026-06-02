param(
    [int]$Epochs = 1,
    [double]$LearningRate = 0.0002,
    [int]$BatchSize = 2,
    [int]$Accum = 2,
    [int]$ValidationExamples = 2000,
    [int]$PrefetchGb = 48,
    [string]$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal",
    [string]$SwapRoot = "E:\anima_gemma_swap",
    [string]$Python = "E:\ComfyUI_sage\python_embeded\python.exe",
    [string]$TargetDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_1p53m_v2_153k\targets",
    [string]$GemmaDir = "D:\anima_gemma_swap_cache_v2_153k\gemma",
    [string]$ResumeKv = "E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt",
    [string]$Out = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_1p53m_v2_153k\bridge\kv_proj_bridge_quality_1p53m_v2_153k_from_a0p35.pt",
    [string]$Log = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\logs\bridge_quality_1p53m_v2_153k\train_4070.log"
)

$ErrorActionPreference = "Stop"

function Get-ShardNames([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return @()
    }
    return @(Get-ChildItem -LiteralPath $Path -Filter "*.pt" -File | Sort-Object Name | ForEach-Object { $_.Name })
}

$TrainScript = Join-Path $SwapRoot "scripts\core\08_train_stream_batched.py"
if (-not (Test-Path -LiteralPath $TrainScript)) {
    throw "trainer not found: $TrainScript"
}
if (-not (Test-Path -LiteralPath $ResumeKv)) {
    throw "resume adapter not found: $ResumeKv"
}

$activeTraining = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "08_train_stream_batched.py" })
if ($activeTraining.Count -gt 0) {
    $ids = ($activeTraining | ForEach-Object { $_.ProcessId }) -join ", "
    throw "refusing duplicate bridge training; active trainer process id(s): $ids"
}

$targetNames = Get-ShardNames $TargetDir
$gemmaNames = Get-ShardNames $GemmaDir
if ($targetNames.Count -eq 0) {
    throw "target cache is empty; run scripts\01_cache_targets_bridge_quality_1p53m_v2_153k_4070.ps1 first"
}
if ($targetNames.Count -ne $gemmaNames.Count) {
    throw "cache shard count mismatch: target=$($targetNames.Count), gemma=$($gemmaNames.Count)"
}

$missing = @()
$gemmaSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$gemmaNames)
foreach ($name in $targetNames) {
    if (-not $gemmaSet.Contains($name)) {
        $missing += $name
    }
}
if ($missing.Count -gt 0) {
    throw "cache shard name mismatch; first missing gemma shard: $($missing[0])"
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
    --kv-anchor-lambda 0.002 `
    --save-every-shards 10 2>&1 | Tee-Object -FilePath $Log
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference

exit $exitCode
