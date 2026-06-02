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
    [string]$BaseTargetDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_1p53m_v2_153k\targets",
    [string]$PoseTargetDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_pose_only_10k\targets",
    [string]$BaseGemmaDir = "D:\anima_gemma_swap_cache_v2_153k\gemma",
    [string]$PoseGemmaDir = "D:\anima_gemma_swap_cache_bridge_quality_pose_only_10k\gemma",
    [string]$PoseCurriculum = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\pose_alltags_allratings_10_surface_curriculum.json",
    [string]$ResumeKv = "E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt",
    [string]$Out = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_1p53m_v2_153k_pose10k\bridge\kv_proj_bridge_quality_1p53m_v2_153k_pose10k_from_a0p35.pt",
    [string]$Log = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\logs\bridge_quality_1p53m_v2_153k_pose10k\train_4070.log"
)

$ErrorActionPreference = "Stop"

$activeTraining = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "08_train_stream_batched.py" })
if ($activeTraining.Count -gt 0) {
    $ids = ($activeTraining | ForEach-Object { $_.ProcessId }) -join ", "
    throw "refusing duplicate bridge training; active trainer process id(s): $ids"
}

$TrainScript = Join-Path $Repo "scripts\train_stream_batched_multidir.py"
foreach ($path in @($TrainScript, $BaseTargetDir, $PoseTargetDir, $BaseGemmaDir, $PoseGemmaDir, $PoseCurriculum, $ResumeKv)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "required path not found: $path"
    }
}

$targetCount = (Get-ChildItem -LiteralPath $BaseTargetDir -Filter "*.pt" -File | Measure-Object).Count
$baseGemmaCount = (Get-ChildItem -LiteralPath $BaseGemmaDir -Filter "*.pt" -File | Measure-Object).Count
$poseGemmaCount = (Get-ChildItem -LiteralPath $PoseGemmaDir -Filter "*.pt" -File | Measure-Object).Count
if ($targetCount -ne ($baseGemmaCount + $poseGemmaCount)) {
    throw "cache shard count mismatch: target=$targetCount gemma=$($baseGemmaCount + $poseGemmaCount)"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out), (Split-Path -Parent $Log) | Out-Null
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore::FutureWarning"

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python -u $TrainScript `
    --targets "$BaseTargetDir" `
    --gemma "$BaseGemmaDir;$PoseGemmaDir" `
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
    --surface-curriculum $PoseCurriculum `
    --save-every-shards 10 2>&1 | Tee-Object -FilePath $Log
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference

exit $exitCode
