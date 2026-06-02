param(
    [string]$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal",
    [string]$SwapRoot = "E:\anima_gemma_swap",
    [string]$Python = "E:\ComfyUI_sage\python_embeded\python.exe",
    [string]$PoseRoot = "D:\Projects\danbooru_pose_alltags_allratings_10",
    [string]$PoseRecords = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\pose_alltags_allratings_10_prompt_records.jsonl",
    [string]$PoseCurriculum = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\pose_alltags_allratings_10_surface_curriculum.json",
    [string]$TargetDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_1p53m_v2_153k\targets",
    [string]$BaseGemmaDir = "D:\anima_gemma_swap_cache_v2_153k\gemma",
    [string]$MixGemmaDir = "D:\anima_gemma_swap_cache_bridge_quality_1p53m_v2_153k_pose10k\gemma",
    [string]$LogDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\logs\bridge_quality_1p53m_v2_153k_pose10k"
)

$ErrorActionPreference = "Stop"

function Assert-NoActiveTargetCache {
    $active = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "06_cache_targets.py" })
    if ($active.Count -gt 0) {
        $ids = ($active | ForEach-Object { $_.ProcessId }) -join ", "
        throw "target cache is already active; wait for it to finish before preparing pose mix. Active PID(s): $ids"
    }
}


if (-not (Test-Path -LiteralPath (Join-Path $PoseRoot "manifest.jsonl"))) {
    throw "pose manifest not found under $PoseRoot"
}

Assert-NoActiveTargetCache
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PoseRecords), (Split-Path -Parent $PoseCurriculum), $TargetDir, $MixGemmaDir, $LogDir | Out-Null

python (Join-Path $Repo "scripts\build_pose_prompt_records.py") `
    --manifest (Join-Path $PoseRoot "manifest.jsonl") `
    --root $PoseRoot `
    --out $PoseRecords `
    --curriculum-out $PoseCurriculum `
    --idx-start 2000000 `
    --weight 3.0 `
    --skip-missing-images

if ($LASTEXITCODE -ne 0) {
    throw "pose prompt conversion failed"
}

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore::FutureWarning"

$CacheTargets = Join-Path $SwapRoot "scripts\core\06_cache_targets.py"
$CacheGemma = Join-Path $SwapRoot "scripts\core\07_cache_gemma_batched.py"
$targetLog = Join-Path $LogDir "pose_target_4070.log"
$gemmaLog = Join-Path $LogDir "pose_gemma_4070.log"

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python -u $CacheTargets `
    --resume `
    --subset $PoseRecords `
    --outdir $TargetDir `
    --shard 2000 `
    --start-idx 2000000 `
    --end-idx 2010000 `
    --shard-prefix "pose" 2>&1 | Tee-Object -FilePath $targetLog
$targetExit = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($targetExit -ne 0) {
    throw "pose target cache failed, see $targetLog"
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python -u $CacheGemma `
    --subset $PoseRecords `
    --target-dir $TargetDir `
    --outdir $MixGemmaDir `
    --patterns "pose_*.pt" `
    --batch-size 8 `
    --resume 2>&1 | Tee-Object -FilePath $gemmaLog
$gemmaExit = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($gemmaExit -ne 0) {
    throw "pose gemma cache failed, see $gemmaLog"
}

& (Join-Path $Repo "scripts\05_audit_pose_mix_1p53m_v2_153k.ps1")
