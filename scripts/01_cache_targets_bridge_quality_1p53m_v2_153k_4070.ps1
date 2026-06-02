param(
    [string]$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal",
    [string]$SwapRoot = "E:\anima_gemma_swap",
    [string]$Python = "E:\ComfyUI_sage\python_embeded\python.exe",
    [string]$Subset = "E:\anima_gemma_swap\archive\old_runs\archive_unused_20260524\old_v2_153k_reference\subset_1p53m_v2_10pct_153000.jsonl",
    [string]$TargetDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_1p53m_v2_153k\targets",
    [string]$LogDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\logs\bridge_quality_1p53m_v2_153k"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Subset)) {
    throw "subset not found: $Subset"
}

$CacheScript = Join-Path $SwapRoot "scripts\core\06_cache_targets.py"
if (-not (Test-Path -LiteralPath $CacheScript)) {
    throw "target cache script not found: $CacheScript"
}

New-Item -ItemType Directory -Force -Path $TargetDir, $LogDir | Out-Null
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore::FutureWarning"

$ranges = @(
    @{ Name = "g0a"; Start = 0; End = 40800 },
    @{ Name = "g0b"; Start = 40800; End = 81600 },
    @{ Name = "g0c"; Start = 81600; End = 122400 },
    @{ Name = "g1"; Start = 122400; End = 153000 }
)

foreach ($range in $ranges) {
    $log = Join-Path $LogDir ("target_{0}.log" -f $range.Name)
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -u $CacheScript `
        --resume `
        --subset $Subset `
        --outdir $TargetDir `
        --shard 2000 `
        --start-idx $range.Start `
        --end-idx $range.End `
        --shard-prefix $range.Name 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference

    if ($exitCode -ne 0) {
        throw "target cache failed for $($range.Name), see $log"
    }
}

& (Join-Path $Repo "scripts\00_audit_bridge_quality_1p53m_v2_153k.ps1")
