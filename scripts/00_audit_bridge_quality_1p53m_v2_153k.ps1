param(
    [string]$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal",
    [string]$Subset = "E:\anima_gemma_swap\archive\old_runs\archive_unused_20260524\old_v2_153k_reference\subset_1p53m_v2_10pct_153000.jsonl",
    [string]$TargetDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_1p53m_v2_153k\targets",
    [string]$GemmaDir = "D:\anima_gemma_swap_cache_v2_153k\gemma",
    [string]$Output = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\bridge_quality_1p53m_v2_153k_audit.json"
)

$ErrorActionPreference = "Stop"

function Get-LineCount([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    return (Get-Content -LiteralPath $Path | Measure-Object -Line).Lines
}

function Get-Shards([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return @()
    }
    return @(Get-ChildItem -LiteralPath $Path -Filter "*.pt" -File | Sort-Object Name)
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null

$targetShards = Get-Shards $TargetDir
$gemmaShards = Get-Shards $GemmaDir
$targetNames = @($targetShards | ForEach-Object { $_.Name })
$gemmaNames = @($gemmaShards | ForEach-Object { $_.Name })
$targetSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$targetNames)
$gemmaSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$gemmaNames)

$paired = @()
foreach ($name in $targetNames) {
    if ($gemmaSet.Contains($name)) {
        $paired += $name
    }
}

$missingGemma = @()
foreach ($name in $targetNames) {
    if (-not $gemmaSet.Contains($name)) {
        $missingGemma += $name
    }
}

$missingTarget = @()
foreach ($name in $gemmaNames) {
    if (-not $targetSet.Contains($name)) {
        $missingTarget += $name
    }
}

$audit = [ordered]@{
    stage = "bridge_quality_1p53m_v2_153k_audit"
    created_local = (Get-Date).ToString("s")
    subset = $Subset
    subset_exists = (Test-Path -LiteralPath $Subset)
    subset_rows = Get-LineCount $Subset
    target_dir = $TargetDir
    gemma_dir = $GemmaDir
    target_shards = $targetShards.Count
    gemma_shards = $gemmaShards.Count
    paired_shards = $paired.Count
    target_size_bytes = (($targetShards | Measure-Object Length -Sum).Sum + 0)
    gemma_size_bytes = (($gemmaShards | Measure-Object Length -Sum).Sum + 0)
    missing_gemma_shards = $missingGemma.Count
    missing_target_shards = $missingTarget.Count
    first_missing_gemma = @($missingGemma | Select-Object -First 10)
    first_missing_target = @($missingTarget | Select-Object -First 10)
    expected_policy = @{
        cuda_visible_devices = "0"
        allowed_gpu = "RTX 4070 Ti SUPER"
        disallowed_gpu = "RTX 5060"
    }
    ready_for_training = (($targetShards.Count -gt 0) -and ($targetShards.Count -eq $gemmaShards.Count) -and ($paired.Count -eq $targetShards.Count))
}

$audit | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Output -Encoding UTF8
$audit | ConvertTo-Json -Depth 6
