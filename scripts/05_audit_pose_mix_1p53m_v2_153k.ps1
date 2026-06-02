param(
    [string]$TargetDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_1p53m_v2_153k\targets",
    [string]$MixGemmaDir = "D:\anima_gemma_swap_cache_bridge_quality_1p53m_v2_153k_pose10k\gemma",
    [string]$PoseRecords = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\pose_alltags_allratings_10_prompt_records.jsonl",
    [string]$Output = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\bridge_quality_1p53m_v2_153k_pose10k_audit.json"
)

$ErrorActionPreference = "Stop"

function Get-Shards([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return @()
    }
    return @(Get-ChildItem -LiteralPath $Path -Filter "*.pt" -File | Sort-Object Name)
}

function Get-LineCount([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    return (Get-Content -LiteralPath $Path | Measure-Object -Line).Lines
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null
$targetShards = Get-Shards $TargetDir
$gemmaShards = Get-Shards $MixGemmaDir
$targetNames = @($targetShards | ForEach-Object { $_.Name })
$gemmaNames = @($gemmaShards | ForEach-Object { $_.Name })
$gemmaSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$gemmaNames)
$targetSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$targetNames)

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

$extraGemma = @()
foreach ($name in $gemmaNames) {
    if (-not $targetSet.Contains($name)) {
        $extraGemma += $name
    }
}

$audit = [ordered]@{
    stage = "bridge_quality_1p53m_v2_153k_pose10k_audit"
    created_local = (Get-Date).ToString("s")
    pose_records = $PoseRecords
    pose_rows = Get-LineCount $PoseRecords
    target_dir = $TargetDir
    mix_gemma_dir = $MixGemmaDir
    target_shards = $targetShards.Count
    gemma_shards = $gemmaShards.Count
    paired_shards = $paired.Count
    missing_gemma_shards = $missingGemma.Count
    extra_gemma_shards = $extraGemma.Count
    first_missing_gemma = @($missingGemma | Select-Object -First 10)
    first_extra_gemma = @($extraGemma | Select-Object -First 10)
    ready_for_training = (($targetShards.Count -gt 0) -and ($targetShards.Count -eq $gemmaShards.Count) -and ($paired.Count -eq $targetShards.Count))
}

$audit | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Output -Encoding UTF8
$audit | ConvertTo-Json -Depth 4
