param(
    [string]$SourceTargetDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_1p53m_v2_153k\targets",
    [string]$SourceGemmaDir = "D:\anima_gemma_swap_cache_bridge_quality_1p53m_v2_153k_pose10k\gemma",
    [string]$PoseOnlyTargetDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_pose_only_10k\targets",
    [string]$PoseOnlyGemmaDir = "D:\anima_gemma_swap_cache_bridge_quality_pose_only_10k\gemma",
    [string]$Output = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\bridge_quality_pose_only_10k_audit.json"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $PoseOnlyTargetDir, $PoseOnlyGemmaDir, (Split-Path -Parent $Output) | Out-Null

function New-CacheLink([string]$Source, [string]$Dest) {
    if (Test-Path -LiteralPath $Dest) {
        return
    }
    try {
        New-Item -ItemType SymbolicLink -Path $Dest -Target $Source -ErrorAction Stop | Out-Null
        return
    } catch {
        try {
            New-Item -ItemType HardLink -Path $Dest -Target $Source -ErrorAction Stop | Out-Null
            return
        } catch {
            Copy-Item -LiteralPath $Source -Destination $Dest -ErrorAction Stop
        }
    }
}

foreach ($source in Get-ChildItem -LiteralPath $SourceTargetDir -Filter "pose_*.pt" -File -ErrorAction SilentlyContinue) {
    $dest = Join-Path $PoseOnlyTargetDir $source.Name
    New-CacheLink $source.FullName $dest
}

foreach ($source in Get-ChildItem -LiteralPath $SourceGemmaDir -Filter "pose_*.pt" -File -ErrorAction SilentlyContinue) {
    $dest = Join-Path $PoseOnlyGemmaDir $source.Name
    New-CacheLink $source.FullName $dest
}

$targetShards = @(Get-ChildItem -LiteralPath $PoseOnlyTargetDir -Filter "*.pt" -File -ErrorAction SilentlyContinue | Sort-Object Name)
$gemmaShards = @(Get-ChildItem -LiteralPath $PoseOnlyGemmaDir -Filter "*.pt" -File -ErrorAction SilentlyContinue | Sort-Object Name)
$targetNames = @($targetShards | ForEach-Object { $_.Name })
$gemmaNames = @($gemmaShards | ForEach-Object { $_.Name })
$gemmaSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$gemmaNames)
$paired = @()
$missingGemma = @()
foreach ($name in $targetNames) {
    if ($gemmaSet.Contains($name)) {
        $paired += $name
    } else {
        $missingGemma += $name
    }
}

$audit = [ordered]@{
    stage = "bridge_quality_pose_only_10k_audit"
    created_local = (Get-Date).ToString("s")
    source_target_dir = $SourceTargetDir
    source_gemma_dir = $SourceGemmaDir
    pose_only_target_dir = $PoseOnlyTargetDir
    pose_only_gemma_dir = $PoseOnlyGemmaDir
    target_shards = $targetShards.Count
    gemma_shards = $gemmaShards.Count
    paired_shards = $paired.Count
    missing_gemma_shards = $missingGemma.Count
    first_missing_gemma = @($missingGemma | Select-Object -First 10)
    ready_for_training = (($targetShards.Count -gt 0) -and ($targetShards.Count -eq $gemmaShards.Count) -and ($paired.Count -eq $targetShards.Count))
}

$audit | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Output -Encoding UTF8
$audit | ConvertTo-Json -Depth 5
