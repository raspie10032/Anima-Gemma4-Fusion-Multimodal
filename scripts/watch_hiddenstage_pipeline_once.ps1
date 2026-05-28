$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$StatePath = "D:\Projects\training\logs\hiddenstage_pipeline_state.json"

Set-Location $RepoRoot
$statusJson = python -m gemmanima.cli pipeline-status --output $StatePath --json
$status = $statusJson | ConvertFrom-Json

Write-Output "next_action=$($status.next_action)"
Write-Output "target_shards=$($status.target_completion.shard_count)/$($status.target_completion.expected.total_shards)"
Write-Output "gemma_pairing=$($status.gemma_pairing.paired_shards)/$($status.gemma_pairing.target_shards)"

if (($status.target_completion."5060_shards" -ge $status.target_completion.expected."5060_shards") -and (-not $status.target_completion.complete)) {
  & (Join-Path $RepoRoot "scripts\rebalance_teacher_targets_when_5060_idle.ps1")
} elseif ($status.next_action -eq "start_gemma_hidden_cache") {
  & (Join-Path $RepoRoot "scripts\start_gemma_cache_split.ps1")
} elseif ($status.next_action -eq "start_hiddenstage_bridge_training") {
  & (Join-Path $RepoRoot "scripts\start_hiddenstage_bridge_training.ps1")
}
