$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$IntervalSeconds = 300

while ($true) {
  try {
    & (Join-Path $RepoRoot "scripts\watch_hiddenstage_pipeline_once.ps1")
  } catch {
    Write-Output "watch error: $($_.Exception.Message)"
  }
  Start-Sleep -Seconds $IntervalSeconds
}
