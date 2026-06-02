param(
    [string]$Prompt = "draw a bright forest anime illustration"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

python -m gemmanima.cli run $Prompt --renderer dry-run --json
