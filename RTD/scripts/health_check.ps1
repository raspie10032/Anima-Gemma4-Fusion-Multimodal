$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

Write-Host "[RTD] Repo: $RepoRoot"
Write-Host "[RTD] Python:"
python --version

Write-Host "[RTD] Core tests:"
python -m pytest tests/test_tipo_runtime.py tests/test_api.py tests/test_server_gui.py -q

Write-Host "[RTD] Renderer backends:"
python -m gemmanima.cli renderer-backends --json

Write-Host "[RTD] Real render health:"
python -m gemmanima.cli real-render-health --json

Write-Host "[RTD] GUI command:"
python -m gemmanima.cli gui-command --json
