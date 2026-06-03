param(
    [int]$SafeLimit = 30000,
    [int]$MixedLimit = 50000
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

Push-Location $Repo
try {
    & $Python scripts\prepare_tagger_precision_pose_v2.py `
        --safe-limit $SafeLimit `
        --mixed-limit $MixedLimit
}
finally {
    Pop-Location
}
