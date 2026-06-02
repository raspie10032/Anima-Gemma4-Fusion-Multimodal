param(
    [string]$Prompt = "draw a bright forest anime illustration",
    [int]$Size = 512,
    [int]$Steps = 12,
    [double]$Cfg = 4.5,
    [int]$Seed = 424242
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

python -m gemmanima.cli run $Prompt --renderer external-script --size $Size --steps $Steps --cfg $Cfg --seed $Seed --json
