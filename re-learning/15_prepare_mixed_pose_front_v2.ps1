param(
    [string]$SourceRoot = "D:\Projects\danbooru_tagger_mixed_stability_v2",
    [string]$OutputRoot = "D:\Projects\danbooru_tagger_mixed_pose_front_v2",
    [switch]$SyncCache,
    [int]$SubjectLead = 8
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

$Script = Join-Path $Repo "scripts\prepare_mixed_pose_front_v2.py"
if (-not (Test-Path -LiteralPath $Script)) { throw "missing script: $Script" }

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$argsList = @(
    $Script,
    "--source-root", $SourceRoot,
    "--output-root", $OutputRoot,
    "--subject-lead", ([string]$SubjectLead)
)
if ($SyncCache) {
    $argsList += "--sync-cache"
}

& $Python @argsList
