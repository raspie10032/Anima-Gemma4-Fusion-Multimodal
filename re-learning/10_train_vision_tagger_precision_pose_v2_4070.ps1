param(
    [ValidateSet("safe", "pose", "mixed")]
    [string]$Phase = "safe",
    [double]$Epochs = 0.25,
    [int]$SaveEvery = 500,
    [int]$LogEvery = 20,
    [int]$MaxSteps = 0
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$TrainingRoot = "D:\Projects\training"
$Python = Join-Path $TrainingRoot ".venv\Scripts\python.exe"
$TrainScript = Join-Path $TrainingRoot "v14_train_visual_expand_lora.py"
$LogDir = Join-Path $Repo "re-learning\logs"
$OutDir = Join-Path $TrainingRoot "out\gemmanima_relearning_v1\vision_tagger_precision_pose_v2"

$SafeRoot = "D:\Projects\danbooru_tagger_safe_precision_v2"
$PoseRoot = "D:\Projects\danbooru_tagger_pose_boost_v2"
$MixedRoot = "D:\Projects\danbooru_tagger_mixed_stability_v2"

if ($Phase -eq "safe") {
    $DataRoots = @($SafeRoot)
    $Lr = "0.00008"
    $ProjLr = "0.00002"
    $CatFloor = "pose_action=3.5,appearance=2.5,clothing=2.5,setting=2.5,body_focus=1.0"
}
elseif ($Phase -eq "pose") {
    $DataRoots = @($SafeRoot, $PoseRoot)
    if ($Epochs -eq 0.25) { $Epochs = 0.35 }
    $Lr = "0.00010"
    $ProjLr = "0.00002"
    $CatFloor = "pose_action=6.0,composition=4.0,appearance=2.0,clothing=2.0,setting=2.0,body_focus=0.75"
}
else {
    $DataRoots = @($SafeRoot, $PoseRoot, $MixedRoot)
    $Lr = "0.00006"
    $ProjLr = "0.000015"
    $CatFloor = "pose_action=4.5,composition=3.5,appearance=2.5,clothing=2.5,setting=2.5,body_focus=1.0"
}

New-Item -ItemType Directory -Force -Path $LogDir, $OutDir | Out-Null

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TORCH_CUDA_MEM_FRACTION = "0.88"

if (-not (Test-Path -LiteralPath $Python)) { throw "missing python: $Python" }
if (-not (Test-Path -LiteralPath $TrainScript)) { throw "missing script: $TrainScript" }
foreach ($Root in $DataRoots) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root "manifest_visual_expand.jsonl"))) { throw "missing manifest: $Root" }
    if (-not (Test-Path -LiteralPath (Join-Path $Root "img_embeds_pre\cache_manifest.jsonl"))) { throw "missing cache: $Root\img_embeds_pre\cache_manifest.jsonl" }
}

$OutLog = Join-Path $LogDir ("10_train_precision_pose_v2_{0}.out.log" -f $Phase)
$ErrLog = Join-Path $LogDir ("10_train_precision_pose_v2_{0}.err.log" -f $Phase)
$PidFile = Join-Path $LogDir ("10_train_precision_pose_v2_{0}.pid" -f $Phase)

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match [regex]::Escape($TrainScript) -and
    $_.CommandLine -match [regex]::Escape($OutDir)
}
if ($existing) {
    $pids = @($existing | Select-Object -ExpandProperty ProcessId)
    Set-Content -Encoding ASCII -Path $PidFile -Value ($pids -join ",")
    Write-Host "precision/pose training already running PID=$($pids -join ',')"
    Write-Host "stdout=$OutLog"
    Write-Host "stderr=$ErrLog"
    Write-Host "out=$OutDir"
    return
}

$argsList = @(
    $TrainScript,
    "--data"
)
$argsList += $DataRoots
$argsList += @(
    "--no-nl",
    "--epochs", ([string]$Epochs),
    "--lr", $Lr,
    "--proj-lr", $ProjLr,
    "--lora-r", "8",
    "--lora-alpha", "16",
    "--save-every", ([string]$SaveEvery),
    "--log-every", ([string]$LogEvery),
    "--cat-floor", $CatFloor,
    "--out", $OutDir
)
if ($MaxSteps -gt 0) {
    $argsList += @("--max-steps", ([string]$MaxSteps))
}

$proc = Start-Process `
    -FilePath $Python `
    -ArgumentList $argsList `
    -WorkingDirectory $TrainingRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Encoding ASCII -Path $PidFile -Value $proc.Id
Write-Host "precision/pose training started phase=$Phase PID=$($proc.Id)"
Write-Host "stdout=$OutLog"
Write-Host "stderr=$ErrLog"
Write-Host "out=$OutDir"
