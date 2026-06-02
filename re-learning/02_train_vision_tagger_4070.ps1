param(
    [double]$Epochs = 0.5,
    [int]$SaveEvery = 500,
    [int]$LogEvery = 20,
    [int]$MaxSteps = 0
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$TrainingRoot = "D:\Projects\training"
$Python = Join-Path $TrainingRoot ".venv\Scripts\python.exe"
$TrainScript = Join-Path $TrainingRoot "v14_train_visual_expand_lora.py"
$DataRoot = "D:\Projects\danbooru_unified"
$CacheManifest = Join-Path $DataRoot "img_embeds_pre\cache_manifest.jsonl"
$OutDir = Join-Path $TrainingRoot "out\gemmanima_relearning_v1\vision_tagger_clean_v1"
$LogDir = Join-Path $Repo "re-learning\logs"
$OutLog = Join-Path $LogDir "02_train_vision_tagger_4070.out.log"
$ErrLog = Join-Path $LogDir "02_train_vision_tagger_4070.err.log"
$PidFile = Join-Path $LogDir "02_train_vision_tagger_4070.pid"

New-Item -ItemType Directory -Force -Path $LogDir, $OutDir | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if (-not (Test-Path -LiteralPath $Python)) { throw "missing python: $Python" }
if (-not (Test-Path -LiteralPath $TrainScript)) { throw "missing script: $TrainScript" }
if (-not (Test-Path -LiteralPath $CacheManifest)) { throw "cache missing; run 01_cache_images_4070.ps1 first: $CacheManifest" }

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match [regex]::Escape($TrainScript) -and
    $_.CommandLine -match [regex]::Escape($DataRoot)
}
if ($existing) {
    $pids = @($existing | Select-Object -ExpandProperty ProcessId)
    Set-Content -Encoding ASCII -Path $PidFile -Value ($pids -join ",")
    Write-Host "vision tagger training already running PID=$($pids -join ',')"
    Write-Host "stdout=$OutLog"
    Write-Host "stderr=$ErrLog"
    Write-Host "out=$OutDir"
    return
}

$argsList = @(
    $TrainScript,
    "--data", $DataRoot,
    "--no-nl",
    "--epochs", ([string]$Epochs),
    "--lr", "0.0002",
    "--proj-lr", "0.00005",
    "--save-every", ([string]$SaveEvery),
    "--log-every", ([string]$LogEvery),
    "--cat-floor", "appearance=3.5,pose_action=2.5,clothing=2.5,setting=2.5",
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
Write-Host "vision tagger training started PID=$($proc.Id)"
Write-Host "stdout=$OutLog"
Write-Host "stderr=$ErrLog"
Write-Host "out=$OutDir"
