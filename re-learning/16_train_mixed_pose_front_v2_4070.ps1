param(
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
$DataRoot = "D:\Projects\danbooru_tagger_mixed_pose_front_v2"
$WarmStart = Join-Path $TrainingRoot "out\gemmanima_relearning_v1\vision_tagger_pose_front_v2"
$OutDir = Join-Path $TrainingRoot "out\gemmanima_relearning_v1\vision_tagger_mixed_pose_front_v2"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TORCH_CUDA_MEM_FRACTION = "0.88"

if (-not (Test-Path -LiteralPath $Python)) { throw "missing python: $Python" }
if (-not (Test-Path -LiteralPath $TrainScript)) { throw "missing script: $TrainScript" }
if (-not (Test-Path -LiteralPath (Join-Path $DataRoot "manifest_visual_expand.jsonl"))) { throw "missing manifest: $DataRoot" }
if (-not (Test-Path -LiteralPath (Join-Path $DataRoot "img_embeds_pre\cache_manifest.jsonl"))) { throw "missing cache: $DataRoot\img_embeds_pre\cache_manifest.jsonl" }
if (-not (Test-Path -LiteralPath (Join-Path $WarmStart "adapter_model.safetensors"))) { throw "missing warm-start adapter: $WarmStart" }

if (-not (Test-Path -LiteralPath $OutDir)) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Copy-Item -LiteralPath (Join-Path $WarmStart "*") -Destination $OutDir -Recurse -Force
    Write-Host "warm-start copied: $WarmStart -> $OutDir"
}

$OutLog = Join-Path $LogDir "16_train_mixed_pose_front_v2.out.log"
$ErrLog = Join-Path $LogDir "16_train_mixed_pose_front_v2.err.log"
$PidFile = Join-Path $LogDir "16_train_mixed_pose_front_v2.pid"

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match [regex]::Escape($TrainScript) -and
    $_.CommandLine -match [regex]::Escape($OutDir)
}
if ($existing) {
    $pids = @($existing | Select-Object -ExpandProperty ProcessId)
    Set-Content -Encoding ASCII -Path $PidFile -Value ($pids -join ",")
    Write-Host "mixed pose-front training already running PID=$($pids -join ',')"
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
    "--lr", "0.00005",
    "--proj-lr", "0.00001",
    "--lora-r", "8",
    "--lora-alpha", "16",
    "--save-every", ([string]$SaveEvery),
    "--log-every", ([string]$LogEvery),
    "--cat-floor", "pose_action=8.0,composition=3.0,appearance=1.5,clothing=1.5,setting=1.5,body_focus=0.75",
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
Write-Host "mixed pose-front training started PID=$($proc.Id)"
Write-Host "stdout=$OutLog"
Write-Host "stderr=$ErrLog"
Write-Host "out=$OutDir"
