param(
    [int]$LogEvery = 200
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$TrainingRoot = "D:\Projects\training"
$Python = Join-Path $TrainingRoot ".venv\Scripts\python.exe"
$EncodeScript = Join-Path $TrainingRoot "v02_encode_images.py"
$DataRoot = "D:\Projects\danbooru_unified"
$Manifest = "manifest_visual_expand.jsonl"
$LogDir = Join-Path $Repo "re-learning\logs"
$OutLog = Join-Path $LogDir "01_cache_images_4070.out.log"
$ErrLog = Join-Path $LogDir "01_cache_images_4070.err.log"
$PidFile = Join-Path $LogDir "01_cache_images_4070.pid"

New-Item -ItemType Directory -Force -Path $LogDir, (Join-Path $DataRoot "img_embeds_pre") | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if (-not (Test-Path -LiteralPath $Python)) { throw "missing python: $Python" }
if (-not (Test-Path -LiteralPath $EncodeScript)) { throw "missing script: $EncodeScript" }
if (-not (Test-Path -LiteralPath (Join-Path $DataRoot $Manifest))) { throw "missing manifest: $DataRoot\$Manifest" }

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match [regex]::Escape($EncodeScript) -and
    $_.CommandLine -match [regex]::Escape($DataRoot)
}
if ($existing) {
    $pids = @($existing | Select-Object -ExpandProperty ProcessId)
    Set-Content -Encoding ASCII -Path $PidFile -Value ($pids -join ",")
    Write-Host "cache already running PID=$($pids -join ',')"
    Write-Host "stdout=$OutLog"
    Write-Host "stderr=$ErrLog"
    return
}

$argsList = @(
    $EncodeScript,
    "--data", $DataRoot,
    "--manifest", $Manifest,
    "--limit", "0",
    "--log-every", ([string]$LogEvery)
)

$proc = Start-Process `
    -FilePath $Python `
    -ArgumentList $argsList `
    -WorkingDirectory $TrainingRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Encoding ASCII -Path $PidFile -Value $proc.Id
Write-Host "cache started PID=$($proc.Id)"
Write-Host "stdout=$OutLog"
Write-Host "stderr=$ErrLog"
