param(
    [ValidateSet("safe", "pose", "mixed")]
    [string]$Dataset = "safe",
    [int]$LogEvery = 200
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$TrainingRoot = "D:\Projects\training"
$Python = Join-Path $TrainingRoot ".venv\Scripts\python.exe"
$EncodeScript = Join-Path $TrainingRoot "v02_encode_images.py"
$Manifest = "manifest_visual_expand.jsonl"
$LogDir = Join-Path $Repo "re-learning\logs"

$DataRoots = @{
    safe = "D:\Projects\danbooru_tagger_safe_precision_v2"
    pose = "D:\Projects\danbooru_tagger_pose_boost_v2"
    mixed = "D:\Projects\danbooru_tagger_mixed_stability_v2"
}
$DataRoot = $DataRoots[$Dataset]
$OutLog = Join-Path $LogDir ("09_cache_precision_pose_v2_{0}.out.log" -f $Dataset)
$ErrLog = Join-Path $LogDir ("09_cache_precision_pose_v2_{0}.err.log" -f $Dataset)
$PidFile = Join-Path $LogDir ("09_cache_precision_pose_v2_{0}.pid" -f $Dataset)

New-Item -ItemType Directory -Force -Path $LogDir, (Join-Path $DataRoot "img_embeds_pre") | Out-Null

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if (-not (Test-Path -LiteralPath $Python)) { throw "missing python: $Python" }
if (-not (Test-Path -LiteralPath $EncodeScript)) { throw "missing script: $EncodeScript" }
if (-not (Test-Path -LiteralPath (Join-Path $DataRoot $Manifest))) { throw "missing manifest: $DataRoot\$Manifest; run 08_prepare first" }

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match [regex]::Escape($EncodeScript) -and
    $_.CommandLine -match [regex]::Escape($DataRoot)
}
if ($existing) {
    $pids = @($existing | Select-Object -ExpandProperty ProcessId)
    Set-Content -Encoding ASCII -Path $PidFile -Value ($pids -join ",")
    Write-Host "cache already running dataset=$Dataset PID=$($pids -join ',')"
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
Write-Host "cache started dataset=$Dataset PID=$($proc.Id)"
Write-Host "stdout=$OutLog"
Write-Host "stderr=$ErrLog"
