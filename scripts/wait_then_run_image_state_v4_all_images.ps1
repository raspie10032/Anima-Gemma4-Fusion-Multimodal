param(
    [int]$ManifestPid = 0
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$Stage = "image_state_conditioning_v4_all_images"
$LogDir = Join-Path $Repo "reports\$Stage\logs"
$Log = Join-Path $LogDir "v4_wait_then_run.log"
$Stats = Join-Path $Repo "reports\$Stage\raw_ai_images_manifest_stats.json"
$Runner = Join-Path $Repo "scripts\run_image_state_conditioning_v4_all_images_4070.ps1"
$WrapperLog = Join-Path $LogDir "v4_runner_wrapper.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Repo

function Log($Message) {
    Add-Content -Path $Log -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
}

Log "watcher started manifest_pid=$ManifestPid"
if ($ManifestPid -gt 0) {
    while (Get-Process -Id $ManifestPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
    Log "manifest process exited pid=$ManifestPid"
}

if (-not (Test-Path $Stats)) {
    Log "raw manifest stats missing; runner not started"
    exit 2
}

Log "starting v4 runner"
& powershell -NoProfile -ExecutionPolicy Bypass -File $Runner *> $WrapperLog
$exit = $LASTEXITCODE
Log "v4 runner exit=$exit"
exit $exit
