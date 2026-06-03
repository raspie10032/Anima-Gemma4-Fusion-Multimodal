param(
    [string]$SourceRoot = "D:\Projects\danbooru_tagger_mixed_stability_v2",
    [string]$OutputRoot = "D:\Projects\danbooru_tagger_mixed_pose_front_v2",
    [int]$PollSeconds = 60,
    [int]$MinCached = 30000,
    [double]$Epochs = 0.25
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$LogDir = Join-Path $Repo "re-learning\logs"
$WatchLog = Join-Path $LogDir "18_cache_then_train_mixed_pose_front_v2.out.log"
$PidFile = Join-Path $LogDir "18_cache_then_train_mixed_pose_front_v2.pid"
$PrepareScript = Join-Path $Repo "re-learning\15_prepare_mixed_pose_front_v2.ps1"
$TrainScript = Join-Path $Repo "re-learning\16_train_mixed_pose_front_v2_4070.ps1"
$CacheManifest = Join-Path $SourceRoot "img_embeds_pre\cache_manifest.jsonl"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Content -Encoding ASCII -Path $PidFile -Value $PID

function Write-WatchLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Encoding UTF8 -Path $WatchLog -Value "[$stamp] $Message"
    Write-Host "[$stamp] $Message"
}

function Get-CacheCount {
    if (-not (Test-Path -LiteralPath $CacheManifest)) { return 0 }
    return (Get-Content -LiteralPath $CacheManifest -Encoding UTF8 | Measure-Object -Line).Lines
}

function Get-SourceCacheProcesses {
    $escapedSource = [regex]::Escape($SourceRoot)
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match "v02_encode_images.py" -and
        $_.CommandLine -match $escapedSource
    }
}

function Get-MixedPoseFrontTraining {
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match "v14_train_visual_expand_lora.py" -and
        $_.CommandLine -match "vision_tagger_mixed_pose_front_v2"
    }
}

Write-WatchLog "watcher started source=$SourceRoot output=$OutputRoot min_cached=$MinCached"
while ($true) {
    $cacheCount = Get-CacheCount
    $cacheProcesses = @(Get-SourceCacheProcesses)
    $trainProcesses = @(Get-MixedPoseFrontTraining)
    if ($trainProcesses.Count -gt 0) {
        $pids = @($trainProcesses | Select-Object -ExpandProperty ProcessId)
        Write-WatchLog "training already running PID=$($pids -join ','); watcher exiting"
        exit 0
    }
    if ($cacheProcesses.Count -eq 0) {
        Write-WatchLog "cache process finished or absent; source cache rows=$cacheCount"
        if ($cacheCount -lt $MinCached) {
            throw "source cache rows $cacheCount below minimum $MinCached; refusing to train"
        }
        Write-WatchLog "final sync begin"
        powershell -ExecutionPolicy Bypass -File $PrepareScript -SourceRoot $SourceRoot -OutputRoot $OutputRoot -SyncCache
        if ($LASTEXITCODE -ne 0) { throw "final sync failed with exit code $LASTEXITCODE" }
        Write-WatchLog "training launch begin"
        powershell -ExecutionPolicy Bypass -File $TrainScript -Epochs $Epochs
        if ($LASTEXITCODE -ne 0) { throw "training launch failed with exit code $LASTEXITCODE" }
        Write-WatchLog "training launch requested; watcher exiting"
        exit 0
    }
    $pids = @($cacheProcesses | Select-Object -ExpandProperty ProcessId)
    Write-WatchLog "cache alive PID=$($pids -join ',') rows=$cacheCount; sleep ${PollSeconds}s"
    Start-Sleep -Seconds $PollSeconds
}
