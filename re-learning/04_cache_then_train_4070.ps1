param(
    [int]$TotalRows = 127300,
    [int]$PollSeconds = 60,
    [double]$TrainEpochs = 1.0
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$TrainingRoot = "D:\Projects\training"
$DataRoot = "D:\Projects\danbooru_unified"
$CacheScript = Join-Path $TrainingRoot "v02_encode_images.py"
$TrainScript = Join-Path $Repo "re-learning\02_train_vision_tagger_4070.ps1"
$CacheManifest = Join-Path $DataRoot "img_embeds_pre\cache_manifest.jsonl"
$LogDir = Join-Path $Repo "re-learning\logs"
$WatcherLog = Join-Path $LogDir "04_cache_then_train_4070.out.log"
$WatcherPid = Join-Path $LogDir "04_cache_then_train_4070.pid"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Content -Encoding ASCII -Path $WatcherPid -Value $PID

function Write-WatcherLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Encoding UTF8 -Path $WatcherLog -Value "[$stamp] $Message"
}

function Get-ManifestRows {
    if (-not (Test-Path -LiteralPath $CacheManifest)) { return 0 }
    return (Get-Content -LiteralPath $CacheManifest -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
}

function Get-CacheProcess {
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match [regex]::Escape($CacheScript) -and
        $_.CommandLine -match [regex]::Escape($DataRoot)
    }
}

function Get-TrainProcess {
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match "v14_train_visual_expand_lora.py" -and
        $_.CommandLine -match [regex]::Escape($DataRoot)
    }
}

Write-WatcherLog "watcher started total_rows=$TotalRows poll=${PollSeconds}s train_epochs=$TrainEpochs"

while ($true) {
    $rows = Get-ManifestRows
    $cacheProc = @(Get-CacheProcess)
    $trainProc = @(Get-TrainProcess)
    Write-WatcherLog "cache_rows=$rows/$TotalRows cache_pids=$($cacheProc.ProcessId -join ',') train_pids=$($trainProc.ProcessId -join ',')"

    if ($trainProc.Count -gt 0) {
        Write-WatcherLog "training already running; watcher complete"
        break
    }

    if ($rows -ge $TotalRows) {
        Write-WatcherLog "cache complete; starting vision tagger training"
        & powershell -ExecutionPolicy Bypass -File $TrainScript -Epochs $TrainEpochs
        Write-WatcherLog "train launch command returned"
        break
    }

    if ($cacheProc.Count -eq 0) {
        Write-WatcherLog "cache process is not running before completion; rows=$rows/$TotalRows"
        throw "cache process stopped before completion: $rows/$TotalRows"
    }

    Start-Sleep -Seconds $PollSeconds
}
