param(
    [double]$ImageCacheGb = 48.0
)

$ErrorActionPreference = "Stop"
$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
Set-Location $Repo

$LogDir = "reports\image_state_conditioning_v4_all_images\logs"
$TrainLog = Join-Path $Repo "$LogDir\07_train.out.log"
$TrainErr = Join-Path $Repo "$LogDir\07_train.err.log"
$WatcherLog = Join-Path $Repo "$LogDir\07_epoch3_cache48_watcher.out.log"
$WatcherErr = Join-Path $Repo "$LogDir\07_epoch3_cache48_watcher.err.log"
$Epoch2Ckpt = Join-Path $Repo "runs\cache\image_state_conditioning_v4_all_images\bridge\image_state_conditioning_v4_all_images_image_translator_epoch2.pt"
$FinalCkpt = "runs\cache\image_state_conditioning_v4_all_images\bridge\image_state_conditioning_v4_all_images_image_translator.pt"
$Epoch3Report = "runs\cache\image_state_conditioning_v4_all_images\reports\image_state_conditioning_v4_all_images_epoch3_cache48_train_report.json"
$Python = "D:\Projects\training\.venv\Scripts\python.exe"

function LogLine([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $WatcherLog -Value "[$stamp] $Message" -Encoding UTF8
}

try {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $WatcherLog) | Out-Null
    LogLine "watcher started; waiting for epoch2 checkpoint; image_cache_gb=$ImageCacheGb"

    while ($true) {
        $hasCheckpoint = Test-Path -LiteralPath $Epoch2Ckpt
        $hasEpoch2Log = $false
        if (Test-Path -LiteralPath $TrainLog) {
            $tail = Get-Content -LiteralPath $TrainLog -Tail 80
            $hasEpoch2Log = [bool]($tail | Where-Object { $_ -match "epoch 2/3 train_mse=" })
        }
        if ($hasCheckpoint -and $hasEpoch2Log) {
            break
        }
        Start-Sleep -Seconds 2
    }

    $ckptInfo = Get-Item -LiteralPath $Epoch2Ckpt
    if ($ckptInfo.Length -lt 1000000) {
        throw "epoch2 checkpoint exists but looks too small: $($ckptInfo.Length) bytes"
    }
    LogLine "epoch2 checkpoint detected: $Epoch2Ckpt ($($ckptInfo.Length) bytes)"

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    if (Test-Path -LiteralPath $TrainLog) {
        Copy-Item -LiteralPath $TrainLog -Destination (Join-Path $Repo "$LogDir\07_train.pre_epoch3_cache48_$timestamp.out.log") -Force
    }
    if (Test-Path -LiteralPath $TrainErr) {
        Copy-Item -LiteralPath $TrainErr -Destination (Join-Path $Repo "$LogDir\07_train.pre_epoch3_cache48_$timestamp.err.log") -Force
    }

    $oldProcs = Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -like "*scripts\train_image_state_translator.py*" -and
            $_.CommandLine -like "*--epochs 3*" -and
            $_.CommandLine -like "*--image-cache-gb 56*"
        }
    foreach ($proc in $oldProcs) {
        LogLine "stopping old training process pid=$($proc.ProcessId)"
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 3

    $env:CUDA_VISIBLE_DEVICES = "0"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"

    $cmd = @(
        "scripts\train_image_state_translator.py",
        "--subset", "reports\image_state_conditioning_v4_all_images\subset_all_images.jsonl",
        "--targets", "runs\cache\image_state_conditioning_v4_all_images\targets",
        "--out", $FinalCkpt,
        "--text-translator-anchor", "E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt",
        "--init-checkpoint", $Epoch2Ckpt,
        "--epochs", "1",
        "--epoch-offset", "2",
        "--batch-size", "32",
        "--lr", "0.0002",
        "--val", "1024",
        "--image-cache-gb", ([string]$ImageCacheGb),
        "--save-each-epoch",
        "--report", $Epoch3Report
    )

    $epoch3Out = Join-Path $Repo "$LogDir\07_train_epoch3_cache48.out.log"
    $epoch3Err = Join-Path $Repo "$LogDir\07_train_epoch3_cache48.err.log"
    LogLine "starting epoch3 cache-adjusted run"
    $p = Start-Process -FilePath $Python -ArgumentList $cmd -WorkingDirectory $Repo -RedirectStandardOutput $epoch3Out -RedirectStandardError $epoch3Err -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath (Join-Path $Repo "$LogDir\07_train_epoch3_cache48.pid") -Value $p.Id -Encoding ASCII
    LogLine "epoch3 process started pid=$($p.Id); log=$epoch3Out"
} catch {
    Add-Content -LiteralPath $WatcherErr -Value $_.Exception.ToString() -Encoding UTF8
    throw
}
