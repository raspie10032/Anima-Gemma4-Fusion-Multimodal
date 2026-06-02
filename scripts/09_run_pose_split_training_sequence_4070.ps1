param(
    [string]$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal",
    [string]$Log = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\logs\pose_split_sequence\sequence.log"
)

$ErrorActionPreference = "Stop"

function Log($Message) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $Message" | Tee-Object -FilePath $Log -Append
}

function Run-Step($Name, $Script, $Args = @()) {
    Log "START $Name"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Script @Args
    $code = $LASTEXITCODE
    Log "END $Name exit=$code"
    if ($code -ne 0) {
        throw "$Name failed with exit code $code"
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) | Out-Null
Set-Location $Repo

$active = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "08_train_stream_batched.py|06_cache_targets.py|07_cache_gemma_batched.py" })
if ($active.Count -gt 0) {
    $ids = ($active | ForEach-Object { $_.ProcessId }) -join ", "
    throw "refusing to start sequence; active cache/train process id(s): $ids"
}

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Log "pose split sequence started"
Run-Step "prepare-pose-mix-cache" (Join-Path $Repo "scripts\04_prepare_pose_mix_1p53m_v2_153k_4070.ps1")
Run-Step "link-pose-only-cache" (Join-Path $Repo "scripts\07_link_pose_only_cache.ps1")
Run-Step "train-pose-only" (Join-Path $Repo "scripts\08_train_bridge_quality_pose_only_10k_4070.ps1")
Run-Step "train-integrated" (Join-Path $Repo "scripts\06_train_bridge_quality_1p53m_v2_153k_pose10k_4070.ps1")
Log "pose split sequence complete"
