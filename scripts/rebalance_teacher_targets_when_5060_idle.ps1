$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$TargetDir = "E:\anima_gemma_swap\cache_hiddenstage_planner_v2\targets"
$LogDir = "D:\Projects\training\logs"
$PythonExe = "E:\ComfyUI_sage\python_embeded\python.exe"
$CacheScript = "E:\anima_gemma_swap\scripts\core\06_cache_targets.py"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Marker = Join-Path $LogDir "teacher_targets_rebalance_started.marker"

if (Test-Path $Marker) {
  Write-Output "rebalance already started: $Marker"
  exit 0
}

function Count-Shards($Pattern) {
  return (Get-ChildItem -File $TargetDir -Filter $Pattern -ErrorAction SilentlyContinue | Measure-Object).Count
}

$count5060 = Count-Shards "shard_5060_*.pt"
$count4070Original = (Get-ChildItem -File $TargetDir -Filter "shard_????.pt" -ErrorAction SilentlyContinue | Measure-Object).Count

if ($count5060 -lt 29) {
  Write-Output "5060 is not idle yet: $count5060/29 shards"
  exit 0
}

if ($count4070Original -ge 68) {
  Write-Output "4070 Ti SUPER original allocation is already complete: $count4070Original/68"
  exit 0
}

Write-Output "rebalance starting: 5060=$count5060/29, 4070_original=$count4070Original/68"
Set-Content -Path $Marker -Value "$(Get-Date -Format o) 5060=$count5060 4070_original=$count4070Original"

# Stop the original 4070 target worker and any child python still reading the original 4070 subset.
$pidFile = Join-Path $LogDir "teacher_targets_hiddenstage_v2_split_4070_ti_super.pid"
if (Test-Path $pidFile) {
  $workerPid = Get-Content $pidFile -ErrorAction SilentlyContinue
  if ($workerPid) {
    Stop-Process -Id $workerPid -Force -ErrorAction SilentlyContinue
  }
}

$originalSubsetToken = "hiddenstage_multimodal_planner_anima_v2_teacher_subset_4070ti_super_70p.jsonl"
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like "*$originalSubsetToken*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Start-Sleep -Seconds 2

$planJson = python -m gemmanima.cli rebalance-targets --completed-4070-shards $count4070Original --json
$plan = $planJson | ConvertFrom-Json
Write-Output $planJson

$jobs = @(
  @{
    Name = "teacher_targets_rebalance_4070_ti_super"
    Gpu = "0"
    Subset = $plan.subset_4070
    Prefix = "shard_re4070"
  },
  @{
    Name = "teacher_targets_rebalance_5060"
    Gpu = "1"
    Subset = $plan.subset_5060
    Prefix = "shard_5060_re"
  }
)

foreach ($job in $jobs) {
  if (-not (Test-Path $job.Subset)) {
    Write-Output "missing subset for $($job.Name): $($job.Subset)"
    continue
  }
  $rows = (Get-Content $job.Subset | Measure-Object).Count
  if ($rows -le 0) {
    Write-Output "skip $($job.Name): no rows"
    continue
  }
  $out = Join-Path $LogDir ($job.Name + ".out.log")
  $err = Join-Path $LogDir ($job.Name + ".err.log")
  $pidOut = Join-Path $LogDir ($job.Name + ".pid")
  $cmd = "`$env:CUDA_VISIBLE_DEVICES='$($job.Gpu)'; `$env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'; Set-Location '$RepoRoot'; & '$PythonExe' '$CacheScript' --subset '$($job.Subset)' --outdir '$TargetDir' --shard 2000 --shard-prefix '$($job.Prefix)' --resume"
  $p = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $cmd) `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err `
    -WindowStyle Hidden `
    -PassThru
  Set-Content -Path $pidOut -Value $p.Id
  Write-Output "started $($job.Name) rows=$rows pid=$($p.Id)"
}
