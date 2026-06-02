$ErrorActionPreference = "SilentlyContinue"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$Stage = "image_state_conditioning_v2_full"
$TargetDir = Join-Path $Repo "runs\cache\$Stage\targets"
$LogDir = Join-Path $Repo "reports\$Stage\logs"
$TrainReport = Join-Path $Repo "runs\cache\$Stage\reports\image_state_conditioning_v2_full_train_report.json"
$Summary = Join-Path $Repo "reports\$Stage\epoch_eval_summary.json"
$ExpectedShards = 195
$ExpectedEpochs = 3
$Prefixes = @("w0", "w1", "w2")

function Get-CacheProcessCount {
  @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*06_cache_targets.py*" -and $_.CommandLine -like "*$Stage*"
  }).Count
}

function Get-TrainProcessCount {
  @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*train_image_state_translator.py*" -and $_.CommandLine -like "*$Stage*"
  }).Count
}

function Get-TrainProcesses {
  @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*train_image_state_translator.py*" -and $_.CommandLine -like "*$Stage*"
  })
}

function Get-GpuRows {
  $rows = @()
  try {
    $raw = @(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>$null)
    foreach ($line in $raw) {
      $parts = @($line -split "," | ForEach-Object { $_.Trim() })
      if ($parts.Count -ge 6) {
        $rows += [pscustomobject]@{
          index = $parts[0]
          name = $parts[1]
          util = $parts[2]
          mem_used = $parts[3]
          mem_total = $parts[4]
          power = $parts[5]
        }
      }
    }
  } catch {
  }
  return $rows
}

function Get-GpuAppRows {
  $rows = @()
  try {
    $raw = @(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null)
    foreach ($line in $raw) {
      $parts = @($line -split "," | ForEach-Object { $_.Trim() })
      if ($parts.Count -ge 3) {
        $rows += [pscustomobject]@{
          pid = $parts[0]
          name = $parts[1]
          mem = $parts[2]
        }
      }
    }
  } catch {
  }
  return $rows
}

function Get-LatestTrainLog {
  $batch32 = Join-Path $LogDir "07_train_3epoch_b32.out.log"
  if (Test-Path $batch32) {
    return $batch32
  }
  $batch192 = Join-Path $LogDir "07_train_3epoch_b192.out.log"
  if (Test-Path $batch192) {
    return $batch192
  }
  $batch256 = Join-Path $LogDir "07_train_3epoch_b256.out.log"
  if (Test-Path $batch256) {
    return $batch256
  }
  $batch96 = Join-Path $LogDir "07_train_3epoch_b96.out.log"
  if (Test-Path $batch96) {
    return $batch96
  }
  $batch12 = Join-Path $LogDir "07_train_3epoch_b12.out.log"
  if (Test-Path $batch12) {
    return $batch12
  }
  $preferred = Join-Path $LogDir "07_train_3epoch.out.log"
  if (Test-Path $preferred) {
    return $preferred
  }
  return Join-Path $LogDir "05_train_3epoch.log"
}

function Get-TrainingProgress($TrainLog, $TrainProcesses) {
  $result = [ordered]@{
    epoch = 0
    shard = 0
    total_shards = $ExpectedShards
    completed_units = 0
    total_units = $ExpectedEpochs * $ExpectedShards
    percent = 0.0
    eta = "estimating"
    elapsed = "estimating"
    latest_line = ""
  }
  if (-not (Test-Path $TrainLog)) {
    return $result
  }

  $lines = @(Get-Content $TrainLog -Tail 80)
  $latest = $null
  foreach ($line in $lines) {
    if ($line -match "epoch\s+(\d+)\s+shard\s+(\d+)/(\d+)") {
      $latest = @{
        epoch = [int]$Matches[1]
        shard = [int]$Matches[2]
        total = [int]$Matches[3]
        line = $line
      }
    } elseif ($line -match "epoch\s+(\d+)/(\d+)") {
      $epochDone = [int]$Matches[1]
      $latest = @{
        epoch = $epochDone
        shard = $ExpectedShards
        total = $ExpectedShards
        line = $line
      }
    }
  }

  if ($latest) {
    $epoch = $latest.epoch
    $shard = $latest.shard
    $completed = ([math]::Max(0, $epoch - 1) * $ExpectedShards) + $shard
    $totalUnits = $ExpectedEpochs * $ExpectedShards
    $result.epoch = $epoch
    $result.shard = $shard
    $result.total_shards = $latest.total
    $result.completed_units = $completed
    $result.total_units = $totalUnits
    $result.percent = [math]::Round(($completed / [math]::Max(1, $totalUnits)) * 100, 1)
    $result.latest_line = $latest.line

    $startTime = $null
    $firstProc = $TrainProcesses | Select-Object -First 1
    if ($firstProc -and $firstProc.CreationDate) {
      $startTime = [Management.ManagementDateTimeConverter]::ToDateTime($firstProc.CreationDate)
    }
    if (-not $startTime) {
      $trainFile = Get-Item $TrainLog
      $startTime = $trainFile.CreationTime
    }
    $elapsedSeconds = ((Get-Date) - $startTime).TotalSeconds
    if ($completed -gt 0 -and $elapsedSeconds -gt 0) {
      $remaining = [math]::Max(0, $totalUnits - $completed)
      $etaSeconds = ($elapsedSeconds / $completed) * $remaining
      $result.elapsed = Format-TimeSpanShort $elapsedSeconds
      $result.eta = Format-TimeSpanShort $etaSeconds
    }
  }
  return $result
}

function Format-TimeSpanShort([double]$Seconds) {
  if ($Seconds -le 0 -or [double]::IsNaN($Seconds) -or [double]::IsInfinity($Seconds)) {
    return "estimating"
  }
  $ts = [TimeSpan]::FromSeconds($Seconds)
  if ($ts.TotalHours -ge 1) {
    return "{0:0}h {1:00}m" -f [math]::Floor($ts.TotalHours), $ts.Minutes
  }
  return "{0:0}m {1:00}s" -f [math]::Floor($ts.TotalMinutes), $ts.Seconds
}

while ($true) {
  Clear-Host
  $now = Get-Date
  Write-Host "GEMMANIMA image-state v2_full progress" -ForegroundColor Cyan
  Write-Host "Time: $($now.ToString('yyyy-MM-dd HH:mm:ss'))"
  Write-Host "Stage: $Stage"
  Write-Host ""

  $allFiles = @(Get-ChildItem $TargetDir -Filter "*.pt" -File)
  $total = $allFiles.Count
  $percent = [math]::Min(100, [math]::Round(($total / $ExpectedShards) * 100, 1))
  Write-Host "Cache target shards: $total / $ExpectedShards ($percent%)"

  foreach ($prefix in $Prefixes) {
    $files = @($allFiles | Where-Object { $_.Name -like "$prefix*.pt" } | Sort-Object LastWriteTime)
    $latest = $files | Select-Object -Last 1
    $latestText = if ($latest) { "$($latest.Name) @ $($latest.LastWriteTime.ToString('HH:mm:ss'))" } else { "-" }
    Write-Host ("  {0}: {1,3} shards  latest {2}" -f $prefix, $files.Count, $latestText)
  }

  $cacheWorkers = Get-CacheProcessCount
  $trainProcesses = Get-TrainProcesses
  $trainWorkers = @($trainProcesses).Count
  Write-Host ""
  Write-Host "Processes: cache_workers=$cacheWorkers train_workers=$trainWorkers"
  if ($trainWorkers -gt 0) {
    Write-Host "  train_pids=$((@($trainProcesses) | ForEach-Object { $_.ProcessId }) -join ',')"
  }

  Write-Host ""
  Write-Host "GPU status:" -ForegroundColor Yellow
  $gpuRows = @(Get-GpuRows)
  if ($gpuRows.Count -eq 0) {
    Write-Host "  nvidia-smi unavailable"
  } else {
    foreach ($gpu in $gpuRows) {
      $mark = if ($gpu.index -eq "0") { "target" } elseif ($gpu.name -like "*5060*") { "avoid" } else { "" }
      Write-Host ("  gpu{0} {1} [{2}] util={3}% vram={4}/{5} MiB power={6} W" -f $gpu.index, $gpu.name, $mark, $gpu.util, $gpu.mem_used, $gpu.mem_total, $gpu.power)
    }
  }
  $gpuApps = @(Get-GpuAppRows)
  if ($gpuApps.Count -gt 0) {
    $trainPidSet = @{}
    foreach ($proc in @($trainProcesses)) { $trainPidSet[[string]$proc.ProcessId] = $true }
    foreach ($app in $gpuApps) {
      if ($trainPidSet.ContainsKey([string]$app.pid)) {
        Write-Host ("  training gpu app: pid={0} mem={1} MiB {2}" -f $app.pid, $app.mem, $app.name)
      }
    }
  }

  if ($total -ge 2) {
    $ordered = $allFiles | Sort-Object LastWriteTime
    $first = $ordered | Select-Object -First 1
    $last = $ordered | Select-Object -Last 1
    $elapsed = ($last.LastWriteTime - $first.LastWriteTime).TotalSeconds
    $completedBetween = [math]::Max(1, $total - 1)
    $secPerShardAcrossAll = $elapsed / $completedBetween
    $remaining = [math]::Max(0, $ExpectedShards - $total)
    $eta = Format-TimeSpanShort ($remaining * $secPerShardAcrossAll)
    Write-Host "Cache ETA: ~$eta"
  } else {
    Write-Host "Cache ETA: estimating"
  }

  Write-Host ""
  if (Test-Path $TrainReport) {
    try {
      $report = Get-Content $TrainReport -Raw | ConvertFrom-Json
      Write-Host "Train report:" -ForegroundColor Green
      Write-Host "  train_mse=$($report.train_mse) val_mse=$($report.val_mse)"
      if ($report.history) {
        foreach ($h in $report.history) {
          Write-Host ("  epoch {0}: train_mse={1} val_mse={2}" -f $h.epoch, $h.train_mse, $h.val_mse)
        }
      }
    } catch {
      Write-Host "Train report exists but is not readable yet."
    }
  } elseif ($trainWorkers -gt 0) {
    Write-Host "Training: running"
    $trainLog = Get-LatestTrainLog
    $progress = Get-TrainingProgress $trainLog $trainProcesses
    if ($progress.completed_units -gt 0) {
      Write-Host ("  progress: epoch {0}/{1}, shard {2}/{3}, overall {4}% ({5}/{6})" -f $progress.epoch, $ExpectedEpochs, $progress.shard, $progress.total_shards, $progress.percent, $progress.completed_units, $progress.total_units)
      Write-Host ("  elapsed: {0}  eta: ~{1}" -f $progress.elapsed, $progress.eta)
      Write-Host ("  latest: {0}" -f $progress.latest_line)
    } else {
      Write-Host "  progress: waiting for first shard log"
    }
    if (Test-Path $trainLog) {
      Write-Host ""
      Write-Host "Recent train log: $trainLog"
      Get-Content $trainLog -Tail 10
    }
  } else {
    Write-Host "Training: waiting for cache completion"
  }

  Write-Host ""
  if (Test-Path $Summary) {
    Write-Host "Epoch render summary ready: $Summary" -ForegroundColor Green
  } else {
    $renderDir = Join-Path $Repo "runs\images\$Stage\epoch_eval"
    $renders = @(Get-ChildItem $renderDir -Filter "*.png" -File)
    Write-Host "Epoch smoke renders: $($renders.Count) / 6"
  }

  Write-Host ""
  Write-Host "Press Ctrl+C to close this progress terminal." -ForegroundColor DarkGray
  Start-Sleep -Seconds 30
}
