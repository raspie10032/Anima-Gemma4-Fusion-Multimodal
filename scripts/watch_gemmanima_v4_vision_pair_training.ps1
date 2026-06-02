param(
    [int]$RefreshSec = 5
)

$ErrorActionPreference = "SilentlyContinue"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$ReportDir = Join-Path $Repo "reports\gemmanima_v4_vision_pair"
$StatePath = Join-Path $ReportDir "state.json"
$LogDir = Join-Path $ReportDir "logs"
$RunnerLog = Join-Path $LogDir "vision_pair_runner.log"

$rx = [regex]'e(\d+) step (\d+)/(\d+) loss ([\d.]+).*?(?:peakVRAM ([\d.]+)GB)?'
$startStep = $null
$t0 = Get-Date
$lastLog = $null

while ($true) {
    Clear-Host
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "GEMMANIMA v4 vision pair training monitor  $now"
    Write-Host ""

    if (-not (Test-Path $StatePath)) {
        Write-Host "Waiting for state file: $StatePath"
        Start-Sleep -Seconds $RefreshSec
        continue
    }

    $state = Get-Content $StatePath -Raw | ConvertFrom-Json
    $stage = [string]$state.stage
    $status = [string]$state.status
    $log = [string]$state.stdout
    $err = [string]$state.stderr
    $pid = [string]$state.pid

    if ($log -ne $lastLog) {
        $startStep = $null
        $t0 = Get-Date
        $lastLog = $log
    }

    Write-Host ("Stage : {0}" -f $stage)
    Write-Host ("Status: {0}" -f $status)
    if ($pid) { Write-Host ("PID   : {0}" -f $pid) }
    Write-Host ("Log   : {0}" -f $log)
    Write-Host ""

    $latest = $null
    if ($log -and (Test-Path $log)) {
        $latest = Select-String -Path $log -Pattern 'e\d+ step \d+/\d+ loss' -AllMatches | Select-Object -Last 1
    }

    if ($latest) {
        $m = $rx.Match($latest.Line)
        if ($m.Success) {
            $epoch = [int]$m.Groups[1].Value
            $cur = [int]$m.Groups[2].Value
            $tot = [int]$m.Groups[3].Value
            $loss = $m.Groups[4].Value
            $vram = $m.Groups[5].Value
            if ($null -eq $startStep) {
                $startStep = $cur
                $t0 = Get-Date
            }

            $pct = [math]::Round(100.0 * $cur / $tot, 1)
            $elapsed = (Get-Date) - $t0
            $done = $cur - $startStep
            $eta = "?"
            $rate = 0.0
            if ($done -gt 0 -and $elapsed.TotalSeconds -gt 0) {
                $rate = $done / $elapsed.TotalSeconds
                $remain = ($tot - $cur) / $rate
                $eta = [TimeSpan]::FromSeconds([int]$remain).ToString("hh\:mm\:ss")
            }
            $fill = [int]($pct / 2)
            $bar = ('#' * $fill).PadRight(50, '-')

            Write-Host ("[{0}] {1,5}%  epoch {2}  {3}/{4}" -f $bar, $pct, $epoch, $cur, $tot)
            Write-Host ("loss={0}  rate={1:n3} step/s  ETA={2}  peakVRAM={3}GB" -f $loss, $rate, $eta, $vram)
            Write-Host ""
            Write-Host $latest.Line
        }
    } else {
        Write-Host "Waiting for first step line..."
        if ($log -and (Test-Path $log)) {
            Write-Host ""
            Get-Content -Path $log -Tail 12
        }
    }

    Write-Host ""
    Write-Host "GPU:"
    $gpu = & nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null
    if ($gpu) {
        $gpu | ForEach-Object { Write-Host "  $_" }
    } else {
        Write-Host "  nvidia-smi unavailable"
    }

    if ($err -and (Test-Path $err)) {
        $errLines = Get-Content -Path $err -Tail 4
        if ($errLines) {
            Write-Host ""
            Write-Host "stderr tail:"
            $errLines | ForEach-Object { Write-Host "  $_" }
        }
    }

    if (Test-Path $RunnerLog) {
        Write-Host ""
        Write-Host "runner tail:"
        Get-Content -Path $RunnerLog -Tail 4 | ForEach-Object { Write-Host "  $_" }
    }

    Start-Sleep -Seconds $RefreshSec
}
