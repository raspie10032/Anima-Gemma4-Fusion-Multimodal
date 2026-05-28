$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = "D:\Projects\training\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$jobs = @(
  @{
    Name = "gemma_hidden_cache_split_4070_ti_super"
    Script = Join-Path $RepoRoot "scripts\run_gemma_hidden_cache_split_4070_ti_super.ps1"
  },
  @{
    Name = "gemma_hidden_cache_split_5060"
    Script = Join-Path $RepoRoot "scripts\run_gemma_hidden_cache_split_5060.ps1"
  }
)

foreach ($job in $jobs) {
  $pidFile = Join-Path $LogDir ($job.Name + ".pid")
  $existingPid = if (Test-Path $pidFile) { Get-Content $pidFile -ErrorAction SilentlyContinue } else { $null }
  if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
    Write-Output "$($job.Name) already running pid=$existingPid"
    continue
  }
  $out = Join-Path $LogDir ($job.Name + ".out.log")
  $err = Join-Path $LogDir ($job.Name + ".err.log")
  $p = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $job.Script) `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err `
    -WindowStyle Hidden `
    -PassThru
  Set-Content -Path $pidFile -Value $p.Id
  Write-Output "started $($job.Name) pid=$($p.Id)"
}
