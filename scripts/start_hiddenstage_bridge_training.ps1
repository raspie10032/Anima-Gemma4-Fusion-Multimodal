$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = "D:\Projects\training\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Name = "hiddenstage_bridge_train_4070_ti_super"
$Script = Join-Path $RepoRoot "scripts\run_hiddenstage_bridge_train_4070_ti_super.ps1"
$pidFile = Join-Path $LogDir ($Name + ".pid")
$existingPid = if (Test-Path $pidFile) { Get-Content $pidFile -ErrorAction SilentlyContinue } else { $null }
if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
  Write-Output "$Name already running pid=$existingPid"
  exit 0
}

$out = Join-Path $LogDir ($Name + ".out.log")
$err = Join-Path $LogDir ($Name + ".err.log")
$p = Start-Process -FilePath "powershell.exe" `
  -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Script) `
  -WorkingDirectory $RepoRoot `
  -RedirectStandardOutput $out `
  -RedirectStandardError $err `
  -WindowStyle Hidden `
  -PassThru
Set-Content -Path $pidFile -Value $p.Id
Write-Output "started $Name pid=$($p.Id)"
