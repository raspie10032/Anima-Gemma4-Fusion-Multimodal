$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OutputRoot = "D:\Projects\danbooru_rating_top_200k"
$LogDir = Join-Path $Root "reports\danbooru_rating_top_200k\logs"
$CredentialFile = "C:\Users\seine\Desktop\dandooru.txt"
$SkipManifest = Join-Path $Root "reports\image_state_conditioning_v2_full\subset_full.jsonl"
$Script = Join-Path $Root "scripts\danbooru_rating_top_collector.py"
$Python = "python"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$OutLog = Join-Path $LogDir "collector.out.log"
$ErrLog = Join-Path $LogDir "collector.err.log"
$PidFile = Join-Path $LogDir "collector.pid"

$Args = @(
    $Script,
    "--credential-file", $CredentialFile,
    "--output-root", $OutputRoot,
    "--ratings", "g", "s", "q", "e",
    "--target-per-rating", "200000",
    "--skip-manifest", $SkipManifest,
    "--api-sleep", "1.0",
    "--download-sleep", "0.2",
    "--image-workers", "3",
    "--request-timeout", "30",
    "--request-attempts", "0"
)

$Proc = Start-Process `
    -FilePath $Python `
    -ArgumentList $Args `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $PidFile -Value $Proc.Id

[pscustomobject]@{
    pid = $Proc.Id
    output_root = $OutputRoot
    stdout = $OutLog
    stderr = $ErrLog
    pid_file = $PidFile
} | ConvertTo-Json
