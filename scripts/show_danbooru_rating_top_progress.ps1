$ErrorActionPreference = "SilentlyContinue"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OutputRoot = "D:\Projects\danbooru_rating_top_200k"
$LogDir = Join-Path $Root "reports\danbooru_rating_top_200k\logs"
$PidFile = Join-Path $LogDir "collector.pid"
$Target = 200000
$Ratings = @("g", "s", "q", "e")

while ($true) {
    Clear-Host
    Write-Host "Danbooru rating top-200k collector"
    Write-Host ("Time: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ("Output: {0}" -f $OutputRoot)
    Write-Host ""

    $PidValue = $null
    $Alive = $false
    if (Test-Path $PidFile) {
        $PidValue = (Get-Content $PidFile -Raw).Trim()
        if ($PidValue) {
            $Alive = [bool](Get-Process -Id ([int]$PidValue) -ErrorAction SilentlyContinue)
        }
    }
    Write-Host ("collector pid={0} alive={1}" -f $PidValue, $Alive)
    Write-Host ""

    foreach ($Rating in $Ratings) {
        $Manifest = Join-Path $OutputRoot ("manifests\rating_{0}.jsonl" -f $Rating)
        $Downloaded = Join-Path $OutputRoot ("manifests\rating_{0}_downloaded.jsonl" -f $Rating)
        $Rejected = Join-Path $OutputRoot ("manifests\rating_{0}_rejected.jsonl" -f $Rating)
        $AcceptedCount = 0
        $DownloadedCount = 0
        $RejectedCount = 0
        if (Test-Path $Manifest) { $AcceptedCount = (Get-Content $Manifest | Measure-Object -Line).Lines }
        if (Test-Path $Downloaded) { $DownloadedCount = (Get-Content $Downloaded | Measure-Object -Line).Lines }
        if (Test-Path $Rejected) { $RejectedCount = (Get-Content $Rejected | Measure-Object -Line).Lines }
        $Pct = [math]::Round(($AcceptedCount / $Target) * 100, 2)
        Write-Host ("rating={0} manifest={1}/{2} ({3}%) downloaded_log={4} rejected={5}" -f $Rating, $AcceptedCount, $Target, $Pct, $DownloadedCount, $RejectedCount)
    }

    Write-Host ""
    $OutLog = Join-Path $LogDir "collector.out.log"
    if (Test-Path $OutLog) {
        Write-Host "Recent collector log:"
        Get-Content $OutLog -Tail 10
    }

    Start-Sleep -Seconds 30
}
