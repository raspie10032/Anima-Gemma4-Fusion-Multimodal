$ErrorActionPreference = "SilentlyContinue"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$DataRoot = "D:\Projects\danbooru_unified"
$CacheDir = Join-Path $DataRoot "img_embeds_pre"
$OutRoot = "D:\Projects\training\out\gemmanima_relearning_v1"
$LogDir = Join-Path $Repo "re-learning\logs"

Write-Host "GEMMANIMA re-learning status"
Write-Host ""

Write-Host "Dataset:"
Write-Host "  root     $DataRoot"
Write-Host "  manifest $(Join-Path $DataRoot 'manifest_visual_expand.jsonl')"
if (Test-Path -LiteralPath (Join-Path $DataRoot "manifest_visual_expand.jsonl")) {
    $rows = (Get-Content -LiteralPath (Join-Path $DataRoot "manifest_visual_expand.jsonl") -Encoding UTF8 | Measure-Object -Line).Lines
    Write-Host "  rows     $rows"
}

Write-Host ""
Write-Host "Cache:"
Write-Host "  dir      $CacheDir"
if (Test-Path -LiteralPath $CacheDir) {
    $pts = (Get-ChildItem -LiteralPath $CacheDir -Filter "*.pt" -File | Measure-Object).Count
    $mani = Join-Path $CacheDir "cache_manifest.jsonl"
    $maniRows = 0
    if (Test-Path -LiteralPath $mani) {
        $maniRows = (Get-Content -LiteralPath $mani -Encoding UTF8 | Measure-Object -Line).Lines
    }
    Write-Host "  pt files $pts"
    Write-Host "  manifest $maniRows"
} else {
    Write-Host "  missing"
}

Write-Host ""
Write-Host "Outputs:"
if (Test-Path -LiteralPath $OutRoot) {
    Get-ChildItem -LiteralPath $OutRoot -Force | Select-Object Mode,Name,Length,LastWriteTime | Format-Table -AutoSize
} else {
    Write-Host "  missing $OutRoot"
}

Write-Host ""
Write-Host "Running processes:"
Get-Process | Where-Object { $_.ProcessName -match "python|llama" } |
    Select-Object Id,ProcessName,CPU,WorkingSet64,Path | Format-Table -AutoSize

Write-Host ""
Write-Host "GPU:"
& nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits

Write-Host ""
Write-Host "Recent logs:"
if (Test-Path -LiteralPath $LogDir) {
    Get-ChildItem -LiteralPath $LogDir -File | Sort-Object LastWriteTime -Descending | Select-Object -First 8 Name,Length,LastWriteTime | Format-Table -AutoSize
}
