$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$LogDir = Join-Path $Repo "re-learning\logs"
$Roots = @{
    safe = "D:\Projects\danbooru_tagger_safe_precision_v2"
    pose = "D:\Projects\danbooru_tagger_pose_boost_v2"
    mixed = "D:\Projects\danbooru_tagger_mixed_stability_v2"
}

function Count-Lines($Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    return (Get-Content -LiteralPath $Path | Measure-Object -Line).Lines
}

Write-Host "== tagger precision/pose v2 =="
foreach ($Name in @("safe", "pose", "mixed")) {
    $Root = $Roots[$Name]
    $Manifest = Join-Path $Root "manifest_visual_expand.jsonl"
    $CacheManifest = Join-Path $Root "img_embeds_pre\cache_manifest.jsonl"
    $Total = Count-Lines $Manifest
    $Cached = Count-Lines $CacheManifest
    $Pct = if ($Total -gt 0) { [math]::Round($Cached * 100.0 / $Total, 2) } else { 0 }
    Write-Host ("cache {0}: {1}/{2} ({3}%) root={4}" -f $Name, $Cached, $Total, $Pct, $Root)
}

Write-Host ""
Write-Host "== active cache/train processes =="
Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and (
        $_.CommandLine -match "v02_encode_images.py" -or
        $_.CommandLine -match "v14_train_visual_expand_lora.py" -or
        $_.CommandLine -match "v18_eval_heldout.py"
    )
} | Select-Object ProcessId,ParentProcessId,Name,CommandLine | Format-List

Write-Host "== GPU =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

Write-Host ""
Write-Host "== latest logs =="
foreach ($Pattern in @("09_cache_precision_pose_v2_*.out.log", "13_cache_mixed_precision_pose_v2_5060.out.log", "10_train_precision_pose_v2_*.out.log")) {
    Get-ChildItem -Path $LogDir -Filter $Pattern -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 3 |
        ForEach-Object {
            Write-Host ("--- {0} ---" -f $_.Name)
            Get-Content -LiteralPath $_.FullName -Tail 8
        }
}
