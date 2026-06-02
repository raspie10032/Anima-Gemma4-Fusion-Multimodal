$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$Stage = "image_state_conditioning_v4_all_images"
$ReportDir = Join-Path $Repo "reports\$Stage"
$CacheRoot = Join-Path $Repo "runs\cache\$Stage"
$LogDir = Join-Path $ReportDir "logs"
$AiImageFolderName = "ai" + [char]0xADF8 + [char]0xB9BC
$RawRoot = Join-Path "D:\" $AiImageFolderName
$RawEmbedRoot = Join-Path $RawRoot "img_embeds_pre_gemmanima"
$RootG = "D:\Projects\danbooru_data_set"
$RootETop = "D:\Projects\danbooru_data_set_e_top"
$DanbooruManifest = Join-Path $ReportDir "hiddenstage_multimodal_planner_anima_v2_danbooru.jsonl"
$RawManifest = Join-Path $ReportDir "raw_ai_images_manifest.jsonl"
$RawStats = Join-Path $ReportDir "raw_ai_images_manifest_stats.json"
$CombinedManifest = Join-Path $ReportDir "hiddenstage_multimodal_planner_anima_v2_all_images.jsonl"
$Subset = Join-Path $ReportDir "subset_all_images.jsonl"
$TargetDir = Join-Path $CacheRoot "targets"
$Checkpoint = Join-Path $CacheRoot "bridge\image_state_conditioning_v4_all_images_image_translator.pt"
$TrainReport = Join-Path $CacheRoot "reports\image_state_conditioning_v4_all_images_train_report.json"
$TextAnchor = "E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt"
$InitCheckpoint = Join-Path $Repo "runs\cache\image_state_conditioning_v2_full\bridge\image_state_conditioning_v2_full_image_translator.pt"
$EmbeddedPython = "E:\ComfyUI_sage\python_embeded\python.exe"
$TrainingPython = "D:\Projects\training\.venv\Scripts\python.exe"
$MultimodalDatasetScript = "E:\anima_gemma_swap\scripts\core\hiddenstage_multimodal_dataset.py"
$TargetScript = "E:\anima_gemma_swap\scripts\core\06_cache_targets.py"
$TrainScript = Join-Path $Repo "scripts\train_image_state_translator.py"

New-Item -ItemType Directory -Force -Path $ReportDir, $CacheRoot, $LogDir, $TargetDir, (Split-Path $Checkpoint), (Split-Path $TrainReport), $RawEmbedRoot | Out-Null
Set-Location $Repo

$env:CUDA_VISIBLE_DEVICES = "0"
$env:GEMMA_EMBED_ON_GPU = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

function Log($Message) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Message"
    $line | Tee-Object -FilePath (Join-Path $LogDir "v4_all_images_runner.log") -Append
}

function Run-Step($Name, $Exe, $ArgsList, $OutLog, $ErrLog) {
    Log "START $Name"
    $proc = Start-Process `
        -FilePath $Exe `
        -ArgumentList $ArgsList `
        -WorkingDirectory $Repo `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    Set-Content -Path (Join-Path $LogDir "$Name.pid") -Value $proc.Id
    $proc.Refresh()
    $exitCode = $proc.ExitCode
    if ($null -eq $exitCode) { $exitCode = 1 }
    Log "END $Name exit=$exitCode"
    if ($exitCode -ne 0) { exit $exitCode }
}

function Start-EmbedWorker($Name, $GpuIndex, $ShardIndex, $OutLog, $ErrLog) {
    $cacheManifest = Join-Path $RawEmbedRoot "cache_manifest_gpu$GpuIndex.jsonl"
    $workerScript = @"
`$ErrorActionPreference = "Stop"
`$env:CUDA_VISIBLE_DEVICES = "$GpuIndex"
`$env:GEMMA_EMBED_ON_GPU = "1"
`$env:PYTHONUTF8 = "1"
`$env:PYTHONIOENCODING = "utf-8"
`$env:PYTHONUNBUFFERED = "1"
`$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
& "$TrainingPython" "scripts\cache_image_embed_pre.py" "--manifest" "$RawManifest" "--resume" "--log-every" "500" "--cache-manifest" "$cacheManifest" "--num-shards" "2" "--shard-index" "$ShardIndex"
exit `$LASTEXITCODE
"@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($workerScript))
    $proc = Start-Process `
        -FilePath "powershell" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded) `
        -WorkingDirectory $Repo `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -Path (Join-Path $LogDir "$Name.pid") -Value $proc.Id
    return $proc
}

function Run-RawImageEmbedsDual() {
    Log "START 03_raw_image_embeds_dual"
    $gpu0 = Start-EmbedWorker `
        "03_raw_image_embeds_gpu0" `
        0 `
        0 `
        (Join-Path $LogDir "03_raw_image_embeds_gpu0.out.log") `
        (Join-Path $LogDir "03_raw_image_embeds_gpu0.err.log")
    $gpu1 = Start-EmbedWorker `
        "03_raw_image_embeds_gpu1" `
        1 `
        1 `
        (Join-Path $LogDir "03_raw_image_embeds_gpu1.out.log") `
        (Join-Path $LogDir "03_raw_image_embeds_gpu1.err.log")
    Log "03_raw_image_embeds_dual pids gpu0=$($gpu0.Id) gpu1=$($gpu1.Id)"
    Wait-Process -Id $gpu0.Id, $gpu1.Id
    $gpu0.Refresh()
    $gpu1.Refresh()
    $exit0 = $gpu0.ExitCode
    $exit1 = $gpu1.ExitCode
    if ($null -eq $exit0) { $exit0 = 1 }
    if ($null -eq $exit1) { $exit1 = 1 }
    Log "END 03_raw_image_embeds_dual exit_gpu0=$exit0 exit_gpu1=$exit1"
    if ($exit0 -ne 0 -or $exit1 -ne 0) { exit 1 }
}

if (-not (Test-Path $DanbooruManifest)) {
    Run-Step "01_danbooru_manifest" $EmbeddedPython @(
        $MultimodalDatasetScript,
        "--data", $RootG, $RootETop,
        "--out", $DanbooruManifest,
        "--limit", "0",
        "--prompt-style", "anima_v2"
    ) (Join-Path $LogDir "01_danbooru_manifest.out.log") (Join-Path $LogDir "01_danbooru_manifest.err.log")
} else {
    Log "SKIP 01_danbooru_manifest exists=$DanbooruManifest"
}

if (-not (Test-Path $RawStats)) {
    Run-Step "02_raw_manifest" $EmbeddedPython @(
        "scripts\build_raw_image_manifest.py",
        "--data", $RawRoot,
        "--out", $RawManifest,
        "--embed-root", $RawEmbedRoot,
        "--limit", "0",
        "--stats", $RawStats
    ) (Join-Path $LogDir "02_raw_manifest.out.log") (Join-Path $LogDir "02_raw_manifest.err.log")
} else {
    Log "SKIP 02_raw_manifest exists=$RawStats"
}

$rawManifestLineCount = 0
if (Test-Path $RawManifest) {
    $rawManifestLineCount = (Get-Content -LiteralPath $RawManifest -ReadCount 10000 | ForEach-Object { $_.Count } | Measure-Object -Sum).Sum
}
$rawEmbedFileCount = (Get-ChildItem -LiteralPath $RawEmbedRoot -Filter "*.pt" -File -ErrorAction SilentlyContinue | Measure-Object).Count
if ($rawManifestLineCount -gt 0 -and $rawEmbedFileCount -ge $rawManifestLineCount) {
    Log "SKIP 03_raw_image_embeds complete raw_embed_files=$rawEmbedFileCount raw_manifest_lines=$rawManifestLineCount"
} else {
    Run-Step "03_raw_image_embeds" $TrainingPython @(
        "scripts\cache_image_embed_pre.py",
        "--manifest", $RawManifest,
        "--resume",
        "--log-every", "500",
        "--cache-manifest", (Join-Path $RawEmbedRoot "cache_manifest.jsonl")
    ) (Join-Path $LogDir "03_raw_image_embeds.out.log") (Join-Path $LogDir "03_raw_image_embeds.err.log")
}

Log "START 04_combine_manifests"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$writer = New-Object System.IO.StreamWriter($CombinedManifest, $false, $utf8NoBom)
try {
    foreach ($manifest in @($DanbooruManifest, $RawManifest)) {
        Get-Content -LiteralPath $manifest -Encoding UTF8 -ReadCount 10000 | ForEach-Object {
            foreach ($line in $_) {
                $writer.WriteLine($line)
            }
        }
    }
} finally {
    $writer.Close()
}
Log "END 04_combine_manifests combined=$CombinedManifest"

Run-Step "05_subset" $TrainingPython @(
    "-m", "gemmanima.cli", "image-state-conditioning-subset",
    "--source-manifest", $CombinedManifest,
    "--output", $Subset,
    "--limit", "0",
    "--json"
) (Join-Path $LogDir "05_subset.out.log") (Join-Path $LogDir "05_subset.err.log")

Run-Step "06_cache_targets" $EmbeddedPython @(
    $TargetScript,
    "--subset", $Subset,
    "--outdir", $TargetDir,
    "--shard", "1000",
    "--resume"
) (Join-Path $LogDir "06_cache_targets.out.log") (Join-Path $LogDir "06_cache_targets.err.log")

Run-Step "07_train" $TrainingPython @(
    $TrainScript,
    "--subset", $Subset,
    "--targets", $TargetDir,
    "--out", $Checkpoint,
    "--text-translator-anchor", $TextAnchor,
    "--init-checkpoint", $InitCheckpoint,
    "--epochs", "3",
    "--batch-size", "32",
    "--lr", "0.0002",
    "--val", "1024",
    "--image-cache-gb", "56",
    "--save-each-epoch",
    "--report", $TrainReport
) (Join-Path $LogDir "07_train.out.log") (Join-Path $LogDir "07_train.err.log")

Log "COMPLETE $Stage checkpoint=$Checkpoint report=$TrainReport"
