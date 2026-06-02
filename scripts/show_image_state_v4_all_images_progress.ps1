$ErrorActionPreference = "SilentlyContinue"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$Stage = "image_state_conditioning_v4_all_images"
$ReportDir = Join-Path $Repo "reports\$Stage"
$LogDir = Join-Path $ReportDir "logs"
$RawManifest = Join-Path $ReportDir "raw_ai_images_manifest.jsonl"
$RawStats = Join-Path $ReportDir "raw_ai_images_manifest_stats.json"
$AiImageFolderName = "ai" + [char]0xADF8 + [char]0xB9BC
$RawEmbedRoot = Join-Path (Join-Path "D:\" $AiImageFolderName) "img_embeds_pre_gemmanima"
$Subset = Join-Path $ReportDir "subset_all_images.jsonl"
$TargetDir = Join-Path $Repo "runs\cache\$Stage\targets"
$TrainReport = Join-Path $Repo "runs\cache\$Stage\reports\image_state_conditioning_v4_all_images_train_report.json"

function Count-Lines($Path) {
    if (Test-Path $Path) {
        return (Get-Content -LiteralPath $Path -ReadCount 20000 | ForEach-Object { $_.Count } | Measure-Object -Sum).Sum
    }
    return 0
}

[ordered]@{
    stage = $Stage
    raw_manifest_lines = Count-Lines $RawManifest
    raw_stats_exists = Test-Path $RawStats
    raw_embed_files = (Get-ChildItem -LiteralPath $RawEmbedRoot -Filter *.pt -File | Measure-Object).Count
    subset_lines = Count-Lines $Subset
    target_shards = (Get-ChildItem -LiteralPath $TargetDir -Filter *.pt -File | Measure-Object).Count
    train_report_exists = Test-Path $TrainReport
    runner_log_tail = @(if (Test-Path (Join-Path $LogDir "v4_all_images_runner.log")) { Get-Content (Join-Path $LogDir "v4_all_images_runner.log") -Tail 8 | ForEach-Object { [string]$_ } } else { @() })
    raw_manifest_log_tail = @(if (Test-Path (Join-Path $LogDir "02_raw_manifest.out.log")) { Get-Content (Join-Path $LogDir "02_raw_manifest.out.log") -Tail 8 | ForEach-Object { [string]$_ } } else { @() })
    raw_embed_log_tail = @(if (Test-Path (Join-Path $LogDir "03_raw_image_embeds.out.log")) { Get-Content (Join-Path $LogDir "03_raw_image_embeds.out.log") -Tail 8 | ForEach-Object { [string]$_ } } else { @() })
    raw_embed_gpu0_log_tail = @(if (Test-Path (Join-Path $LogDir "03_raw_image_embeds_gpu0.out.log")) { Get-Content (Join-Path $LogDir "03_raw_image_embeds_gpu0.out.log") -Tail 8 | ForEach-Object { [string]$_ } } else { @() })
    raw_embed_gpu1_log_tail = @(if (Test-Path (Join-Path $LogDir "03_raw_image_embeds_gpu1.out.log")) { Get-Content (Join-Path $LogDir "03_raw_image_embeds_gpu1.out.log") -Tail 8 | ForEach-Object { [string]$_ } } else { @() })
} | ConvertTo-Json -Depth 5
