$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$Stage = "image_state_conditioning_v2_full"
$Subset = "reports\$Stage\subset_full.jsonl"
$TargetDir = "runs\cache\$Stage\targets"
$Checkpoint = "runs\cache\$Stage\bridge\image_state_conditioning_v2_full_image_translator.pt"
$TrainReport = "runs\cache\$Stage\reports\image_state_conditioning_v2_full_train_report.json"
$TextAnchor = "E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt"
$EmbeddedPython = "E:\ComfyUI_sage\python_embeded\python.exe"
$TrainScript = "scripts\train_image_state_translator.py"
$RenderScript = "scripts\render_image_state_conditioning.py"
$LogDir = Join-Path $Repo "reports\$Stage\logs"
$RenderDir = "runs\images\$Stage\epoch_eval"
$Summary = "reports\$Stage\epoch_eval_summary.json"

New-Item -ItemType Directory -Force -Path $LogDir, (Join-Path $Repo $RenderDir), (Split-Path (Join-Path $Repo $Summary)) | Out-Null
Set-Location $Repo

$env:CUDA_VISIBLE_DEVICES = "0"
$env:GEMMA_EMBED_ON_GPU = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$BatchSize = "32"
$ImageCacheGb = "56"
$TrainOut = Join-Path $LogDir "07_train_3epoch_b32.out.log"
$TrainErr = Join-Path $LogDir "07_train_3epoch_b32.err.log"
$TrainPid = Join-Path $LogDir "07_train_3epoch_b32.pid"
$RunnerLog = Join-Path $LogDir "07_train_eval_runner_b32.log"

Add-Content -Path $RunnerLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') starting train"

$TrainArgs = @(
    $TrainScript,
    "--subset", $Subset,
    "--targets", $TargetDir,
    "--out", $Checkpoint,
    "--text-translator-anchor", $TextAnchor,
    "--epochs", "3",
    "--batch-size", $BatchSize,
    "--lr", "0.0002",
    "--val", "512",
    "--image-cache-gb", $ImageCacheGb,
    "--save-each-epoch",
    "--report", $TrainReport
)

$TrainProc = Start-Process `
    -FilePath $EmbeddedPython `
    -ArgumentList $TrainArgs `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $TrainOut `
    -RedirectStandardError $TrainErr `
    -WindowStyle Hidden `
    -PassThru
Set-Content -Path $TrainPid -Value $TrainProc.Id
$TrainProc.WaitForExit()
$TrainProc.Refresh()
Add-Content -Path $RunnerLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') train_exit=$($TrainProc.ExitCode)"
if ($TrainProc.ExitCode -ne 0) {
    exit $TrainProc.ExitCode
}

$renders = @()
foreach ($epoch in 1,2,3) {
    $EpochCheckpoint = "runs\cache\$Stage\bridge\image_state_conditioning_v2_full_image_translator_epoch$epoch.pt"
    foreach ($idx in 0, 93274) {
        $Out = "$RenderDir\epoch${epoch}_idx${idx}.png"
        $RenderOut = Join-Path $LogDir "08_render_epoch${epoch}_idx${idx}.out.log"
        $RenderErr = Join-Path $LogDir "08_render_epoch${epoch}_idx${idx}.err.log"
        Add-Content -Path $RunnerLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') render epoch=$epoch idx=$idx"
        $RenderArgs = @(
            $RenderScript,
            "--subset", $Subset,
            "--checkpoint", $EpochCheckpoint,
            "--idx", "$idx",
            "--out", $Out,
            "--seed", "$(930001 + $idx + $epoch)",
            "--size", "512",
            "--steps", "16",
            "--cfg", "4.5",
            "--unet-dtype", "default",
            "--json"
        )
        $RenderProc = Start-Process `
            -FilePath $EmbeddedPython `
            -ArgumentList $RenderArgs `
            -WorkingDirectory $Repo `
            -RedirectStandardOutput $RenderOut `
            -RedirectStandardError $RenderErr `
            -WindowStyle Hidden `
            -PassThru
        $RenderProc.WaitForExit()
        $RenderProc.Refresh()
        Add-Content -Path $RunnerLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') render_exit=$($RenderProc.ExitCode) epoch=$epoch idx=$idx"
        if ($RenderProc.ExitCode -ne 0) {
            exit $RenderProc.ExitCode
        }
        $renders += @{
            epoch = $epoch
            idx = $idx
            output = $Out
            checkpoint = $EpochCheckpoint
        }
    }
}

$Train = Get-Content $TrainReport -Raw | ConvertFrom-Json
[ordered]@{
    stage = $Stage
    train_report = $TrainReport
    checkpoint = $Checkpoint
    history = $Train.history
    renders = $renders
    gpu_policy = @{
        cuda_visible_devices = "0"
        forbidden_gpu = "RTX 5060"
    }
} | ConvertTo-Json -Depth 8 | Set-Content -Path $Summary -Encoding UTF8

Add-Content -Path $RunnerLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') complete"
