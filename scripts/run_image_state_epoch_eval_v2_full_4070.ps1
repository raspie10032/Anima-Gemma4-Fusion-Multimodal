$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$Stage = "image_state_conditioning_v2_full"
$Subset = "reports\$Stage\subset_full.jsonl"
$TrainReport = "runs\cache\$Stage\reports\image_state_conditioning_v2_full_train_report.json"
$Checkpoint = "runs\cache\$Stage\bridge\image_state_conditioning_v2_full_image_translator.pt"
$EmbeddedPython = "E:\ComfyUI_sage\python_embeded\python.exe"
$RenderScript = "scripts\render_image_state_conditioning.py"
$LogDir = Join-Path $Repo "reports\$Stage\logs"
$RenderDir = "runs\images\$Stage\epoch_eval"
$Summary = "reports\$Stage\epoch_eval_summary.json"
$RunnerLog = Join-Path $LogDir "09_epoch_eval_runner.log"

New-Item -ItemType Directory -Force -Path $LogDir, (Join-Path $Repo $RenderDir), (Split-Path (Join-Path $Repo $Summary)) | Out-Null
Set-Location $Repo

$env:CUDA_VISIBLE_DEVICES = "0"
$env:GEMMA_EMBED_ON_GPU = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

Add-Content -Path $RunnerLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') starting epoch eval"

$renders = @()
foreach ($epoch in 1,2,3) {
    $EpochCheckpoint = "runs\cache\$Stage\bridge\image_state_conditioning_v2_full_image_translator_epoch$epoch.pt"
    foreach ($idx in 0, 93274) {
        $Out = "$RenderDir\epoch${epoch}_idx${idx}.png"
        $RenderOut = Join-Path $LogDir "09_render_epoch${epoch}_idx${idx}.out.log"
        $RenderErr = Join-Path $LogDir "09_render_epoch${epoch}_idx${idx}.err.log"
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
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $EmbeddedPython @RenderArgs 1> $RenderOut 2> $RenderErr
        $RenderExit = $LASTEXITCODE
        $ErrorActionPreference = $PreviousErrorActionPreference
        Add-Content -Path $RunnerLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') render_exit=$RenderExit epoch=$epoch idx=$idx"
        if ($RenderExit -ne 0) {
            exit $RenderExit
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
