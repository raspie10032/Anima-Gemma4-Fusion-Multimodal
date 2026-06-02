param(
    [double]$TaggerEpochs = 0.5,
    [double]$UnderstanderEpochs = 0.25,
    [int]$TaggerMaxSteps = 0,
    [int]$UnderstanderMaxSteps = 0,
    [int]$SaveEvery = 500,
    [int]$LogEvery = 20,
    [switch]$SkipTagger,
    [switch]$SkipUnderstander
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$TrainingRoot = "D:\Projects\training"
$Python = Join-Path $TrainingRoot ".venv\Scripts\python.exe"
$TrainScript = Join-Path $TrainingRoot "v14_train_visual_expand_lora.py"
$DataRoot = "D:\Projects\danbooru_data_set"
$DataRootTop = "D:\Projects\danbooru_data_set_e_top"
$PlannerManifest = Join-Path $Repo "reports\image_state_conditioning_v4_all_images\hiddenstage_multimodal_planner_anima_v2_all_images.jsonl"

$ReportDir = Join-Path $Repo "reports\gemmanima_v4_vision_pair"
$LogDir = Join-Path $ReportDir "logs"
$TaggerOut = Join-Path $TrainingRoot "out\gemmanima_v4_vision_tagger"
$UnderstanderOut = Join-Path $TrainingRoot "out\gemmanima_v4_image_understander"

New-Item -ItemType Directory -Force -Path $ReportDir, $LogDir, $TaggerOut, $UnderstanderOut | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

function Write-RunnerLog($Message) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Message"
    $line | Tee-Object -FilePath (Join-Path $LogDir "vision_pair_runner.log") -Append
}

function Assert-Path($Path, $Kind) {
    if (-not (Test-Path $Path)) {
        throw "$Kind missing: $Path"
    }
}

function Write-State($Stage, $Status, $Extra = @{}) {
    $state = [ordered]@{
        timestamp = (Get-Date).ToString("o")
        stage = $Stage
        status = $Status
        tagger_out = $TaggerOut
        understander_out = $UnderstanderOut
    }
    foreach ($key in $Extra.Keys) {
        $state[$key] = $Extra[$key]
    }
    $state | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -Path (Join-Path $ReportDir "state.json")
}

function Invoke-Stage($Name, $ArgsList) {
    $outLog = Join-Path $LogDir "$Name.out.log"
    $errLog = Join-Path $LogDir "$Name.err.log"
    $pidFile = Join-Path $LogDir "$Name.pid"
    Write-RunnerLog "START $Name"
    Write-State $Name "starting" @{ stdout = $outLog; stderr = $errLog }

    $proc = Start-Process `
        -FilePath $Python `
        -ArgumentList $ArgsList `
        -WorkingDirectory $TrainingRoot `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -Encoding ASCII -Path $pidFile -Value $proc.Id
    Write-State $Name "running" @{ pid = $proc.Id; stdout = $outLog; stderr = $errLog }

    Wait-Process -Id $proc.Id
    $proc.Refresh()
    $exitCode = $proc.ExitCode
    if ($null -eq $exitCode) { $exitCode = 1 }

    Write-RunnerLog "END $Name exit=$exitCode"
    if ($exitCode -ne 0) {
        Write-State $Name "failed" @{ pid = $proc.Id; exit_code = $exitCode; stdout = $outLog; stderr = $errLog }
        exit $exitCode
    }
    Write-State $Name "completed" @{ pid = $proc.Id; exit_code = $exitCode; stdout = $outLog; stderr = $errLog }
}

Assert-Path $Python "training python"
Assert-Path $TrainScript "training script"
Assert-Path (Join-Path $DataRoot "manifest_visual_expand.jsonl") "main visual manifest"
Assert-Path (Join-Path $DataRootTop "manifest_visual_expand.jsonl") "top visual manifest"
Assert-Path (Join-Path $DataRoot "img_embeds_pre\cache_manifest.jsonl") "main pre-projector cache"
Assert-Path (Join-Path $DataRootTop "img_embeds_pre\cache_manifest.jsonl") "top pre-projector cache"
Assert-Path $PlannerManifest "planner manifest"

Write-RunnerLog "vision pair training runner ready"
Write-State "runner" "ready"

if (-not $SkipTagger) {
    $taggerArgs = @(
        $TrainScript,
        "--data", $DataRoot, $DataRootTop,
        "--no-nl",
        "--epochs", ([string]$TaggerEpochs),
        "--lr", "0.0002",
        "--proj-lr", "0.00005",
        "--save-every", ([string]$SaveEvery),
        "--log-every", ([string]$LogEvery),
        "--cat-floor", "appearance=3.5,pose_action=2.5,clothing=2.5,setting=2.5",
        "--out", $TaggerOut
    )
    if ($TaggerMaxSteps -gt 0) {
        $taggerArgs += @("--max-steps", ([string]$TaggerMaxSteps))
    }
    Invoke-Stage "01_vision_tagger" $taggerArgs
}

if (-not $SkipUnderstander) {
    $understanderArgs = @(
        $TrainScript,
        "--planner-manifest", $PlannerManifest,
        "--epochs", ([string]$UnderstanderEpochs),
        "--lr", "0.00005",
        "--proj-lr", "0.00001",
        "--tag-alpha", "0",
        "--save-every", ([string]$SaveEvery),
        "--log-every", ([string]$LogEvery),
        "--out", $UnderstanderOut
    )
    if ($UnderstanderMaxSteps -gt 0) {
        $understanderArgs += @("--max-steps", ([string]$UnderstanderMaxSteps))
    }
    Invoke-Stage "02_image_understander" $understanderArgs
}

Write-RunnerLog "vision pair training runner completed"
Write-State "runner" "completed"
