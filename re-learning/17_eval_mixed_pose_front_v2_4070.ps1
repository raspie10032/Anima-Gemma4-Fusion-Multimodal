param(
    [int]$Limit = 0,
    [double]$Temp = 0.2,
    [int]$MaxNew = 96
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$TrainingRoot = "D:\Projects\training"
$Python = Join-Path $TrainingRoot ".venv\Scripts\python.exe"
$MergeScript = Join-Path $TrainingRoot "v15_merge_visual_expand.py"
$EvalScript = Join-Path $TrainingRoot "v18_eval_heldout.py"
$AdapterDir = Join-Path $TrainingRoot "out\gemmanima_relearning_v1\vision_tagger_mixed_pose_front_v2"
$MergedDir = Join-Path $TrainingRoot "out\gemmanima_relearning_v1\merged_vision_tagger_mixed_pose_front_v2"
$EvalRoot = Join-Path $Repo "re-learning\eval"
$PoseEvalManifest = "tagger_precision_pose_v2_poseheavy_manifest.jsonl"
$SafetyEvalManifest = "tagger_mixed_pose_front_v2_safety_manifest.jsonl"
$PoseEvalOut = Join-Path $EvalRoot "vision_tagger_mixed_pose_front_v2_poseheavy_v18.jsonl"
$SafetyEvalOut = Join-Path $EvalRoot "vision_tagger_mixed_pose_front_v2_safety_v18.jsonl"

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TORCH_CUDA_MEM_FRACTION = "0.88"

if (-not (Test-Path -LiteralPath $Python)) { throw "missing python: $Python" }
if (-not (Test-Path -LiteralPath $MergeScript)) { throw "missing merge script: $MergeScript" }
if (-not (Test-Path -LiteralPath $EvalScript)) { throw "missing eval script: $EvalScript" }
if (-not (Test-Path -LiteralPath (Join-Path $AdapterDir "adapter_model.safetensors"))) {
    if (-not (Test-Path -LiteralPath (Join-Path $AdapterDir "ckpt\adapter.pt"))) {
        throw "missing adapter or ckpt: $AdapterDir"
    }
    & $Python $MergeScript `
        --ckpt (Join-Path $AdapterDir "ckpt") `
        --out $MergedDir `
        --lora-r 8 `
        --lora-alpha 16
}
else {
    & $Python $MergeScript `
        --adapter $AdapterDir `
        --out $MergedDir `
        --lora-r 8 `
        --lora-alpha 16
}

function Invoke-Eval {
    param(
        [string]$Manifest,
        [string]$Out
    )
    $argsList = @(
        $EvalScript,
        "--backend", "transformers",
        "--data", $EvalRoot,
        "--manifest", $Manifest,
        "--model", $MergedDir,
        "--temp", ([string]$Temp),
        "--max-new", ([string]$MaxNew),
        "--out", $Out
    )
    if ($Limit -gt 0) {
        $argsList += @("--limit", ([string]$Limit))
    }
    & $Python @argsList
    Write-Host "eval_out=$Out"
}

Invoke-Eval -Manifest $PoseEvalManifest -Out $PoseEvalOut
Invoke-Eval -Manifest $SafetyEvalManifest -Out $SafetyEvalOut
