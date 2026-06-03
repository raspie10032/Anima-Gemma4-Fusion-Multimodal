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
$AdapterDir = Join-Path $TrainingRoot "out\gemmanima_relearning_v1\vision_tagger_precision_pose_v2"
$MergedDir = Join-Path $TrainingRoot "out\gemmanima_relearning_v1\merged_vision_tagger_precision_pose_v2"
$EvalRoot = Join-Path $Repo "re-learning\eval"
$EvalManifest = "tagger_precision_pose_v2_manifest.jsonl"
$EvalOut = Join-Path $EvalRoot "vision_tagger_precision_pose_v2_v18_smoke.jsonl"

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TORCH_CUDA_MEM_FRACTION = "0.88"

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

$argsList = @(
    $EvalScript,
    "--backend", "transformers",
    "--data", $EvalRoot,
    "--manifest", $EvalManifest,
    "--model", $MergedDir,
    "--temp", ([string]$Temp),
    "--max-new", ([string]$MaxNew),
    "--out", $EvalOut
)
if ($Limit -gt 0) {
    $argsList += @("--limit", ([string]$Limit))
}

& $Python @argsList
Write-Host "eval_out=$EvalOut"
