param(
    [switch]$RunHeldout,
    [switch]$FetchHeldout,
    [int]$HeldoutPerRating = 50
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal"
$TrainingRoot = "D:\Projects\training"
$Python = Join-Path $TrainingRoot ".venv\Scripts\python.exe"
$MergeScript = Join-Path $TrainingRoot "v15_merge_visual_expand.py"
$EvalScript = Join-Path $TrainingRoot "v18_eval_heldout.py"
$FetchHeldoutScript = Join-Path $TrainingRoot "v17_fetch_heldout.py"

$Adapter = Join-Path $TrainingRoot "out\gemmanima_relearning_v1\vision_tagger_e10k_poseboost_rank8_vram_limit_v1"
$Merged = Join-Path $TrainingRoot "out\gemmanima_relearning_v1\merged_vision_tagger_e10k_poseboost_rank8_vram_limit_v1"
$EvalDir = Join-Path $Repo "re-learning\eval"
$SmokeManifest = "vision_tagger_clean_v1_smoke_manifest.jsonl"
$SmokeOut = Join-Path $EvalDir "vision_tagger_e10k_poseboost_rank8_vram_limit_v1_v18_smoke.jsonl"
$HeldoutRoot = "D:\Projects\danbooru_heldout_eval"
$HeldoutOut = Join-Path $EvalDir "vision_tagger_e10k_poseboost_rank8_vram_limit_v1_v18_heldout.jsonl"

if (-not (Test-Path -LiteralPath $Python)) { throw "missing python: $Python" }
if (-not (Test-Path -LiteralPath $MergeScript)) { throw "missing merge script: $MergeScript" }
if (-not (Test-Path -LiteralPath $EvalScript)) { throw "missing eval script: $EvalScript" }
if (-not (Test-Path -LiteralPath (Join-Path $Adapter "adapter_model.safetensors"))) {
    throw "final adapter is not ready yet: $Adapter"
}
if (-not (Test-Path -LiteralPath (Join-Path $Adapter "embed_vision.pt"))) {
    throw "trained projector is not ready yet: $Adapter\embed_vision.pt"
}

$env:CUDA_VISIBLE_DEVICES = "0"
$env:TORCH_CUDA_MEM_FRACTION = "0.88"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

Write-Host "[06] merge final adapter -> $Merged" -ForegroundColor Cyan
& $Python $MergeScript `
    --adapter $Adapter `
    --out $Merged `
    --lora-r 8 `
    --lora-alpha 16

if ($LASTEXITCODE -ne 0) { throw "merge failed with exit code $LASTEXITCODE" }

Write-Host "[06] smoke eval on existing re-learning manifest" -ForegroundColor Cyan
& $Python $EvalScript `
    --backend transformers `
    --data $EvalDir `
    --manifest $SmokeManifest `
    --model $Merged `
    --limit 4 `
    --temp 0.2 `
    --max-new 96 `
    --out $SmokeOut

if ($LASTEXITCODE -ne 0) { throw "smoke eval failed with exit code $LASTEXITCODE" }

if ($FetchHeldout) {
    if (-not (Test-Path -LiteralPath $FetchHeldoutScript)) { throw "missing heldout fetch script: $FetchHeldoutScript" }
    Write-Host "[06] fetch heldout set -> $HeldoutRoot" -ForegroundColor Cyan
    & $Python $FetchHeldoutScript --per-rating $HeldoutPerRating --ratings "g,s,q,e"
    if ($LASTEXITCODE -ne 0) { throw "heldout fetch failed with exit code $LASTEXITCODE" }
}

if ($RunHeldout) {
    if (-not (Test-Path -LiteralPath (Join-Path $HeldoutRoot "manifest.jsonl"))) {
        throw "heldout manifest missing: $HeldoutRoot\manifest.jsonl (rerun with -FetchHeldout or create it first)"
    }
    Write-Host "[06] heldout eval -> $HeldoutOut" -ForegroundColor Cyan
    & $Python $EvalScript `
        --backend transformers `
        --data $HeldoutRoot `
        --model $Merged `
        --temp 0.2 `
        --max-new 96 `
        --out $HeldoutOut
    if ($LASTEXITCODE -ne 0) { throw "heldout eval failed with exit code $LASTEXITCODE" }
}

Write-Host "[06] done" -ForegroundColor Green
Write-Host "merged=$Merged"
Write-Host "smoke_eval=$SmokeOut"
if ($RunHeldout) { Write-Host "heldout_eval=$HeldoutOut" }
