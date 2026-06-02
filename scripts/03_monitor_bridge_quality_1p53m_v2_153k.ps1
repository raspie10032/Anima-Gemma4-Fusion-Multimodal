param(
    [string]$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal",
    [string]$Log = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\logs\bridge_quality_1p53m_v2_153k\train_4070.log",
    [string]$TargetDir = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_1p53m_v2_153k\targets",
    [string]$GemmaDir = "D:\anima_gemma_swap_cache_v2_153k\gemma",
    [string]$Out = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_1p53m_v2_153k\bridge\kv_proj_bridge_quality_1p53m_v2_153k_from_a0p35.pt"
)

$ErrorActionPreference = "Continue"

"== bridge quality 1p53m v2 153k monitor =="
"time: $(Get-Date -Format s)"
"target shards: $((Get-ChildItem -LiteralPath $TargetDir -Filter '*.pt' -File -ErrorAction SilentlyContinue | Measure-Object).Count)"
"gemma shards: $((Get-ChildItem -LiteralPath $GemmaDir -Filter '*.pt' -File -ErrorAction SilentlyContinue | Measure-Object).Count)"
"output exists: $(Test-Path -LiteralPath $Out)"
"active trainer:"
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "08_train_stream_batched.py" } | Select-Object ProcessId,CommandLine
"gpu:"
nvidia-smi
if (Test-Path -LiteralPath $Log) {
    "log tail:"
    Get-Content -LiteralPath $Log -Tail 60
} else {
    "log not found: $Log"
}
