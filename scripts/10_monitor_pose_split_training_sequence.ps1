param(
    [string]$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal",
    [string]$SequenceLog = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\bridge_quality_diagnosis\logs\pose_split_sequence\sequence.log"
)

$ErrorActionPreference = "Continue"

"== pose split sequence monitor =="
"time: $(Get-Date -Format s)"
"active cache/train:"
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "09_run_pose_split|06_cache_targets.py|07_cache_gemma_batched.py|08_train_stream_batched.py|04_prepare_pose_mix|06_train_bridge_quality|08_train_bridge_quality" } | Select-Object ProcessId,CommandLine
"gpu:"
nvidia-smi
"153k base audit:"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Repo "scripts\00_audit_bridge_quality_1p53m_v2_153k.ps1")
"pose mix audit:"
if (Test-Path (Join-Path $Repo "reports\bridge_quality_diagnosis\bridge_quality_1p53m_v2_153k_pose10k_audit.json")) {
    Get-Content (Join-Path $Repo "reports\bridge_quality_diagnosis\bridge_quality_1p53m_v2_153k_pose10k_audit.json")
} else {
    "not created yet"
}
"pose-only audit:"
if (Test-Path (Join-Path $Repo "reports\bridge_quality_diagnosis\bridge_quality_pose_only_10k_audit.json")) {
    Get-Content (Join-Path $Repo "reports\bridge_quality_diagnosis\bridge_quality_pose_only_10k_audit.json")
} else {
    "not created yet"
}
"sequence log tail:"
if (Test-Path $SequenceLog) {
    Get-Content $SequenceLog -Tail 80
} else {
    "not created yet"
}
"pose-only train tail:"
$poseOnlyLog = Join-Path $Repo "reports\bridge_quality_diagnosis\logs\bridge_quality_pose_only_10k\train_4070.log"
if (Test-Path $poseOnlyLog) {
    Get-Content $poseOnlyLog -Tail 40
} else {
    "not created yet"
}
"integrated train tail:"
$integratedLog = Join-Path $Repo "reports\bridge_quality_diagnosis\logs\bridge_quality_1p53m_v2_153k_pose10k\train_4070.log"
if (Test-Path $integratedLog) {
    Get-Content $integratedLog -Tail 40
} else {
    "not created yet"
}
