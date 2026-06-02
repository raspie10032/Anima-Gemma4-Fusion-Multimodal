param(
    [string]$Repo = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal",
    [string]$Output = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\conditioning_translation_1m_v1\conditioning_translation_1m_v1.jsonl",
    [string]$Summary = "C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\reports\conditioning_translation_1m_v1\summary.json"
)

$ErrorActionPreference = "Stop"

$Source = "D:\Projects\danbooru_unified\manifest_visual_expand.jsonl"
$ImageEmbedRoot = "D:\Projects\danbooru_unified\img_embeds_pre"
$RuleTaxonomy = "E:\1\danbooru-e621-tag-list-processor-main\output\tag_taxonomy\prefilter_2026-05-22\tag_taxonomy_rule_classified.jsonl"
$LlmTaxonomy = "E:\1\danbooru-e621-tag-list-processor-main\output\tag_taxonomy\llm_2026-05-22\tag_taxonomy_llm_classified_merged.jsonl"

foreach ($path in @($Source, $ImageEmbedRoot, $RuleTaxonomy, $LlmTaxonomy)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "required path not found: $path"
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output), (Split-Path -Parent $Summary) | Out-Null

python (Join-Path $Repo "scripts\build_conditioning_translation_1m_dataset.py") `
    --source $Source `
    --taxonomy $RuleTaxonomy `
    --taxonomy $LlmTaxonomy `
    --out $Output `
    --summary $Summary `
    --limit 1000000 `
    --idx-start 5000000 `
    --image-embed-root $ImageEmbedRoot `
    --require-image-embed

exit $LASTEXITCODE
