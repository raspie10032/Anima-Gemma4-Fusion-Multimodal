# vision_tagger_clean_v1 hold note

Status: keep for reference only. Do not promote, quantize, publish, or use as the active GEMMANIMA vision tagger without a later passing evaluation.

Artifacts kept:
- Adapter: `D:\Projects\training\out\gemmanima_relearning_v1\vision_tagger_clean_v1`
- Merged HF model: `D:\Projects\training\out\gemmanima_relearning_v1\merged_vision_tagger_clean_v1`

Evaluation summary:
- 20-image smoke manifest: `re-learning\eval\vision_tagger_clean_v1_smoke_manifest.jsonl`
- v18 smoke, temp 0.7: overall output-tag accuracy 18.8%
- v18 smoke, temp 0.2: overall output-tag accuracy 21.6%, pose 4.0%
- Greedy raw spotcheck repeated a fixed NSFW-heavy tag template on G-rated images.

Decision:
- Keep the model as a failed/diagnostic checkpoint because the training run is still useful for comparison.
- Use `D:\Projects\danbooru_unified_balanced_v1` for the next balanced re-learning attempt.
