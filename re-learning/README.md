# GEMMANIMA Re-Learning Workspace

This folder is the project-local control surface for the clean Danbooru
re-learning run. Large cache and output files stay on `D:\Projects`; this
folder keeps the runnable scripts, logs, config, and shortcuts in one place.

## Current Clean Dataset

- Source root: `D:\Projects\danbooru_unified`
- Train manifest: `D:\Projects\danbooru_unified\manifest_visual_expand.jsonl`
- Processed full set: `D:\Projects\danbooru_unified\processed_v1`
- Image cache target: `D:\Projects\danbooru_unified\img_embeds_pre`
- Output root: `D:\Projects\training\out\gemmanima_relearning_v1`

## Order

1. Build image pre-projector cache:
   `.\re-learning\01_cache_images_4070.ps1`
2. Check progress:
   `.\re-learning\03_status.ps1`
3. Train clean vision tagger:
   `.\re-learning\02_train_vision_tagger_4070.ps1`
4. Or keep the handoff running until cache completion starts training:
   `.\re-learning\04_cache_then_train_4070.ps1`

The cache stage uses the training venv and lets `v02_encode_images.py` select
the RTX 4070 Ti SUPER. The RTX 5060 is not used for PyTorch training/cache.
