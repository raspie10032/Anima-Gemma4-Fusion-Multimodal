# HiddenStage Training Pipeline

## Current State

The multimodal planner LoRA is complete and passed evaluation:

- Output: `D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2`
- Eval step: `20500`
- Eval loss: `1.0061092711985111`
- Threshold: `1.5`

The next model training target is the HiddenStage Exit bridge:

```text
Gemma hidden [B,S,1536] -> Anima crossattn_emb [B,512,1024]
```

## Stage 1: Teacher Crossattn Targets

Teacher target extraction is running from:

```text
runs\teacher_targets\hiddenstage_multimodal_planner_anima_v2_teacher_subset.jsonl
```

Output:

```text
E:\anima_gemma_swap\cache_hiddenstage_planner_v2\targets
```

GPU split:

- RTX 4070 Ti SUPER: `shard_[0-9][0-9][0-9][0-9].pt`
- RTX 5060: `shard_5060_*.pt`

Current automation safety rule: the RTX 5060 is reserved for the user. New PoC1 pilot automation must target only the RTX 4070 Ti SUPER with `CUDA_VISIBLE_DEVICES='0'` and must not schedule commands for CUDA device 1 / RTX 5060.

Each target shard stores:

- `idx`
- `t5_ids`
- `target`: original Anima LLMAdapter output, shape `[T,1024]`, dtype `float16`

## Stage 2: Gemma Hidden Cache

After target extraction completes, run:

```powershell
.\scripts\run_gemma_hidden_cache_split_4070_ti_super.ps1
.\scripts\run_gemma_hidden_cache_split_5060.ps1
```

Output:

```text
D:\anima_gemma_swap_cache_hiddenstage_planner_v2\gemma
```

The 4070 Ti SUPER runner uses `GEMMA_EMBED_ON_GPU=1`.
The 5060 runner keeps embeddings on CPU with `GEMMA_EMBED_ON_GPU=0`.

Cache build reporting now has a repo-local manifest contract:

```text
gemmanima.training.cache_manifest.CacheBuildManifest
schemas/cache_build_manifest.schema.json
```

PoC cache commands should emit a validated cache manifest before the project treats the cache as ready for distillation or bridge training.

## PoC1 1k Smoke Result

The first isolated 1k smoke pass completed under repo-local ignored runtime paths:

```text
runs\teacher_targets\poc1_1k_teacher_subset.jsonl
runs\cache\poc1_1k\targets
runs\cache\poc1_1k\gemma
runs\cache\poc1_1k\bridge\poc1_bridge.pt
runs\images\poc1_bridge_real_smoke.png
```

Observed result:

- Teacher targets cached: 1000 examples, 1 shard.
- Gemma hidden cached: 1000 examples, 1 paired shard.
- Bridge smoke train: epoch 1, limit 1 shard.
- Val MSE: `0.0031542012887075545`, gate `0.004`, passed.
- Forward smoke: `[1,16,1536] -> [1,32,1024]`, finite true.
- Real render smoke: 512px, 8 steps, output saved.
- Full-bridge baseline vs PoC1 checkpoint compare report written.

Summary report:

```text
reports\poc1_1k_smoke_report.json
reports\poc1_generation_compare_report.json
```

## PoC1 10k Pilot Result

The 10k pilot cache and bridge training pass completed under the 4070-only training manager. It remains separate from the completed 1k smoke evidence above because it uses distinct repo-local ignored runtime paths.

Use explicit repo-local ignored paths and the 4070-only GPU profile so the 10k artifacts cannot be confused with the 1k smoke run or accidentally target the RTX 5060:

```powershell
python -m gemmanima.cli poc1-cache-plan --limit 10000 --gpu-profile 4070-only --subset runs\teacher_targets\poc1_10k_teacher_subset.jsonl --target-dir runs\cache\poc1_10k\targets --gemma-dir runs\cache\poc1_10k\gemma --json
```

This is a planning and dry command path. It prints the teacher subset export command, teacher target cache command, Gemma hidden cache command, and cache-manifest write commands. It does not run GPU cache generation, bridge training, rendering, or image comparison by itself.

The generated runtime commands were run with `CUDA_VISIBLE_DEVICES='0'`. The RTX 5060 remained reserved for the user and was not targeted by automation.

Inspect the final runtime status without touching the GPU:

```powershell
python -m gemmanima.cli poc1-runtime-status --json
```

The status command observes the default 10k paths:

```text
runs\cache\poc1_10k\targets
runs\cache\poc1_10k\gemma
runs\cache\poc1_10k\bridge\poc1_10k_bridge.pt
```

It reports target shard count, Gemma shard count, paired and missing shards, bridge checkpoint audit, and `ready_for_bridge_training`.

Observed 10k result:

- Teacher targets cached: 10000 examples, 10 shards.
- Gemma hidden cached: 10 paired shards.
- Cache pairing: target `10`, Gemma `10`, paired `10`, missing `0`.
- Bridge train: epoch 1 across paired 10k pilot shards.
- Checkpoint: `runs\cache\poc1_10k\bridge\poc1_10k_bridge.pt`
- Val MSE: `0.0020450321631506085`, gate `0.004`, passed.

Text rendering preservation prompt pack:

```powershell
python -m gemmanima.cli text-rendering-eval-pack --json
```

The prompt pack defines deterministic case ids, seeds, expected teacher/student image paths, and comparison report paths. A dry-run execution plan can print the same targets without running GPU commands:

```powershell
python -m gemmanima.cli text-rendering-eval-plan --json
```

The plan uses `mode: dry_run`, `executes_gpu_commands: false`, and `artifact_policy: declare_paths_only`. It is a scheduling contract for later render and comparison work, not evidence that images or metrics exist.

Status reporting remains artifact-based:

```powershell
python -m gemmanima.cli text-rendering-eval-status --json
```

The status command must not invent OCR scores, image metrics, or pass/fail values. It reports whether each expected teacher image, student image, and compare report exists, and keeps cases in a pending state until those artifacts are actually generated.

Observed text-rendering eval result:

- Added `text-rendering-eval-run-plan --json` to emit executable 4070-only teacher/student render commands plus compare commands.
- Generated all 6 teacher/student pairs with `CUDA_VISIBLE_DEVICES='0'`; the RTX 5060 remained reserved for the user and was not targeted by automation.
- Wrote 6 compare reports under `reports\text_rendering_eval`.
- Status: `ready_cases=6`, `pending_cases=0`.
- Contact sheet: `reports\text_rendering_eval\contact_sheet.png`.
- Manual visual review: object-only prompting prevented the earlier person-focused drift, but target text legibility failed on all 6 cases. Treat these artifacts as observed failure evidence for text preservation, not as a pass.
- Manual visual review report: `reports\text_rendering_eval\visual_review.json`.

Qwen teacher baseline vs Gemma PoC1 10k text-rendering result:

- Added `text-rendering-qwen-baseline-plan --json` and `text-rendering-qwen-baseline-prompts --json` to use the real Qwen/Anima teacher path from `11_eval_generate.py --mode qwen`, then compare against the Gemma bridge path with `--mode gemma --adapter runs\cache\poc1_10k\bridge\poc1_10k_bridge.pt`.
- Wrote baseline prompts to `reports\text_rendering_qwen_baseline\prompts.jsonl`.
- Generated 6 Qwen teacher images and 6 Gemma PoC1 10k student images under `runs\images\text_rendering_qwen_baseline` with `CUDA_VISIBLE_DEVICES='0'`; the RTX 5060 remained reserved for the user.
- Wrote 6 Qwen-vs-Gemma compare reports under `reports\text_rendering_qwen_baseline`.
- Contact sheet: `reports\text_rendering_qwen_baseline\contact_sheet.png`.
- Manual visual review report: `reports\text_rendering_qwen_baseline\visual_review.json`.
- Manual visual review: Qwen teacher rendered readable text on 5 cases and partial stylized ring text on 1 case; Gemma PoC1 10k preserved some object/composition cues but produced no fully readable target text. Treat this as evidence that the next bridge pass should preserve Qwen text-rendering features, not merely match coarse scene semantics.

Text-preservation micro-overfit result:

- Added `text-preservation-bridge-plan --json` and `text-preservation-bridge-status --json`.
- Reused the existing TE distillation path: `06_cache_targets.py` for Qwen/Anima target conditioning, `07_cache_gemma_batched.py` for Gemma hidden states, and `08_train_stream_batched.py` for the bridge MSE objective.
- Used the 6 Qwen baseline text prompts as a separate repo-local micro-overfit set under `runs\cache\text_preservation_qwen`.
- Resumed from `runs\cache\poc1_10k\bridge\poc1_10k_bridge.pt` and trained on the RTX 4070 Ti SUPER only with `CUDA_VISIBLE_DEVICES='0'`.
- Checkpoint: `runs\cache\text_preservation_qwen\bridge\text_preservation_bridge.pt`.
- Observed bridge Val MSE: `0.0006606672153187295`, gate `0.004`, passed.
- Generated `gemma_text_preservation` images for all 6 Qwen baseline prompts and wrote 6 compare reports under `reports\text_rendering_qwen_baseline`.
- Contact sheet: `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation.png`.
- Manual visual review report: `reports\text_rendering_qwen_baseline\visual_review_text_preservation.json`.
- Observed result: text-preservation restored readable text on 5 cases and partial stylized ring text on 1 case. Mean image MSE vs Qwen improved from `0.15582648385316133` for PoC1 10k to `0.030663649920218933` for the micro-overfit checkpoint.
- Promotion note: this is strong evidence that the existing TE distillation method can carry Qwen text-rendering features, but it is still a 6-case overfit artifact. The next promotion step should expand/blend the text prompt set before treating it as a general bridge.

Text-preservation blended candidate result:

- Added expanded text-preservation prompt generation via `text-preservation-prompts --include-eval-cases` and reproducible blended planning via `text-preservation-blended-plan --json`.
- First promoted to a 54-prompt v3 pack, then expanded to a 518-prompt v4 pack after the sample size concern: the original 6 Qwen baseline eval prompts plus 512 additional unique object-only text preservation prompts.
- Created blended training dirs under `runs\cache\text_preservation_blended_v4` using hardlinks: 8 repeated text shards plus 2 PoC1 10k general shards, for 10 paired training shards without copying the full PoC1 cache.
- Trained on the RTX 4070 Ti SUPER only with `CUDA_VISIBLE_DEVICES='0'`; the RTX 5060 remained reserved for the user.
- Best observed blended checkpoint: `runs\cache\text_preservation_blended_v4\bridge\text_preservation_blended_v4_bridge.pt`.
- Observed bridge Val MSE: `0.00019171485413001696`, gate `0.004`, passed.
- Contact sheet: `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v4.png`.
- Manual visual review report: `reports\text_rendering_qwen_baseline\visual_review_text_preservation_blended_v4.json`.
- Observed result: readable text on 5 cases, partial stylized ring text on 1 case, and 0 failed cases. Mean image MSE vs Qwen improved from PoC1 10k `0.15582648385316133` to blended v4 `0.02005120733520016`.
- Selection note: blended v1/v2 proved the hardlink mix path but regressed `HP 42`; v3 restored `HP 42`; v4 increased sample count to 518 while keeping the fixed 6-case gate readable. Treat v4 as the current text-preservation candidate for the next broader validation pass.

Text-preservation v5 held-out candidate result:

- Added `text-preservation-v5-plan --json` to build a larger no-sample-marker blended run and to include every generated text shard when the prompt count exceeds one target shard.
- v5 prompt pack: 1030 text prompts, made from the fixed 6 eval prompts plus 1024 additional unique object-only text prompts. The expanded prompt writer omits `sample N` markers so non-target sample ids do not leak into rendered images.
- Created blended training dirs under `runs\cache\text_preservation_blended_v5` using hardlinks: 6 repeats across 2 text shards plus 4 PoC1 10k general shards, for 16 paired training shards without copying the full PoC1 cache.
- Trained on the RTX 4070 Ti SUPER only with `CUDA_VISIBLE_DEVICES='0'`; the RTX 5060 remained reserved for the user.
- Checkpoint: `runs\cache\text_preservation_blended_v5\bridge\text_preservation_blended_v5_bridge.pt`.
- Observed bridge Val MSE: `0.000139208897962817`, gate `0.004`, passed.
- Fixed 6-case contact sheet: `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v5_fixed6.png`.
- Fixed 6-case mean image MSE vs Qwen improved from blended v4 `0.02005120733520016` to v5 `0.013701042432027558`.
- Clean held-out prompt set: `reports\text_preservation_heldout_v5_clean\prompts.jsonl`.
- Clean held-out contact sheets: `reports\text_preservation_heldout_v5_clean\contact_sheet_sample24.png` and `reports\text_preservation_heldout_v5_clean\contact_sheet_all64.png`.
- Clean held-out manual visual review report: `reports\text_preservation_heldout_v5_clean\visual_review.json`.
- Clean held-out result: 47 readable, 13 partial, and 4 failed cases across 64 prompts. Mean image MSE vs Qwen: `0.03992582718410631`, median MSE: `0.025133159942924976`.
- General scene smoke: `reports\poc1_general_regression_v5_review.json` and `reports\poc1_general_regression_v4_v5_contact.png`. v5 remains nonblank and scene-consistent, but it shifts style more strongly than v4, so it should be treated as the current text-preservation candidate rather than a default bridge replacement until broader semantic regression passes.

Text-preservation v6 hard-negative experiment:

- Added `text-preservation-v6-plan --json` and `text-preservation-v6-prompts --json` for a focused hard-negative pass against held-out failures such as `BRIGHT`, `NOON BELL`, `GREEN TEA`, `QUIET HILL`, `TICKET A`, `SILVER LINE`, `VIOLET`, and `ROW 7`.
- v6 prompt pack: 326 prompts, made from the fixed 6 eval prompts plus 320 no-sample-marker hard-negative prompts.
- Created blended training dirs under `runs\cache\text_preservation_blended_v6` using hardlinks: 10 repeated hard-negative text shards plus 4 PoC1 10k general shards, for 14 paired training shards.
- Trained on the RTX 4070 Ti SUPER only with `CUDA_VISIBLE_DEVICES='0'`; the RTX 5060 remained reserved for the user.
- Checkpoint: `runs\cache\text_preservation_blended_v6\bridge\text_preservation_blended_v6_bridge.pt`.
- Observed bridge Val MSE: `0.0001442893963030656`, gate `0.004`, passed.
- Fixed 6-case mean image MSE regressed from v5 `0.013701042432027558` to v6 `0.021531414221196126`, with `LUNA GATE` visibly worse.
- Clean held-out mean image MSE regressed from v5 `0.03992582718410631` to v6 `0.07254247733180819`.
- Selection note: do not promote v6. Keep it only as evidence that hard-negative-only fine-tuning needs a lower weight, replay of v5 text prompts, or a more conservative learning rate before it can improve hard cases without damaging fixed gates.

General scene regression result:

- Added `text-preservation-general-scene-prompts --json` and `text-preservation-general-scene-eval-plan --json`.
- Generated a 15-case Qwen-vs-v5 general scene regression pack covering cafe, forest, city, fantasy character, beach, market, robot, classroom, food, action, spaceship, rainy street, desk, garden, and library scenes.
- Artifacts: `reports\general_scene_regression_v5\prompts.jsonl`, `reports\general_scene_regression_v5\metrics_summary.json`, `reports\general_scene_regression_v5\visual_review.json`, and `reports\general_scene_regression_v5\contact_sheet_all15.png`.
- Observed result: v5 generated nonblank general scenes across all 15 cases and did not collapse into text-only artifacts. Mean image MSE vs Qwen was `0.09042903129011393`, but this is a coarse smoke metric because Qwen and Gemma often choose different valid compositions.
- Promotion note: v5 remains the current text-preservation candidate. A larger semantic regression pack is still required before replacing PoC1 10k or v4 as the default bridge.

Expanded general scene regression result:

- Added scaled prompt generation for `text-preservation-general-scene-prompts --count 50` so the 15 base scene types can expand into 50 unique prompt records without duplicating exact prompts.
- Ran a 50-case Qwen-vs-v5 general scene regression pack on the RTX 4070 Ti SUPER only.
- Artifacts: `reports\general_scene_regression_v5_50\prompts.jsonl`, `reports\general_scene_regression_v5_50\metrics_summary.json`, `reports\general_scene_regression_v5_50\visual_review.json`, and `reports\general_scene_regression_v5_50\contact_sheet_all50.png`.
- Observed result: 50 comparison reports were generated. v5 remained nonblank and did not collapse into text-only artifacts across the expanded general-scene pack.
- Mean image MSE vs Qwen: `0.06805774327367545`; median MSE: `0.044416142627596855`; max MSE: `0.3583686649799347`.
- Promotion note: this improves confidence that v5 does not destroy ordinary scene generation, but pixel metrics still reflect valid style/composition differences. v5 remains the current text-preservation candidate, not yet an unconditional default replacement.

Text-preservation v7 balanced replay experiment:

- Added `text-preservation-v7-plan --json` for a balanced replay pass that resumes from v5, replays v5 text shards, includes the v6 hard-negative shard at lower relative weight, and links all 10 PoC1 10k general shards.
- v7 hardlinked training mix: 4 repeats across the 2 v5 text shards, 1 hard-negative shard replay, and 10 general shards, for 19 paired training shards.
- Trained on the RTX 4070 Ti SUPER only with `CUDA_VISIBLE_DEVICES='0'`; the RTX 5060 remained reserved for the user.
- Checkpoint: `runs\cache\text_preservation_blended_v7\bridge\text_preservation_blended_v7_bridge.pt`.
- Observed bridge Val MSE: `0.00011672825917230512`, gate `0.004`, passed.
- Fixed 6-case Qwen-vs-v7 reports and contact sheet: `reports\text_rendering_qwen_baseline\metrics_summary_v7_fixed6.json` and `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v7_fixed6.png`.
- Fixed 6-case mean image MSE regressed from v5 `0.013701042432027558` to v7 `0.024075733303713303`; `LUNA GATE` regressed most visibly.
- Selection note: do not promote v7. Bridge validation MSE alone is not sufficient for promotion. Keep v5 as the current candidate and require fixed text replay preservation before another large run.

Text-preservation v8 fixed-gate replay experiment:

- Added `text-preservation-v8-plan --json` for a conservative fixed-gate preserving replay pass that resumes from v5 and heavily replays the original 6-case text-preservation cache.
- Added `text-preservation-promotion-status --json` as a non-GPU artifact observer that compares candidate fixed 6-case image metrics against the v5 baseline.
- v8 hardlinked training mix: 8 fixed-gate replays, 2 repeats across the 2 v5 text shards, 1 hard-negative shard replay, and 4 PoC1 10k general shards, for 17 paired training shards.
- Trained on the RTX 4070 Ti SUPER only with `CUDA_VISIBLE_DEVICES='0'`; the RTX 5060 remained reserved for the user.
- Checkpoint: `runs\cache\text_preservation_blended_v8\bridge\text_preservation_blended_v8_bridge.pt`.
- Observed bridge Val MSE: `0.0006323225916275987`, gate `0.004`, passed.
- Fixed 6-case Qwen-vs-v8 reports and contact sheet: `reports\text_rendering_qwen_baseline\metrics_summary_v8_fixed6.json` and `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v8_fixed6.png`.
- Fixed 6-case mean image MSE regressed from v5 `0.013701042432027558` to v8 `0.025927293114364147`.
- Selection note: do not promote v8. v6, v7, and v8 now show the same lesson: more replay or lower bridge Val MSE does not guarantee fixed text rendering preservation. Keep v5 as the current candidate.

Protected baseline promotion status:

- `text-preservation-promotion-status --json` now includes a top-level `recommendation` block.
- The command can also write a durable report:

```powershell
python -m gemmanima.cli text-preservation-promotion-status --output reports\text_rendering_qwen_baseline\promotion_status.json --json
```

- Current report: `reports\text_rendering_qwen_baseline\promotion_status.json`.
- Compact review report: `reports\text_rendering_qwen_baseline\promotion_status_compact.json`.
- Current recommendation: `protect_baseline`, protected baseline `v5`, promote candidate `null`, rejected candidates `v6`, `v7`, and `v8`.
- Strict fixed-gate checks now include report count, mean MSE, max MSE, and per-case MSE against the v5 protected baseline.
- This is a non-GPU artifact observer. It must remain safe to run while the RTX 5060 is reserved for the user.

Release gate and v9 training gate:

- Added `text-preservation-release-gate --json` as a non-GPU release gate for the protected v5 text-preservation baseline.
- Durable report: `reports\text_rendering_qwen_baseline\release_gate_status.json`.
- Current release gate status: `pass`.
- Current v9 training gate status: `blocked_until_objective_redesign`.
- Interpretation: v5 is protected by fixed6, held-out, general-scene, and promotion evidence. Another v9 replay-weighting run should not start until the objective itself changes or an image/text-level feedback path is added.
- Added `text-preservation-v9-objective-plan --json` as the non-GPU design contract that records the next valid v9 path before any training command can exist.
- Durable report: `reports\text_rendering_qwen_baseline\v9_objective_plan.json`.
- Current objective redesign status: `blocked_until_artifact_gate_first_objective_redesign`; recommended approach: `artifact_gate_first`.
- Added `text-preservation-v9-artifact-gate-objective --json` as the concrete objective contract. It allows candidate planning but keeps GPU training blocked until the trainer supports artifact feedback.
- Durable report: `reports\text_rendering_qwen_baseline\v9_artifact_gate_objective.json`.
- Added `text-preservation-v9-trainer-support-audit --json` to check the external bridge trainer without touching the GPU.
- Durable report: `reports\text_rendering_qwen_baseline\v9_trainer_support_audit.json`.
- Current trainer support status after patching `08_train_stream_batched.py`: `supported`.
- Added `text-preservation-v9-artifact-feedback`, `text-preservation-v9-artifact-gate-loss-config`, and `text-preservation-v9-candidate-plan --json` to produce a 4070-only artifact-feedback candidate plan.
- v9 artifact-gate candidate trained on the RTX 4070 Ti SUPER only. Checkpoint: `runs\cache\text_preservation_blended_v9\bridge\text_preservation_blended_v9_bridge.pt`.
- Observed bridge Val MSE: `0.0005838303222844843`, gate `0.004`, passed.
- Fixed 6-case Qwen-vs-v9 reports and contact sheet: `reports\text_rendering_qwen_baseline\metrics_summary_v9_fixed6.json` and `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v9_fixed6.png`.
- Fixed 6-case mean image MSE regressed from v5 `0.013701042432027558` to v9 `0.025934352500674624`.
- Selection note: do not promote v9. Keep v5 as the protected text-preservation baseline.
- Next objective: v10 added KV-anchor regularization against the protected v5 checkpoint while using artifact feedback. It trained on the RTX 4070 Ti SUPER only.
- v10 checkpoint: `runs\cache\text_preservation_blended_v10\bridge\text_preservation_blended_v10_bridge.pt`.
- Observed bridge Val MSE: `0.0005838341403432423`, gate `0.004`, passed.
- Fixed 6-case Qwen-vs-v10 reports and contact sheet: `reports\text_rendering_qwen_baseline\metrics_summary_v10_fixed6.json` and `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v10_fixed6.png`.
- Fixed 6-case mean image MSE regressed from v5 `0.013701042432027558` to v10 `0.026236171058068674`.
- Selection note: do not promote v10. Before v11, audit whether artifact feedback weights are actually aligning to the intended blended shard sample ids.
- Added `text-preservation-artifact-feedback-alignment-audit --json`. v9/v10 feedback ids occurred 66 times: 48 in fixed-gate shards, 12 in v5 text replay, and 6 in hard-negative replay.
- v11 added source-bucket filtered artifact feedback so weights apply only to `00_fixed_gate` shards, then trained on the RTX 4070 Ti SUPER only.
- v11 checkpoint: `runs\cache\text_preservation_blended_v11\bridge\text_preservation_blended_v11_bridge.pt`.
- Observed bridge Val MSE: `0.0006095190765336156`, gate `0.004`, passed.
- Fixed 6-case Qwen-vs-v11 reports and contact sheet: `reports\text_rendering_qwen_baseline\metrics_summary_v11_fixed6.json` and `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v11_fixed6.png`.
- Fixed 6-case mean image MSE regressed from v5 `0.013701042432027558` to v11 `0.026408916998965044`.
- Selection note: do not promote v11. Source filtering fixed the feedback spillover, but it did not fix the fixed6 image drift.
- Added `text-preservation-kv-delta-audit --json` as a non-GPU checkpoint drift audit against the protected v5 KV tensors.
- Durable report: `reports\text_rendering_qwen_baseline\kv_delta_audit_v9_v10_v11.json`.
- Current KV delta result: v9 element-weighted MSE `1.257022133371713e-07`, v10 `1.2565436002131657e-07`, v11 `1.2578555395113333e-07`; v10/v11 are not materially changed versus v9, so the KV anchor/source filtering did not meaningfully reduce checkpoint drift.
- Next selection note: do not start another replay-weighting-only training run. The next candidate must change the training surface or objective, not just sample weights, source buckets, or the same KV anchor.
- Added `text-preservation-v12-surface-plan --json` as the next non-GPU workflow contract. It keeps v5 protected, blocks GPU training, and moves the workflow to `v12_surface_redesign`.
- Durable report: `reports\text_rendering_qwen_baseline\v12_training_surface_plan.json`.
- Recommended v12 surface: `render_readability_conditioned_target_refresh`. Required artifacts before training are a render/readability label manifest, surface curriculum manifest, Qwen target refresh manifest, fixed6 per-case baseline map, held-out partial/failed case pack, and trainer surface contract audit.
- Current workflow position: `v12_surface_redesign`; next safe step: `build_render_readability_label_manifest`.
- Added `text-preservation-render-readability-label-manifest --json` and generated `reports\text_rendering_qwen_baseline\render_readability_label_manifest_v12.json`.
- Manifest result: 70 labeled render records from fixed6 and clean held-out reviews: 6 accepted baseline, 47 readable, 13 partial, and 4 failed. The 17 partial/failed held-out records are marked `v12_priority_refresh`.
- Current workflow position: `build_render_readability_label_manifest`; next safe step: `build_surface_curriculum_manifest`.
- Added `text-preservation-surface-curriculum-manifest --json` and generated `reports\text_rendering_qwen_baseline\surface_curriculum_manifest_v12.json`.
- Curriculum result: 35 records total: 4 failed refresh, 13 partial refresh, 6 fixed-gate guards, and 12 readable replay guards. Failed records receive the highest sample weight, partial records the next highest, and GPU training remains blocked.
- Current workflow position: `build_surface_curriculum_manifest`; next safe step: `build_qwen_target_refresh_manifest`.
- Added `text-preservation-qwen-target-refresh-manifest --json` and generated `reports\text_rendering_qwen_baseline\qwen_target_refresh_manifest_v12.json`.
- Prompt subset: `reports\text_rendering_qwen_baseline\qwen_target_refresh_prompts_v12.jsonl`, 35 records from the surface curriculum.
- Target cache destination: `runs\cache\text_preservation_blended_v12\targets`; emitted command uses `CUDA_VISIBLE_DEVICES='0'` for the RTX 4070 Ti SUPER only.
- Current workflow position: `build_qwen_target_refresh_manifest`; next safe step: `audit_trainer_surface_contract`.
- Added `text-preservation-v12-trainer-surface-contract-audit --json` and generated `reports\text_rendering_qwen_baseline\v12_trainer_surface_contract_audit.json`.
- Trainer audit result after patching `08_train_stream_batched.py`: `supported`.
- v12 target/Gemma cache completed on the RTX 4070 Ti SUPER only: `runs\cache\text_preservation_blended_v12\targets\shard_0000.pt` and `runs\cache\text_preservation_blended_v12\gemma\shard_0000.pt`, 35 examples, 1 paired shard.
- v12 candidate trained from protected v5 with the surface curriculum, per-case gate budget `0.004`, and pre-train release gate assertion. Checkpoint: `runs\cache\text_preservation_blended_v12\bridge\text_preservation_blended_v12_bridge.pt`.
- Observed bridge Val MSE: `0.002338456588664225`, gate `0.004`, passed.
- Fixed6 Qwen-vs-v12 render/compare completed and v12 is rejected. Mean image MSE regressed to `0.057429684209637344` versus protected v5 `0.013701042432027558`, with all 6 fixed cases flagged as per-case regressions.
- Added fixed6 v12 artifacts: `reports\text_rendering_qwen_baseline\metrics_summary_v12_fixed6.json` and `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v12_fixed6.png`.
- Promotion observer and release gate regenerated: v5 remains the protected baseline, v6-v12 are rejected, pending candidates are empty, and release gate recommendation remains `protect_baseline`.
- Added `text-preservation-v13-recovery-plan --json` and generated `reports\text_rendering_qwen_baseline\v13_recovery_plan.json`. Current workflow position: `v13_recovery_planning`; next safe step: `build_guard_weighted_v13_manifest`.
- Added `text-preservation-v13-guard-weighted-manifest --json` and generated `reports\text_rendering_qwen_baseline\v13_guard_weighted_manifest.json` plus `reports\text_rendering_qwen_baseline\v13_guard_weighted_prompts.jsonl`. It limits the first v13 run to 12 records: 6 fixed-gate guards, 4 readable replay guards, and 2 capped failed-refresh ablation cases.
- Cached v13 Qwen targets and Gemma hidden states on the RTX 4070 Ti SUPER only, then trained `runs\cache\text_preservation_blended_v13\bridge\text_preservation_blended_v13_bridge.pt` from protected v5. Bridge Val MSE `0.0002798435161821544` passed the `0.004` bridge gate.
- Fixed6 Qwen-vs-v13 render/compare completed. v13 improves over v12 but is still rejected: mean image MSE `0.016183895776824404` versus protected v5 `0.013701042432027558`; per-case regressions remain on `text_eval_001_sign_luna_gate` and `text_eval_006_label_tea`.
- Added fixed6 v13 artifacts: `reports\text_rendering_qwen_baseline\metrics_summary_v13_fixed6.json` and `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v13_fixed6.png`.
- Added focus-only fixed-gate v14/v15/v16 ablations after v13. All were trained from protected v5 on the RTX 4070 Ti SUPER only and all remain rejected by the fixed6 per-case promotion gate.
- v14 checkpoint: `runs\cache\text_preservation_blended_v14\bridge\text_preservation_blended_v14_bridge.pt`; bridge Val MSE `0.0004054944438394159`; fixed6 mean image MSE `0.007978032362492135`, but `text_eval_005_handwritten_note_meet_at_dawn` and `text_eval_006_label_tea` regressed versus v5.
- v15 checkpoint: `runs\cache\text_preservation_blended_v15\bridge\text_preservation_blended_v15_bridge.pt`; fixed6 mean image MSE `0.007978032362492135`; rejected for the same MEET AT DAWN and TEA per-case regressions. Diagnosis: the trainer's default artifact-weight clamp (`max_weight=4.0`) made guard/focus weights effectively equal.
- v16 checkpoint: `runs\cache\text_preservation_blended_v16\bridge\text_preservation_blended_v16_bridge.pt`; `reports\text_rendering_qwen_baseline\v16_true_focus_gate_loss_config.json` raises `max_weight` to `16.0`, so focus weights are actually active. Bridge Val MSE `0.00040636849007569253`; fixed6 mean image MSE `0.008126146024248252`; still rejected on MEET AT DAWN and TEA.
- Added fixed6 v14-v16 artifacts: `metrics_summary_v14_fixed6.json`, `metrics_summary_v15_fixed6.json`, `metrics_summary_v16_fixed6.json`, plus matching `contact_sheet_text_preservation_blended_v*_fixed6.png` files under `reports\text_rendering_qwen_baseline`.
- Current workflow position: `v16_true_focus_fixed6_rejected`; next safe step: stop small replay-only weighting ablations and design a targeted target/teacher-side refresh for the two remaining regression surfaces while continuing to protect v5.

```powershell
python -m gemmanima.cli text-preservation-v9-objective-plan --output reports\text_rendering_qwen_baseline\v9_objective_plan.json --json
python -m gemmanima.cli text-preservation-v9-artifact-gate-objective --output reports\text_rendering_qwen_baseline\v9_artifact_gate_objective.json --json
python -m gemmanima.cli text-preservation-v9-trainer-support-audit --output reports\text_rendering_qwen_baseline\v9_trainer_support_audit.json --json
python -m gemmanima.cli text-preservation-v9-artifact-feedback --output reports\text_rendering_qwen_baseline\v9_artifact_feedback.jsonl --json
python -m gemmanima.cli text-preservation-v9-artifact-gate-loss-config --output reports\text_rendering_qwen_baseline\v9_artifact_gate_loss_config.json --json
python -m gemmanima.cli text-preservation-v9-candidate-plan --json
python -m gemmanima.cli text-preservation-v10-candidate-plan --json
python -m gemmanima.cli text-preservation-artifact-feedback-alignment-audit --output reports\text_rendering_qwen_baseline\v10_artifact_feedback_alignment_audit.json --json
python -m gemmanima.cli text-preservation-v11-candidate-plan --json
python -m gemmanima.cli text-preservation-kv-delta-audit --output reports\text_rendering_qwen_baseline\kv_delta_audit_v9_v10_v11.json --json
python -m gemmanima.cli text-preservation-v12-surface-plan --output reports\text_rendering_qwen_baseline\v12_training_surface_plan.json --json
python -m gemmanima.cli text-preservation-render-readability-label-manifest --output reports\text_rendering_qwen_baseline\render_readability_label_manifest_v12.json --json
python -m gemmanima.cli text-preservation-surface-curriculum-manifest --output reports\text_rendering_qwen_baseline\surface_curriculum_manifest_v12.json --json
python -m gemmanima.cli text-preservation-qwen-target-refresh-manifest --output reports\text_rendering_qwen_baseline\qwen_target_refresh_manifest_v12.json --json
python -m gemmanima.cli text-preservation-v12-trainer-surface-contract-audit --output reports\text_rendering_qwen_baseline\v12_trainer_surface_contract_audit.json --json
python -m gemmanima.cli text-preservation-v13-recovery-plan --output reports\text_rendering_qwen_baseline\v13_recovery_plan.json --json
python -m gemmanima.cli text-preservation-v13-guard-weighted-manifest --output reports\text_rendering_qwen_baseline\v13_guard_weighted_manifest.json --json
python -m gemmanima.cli text-preservation-v14-focus-fixed-gate-manifest --output reports\text_rendering_qwen_baseline\v16_true_focus_fixed_gate_manifest.json --prompt-file reports\text_rendering_qwen_baseline\v16_true_focus_fixed_gate_prompts.jsonl --target-dir runs\cache\text_preservation_blended_v16\targets --focus-case-id text_eval_005_handwritten_note_meet_at_dawn --focus-case-id text_eval_006_label_tea --json
python -m gemmanima.cli text-preservation-promotion-status --output reports\text_rendering_qwen_baseline\promotion_status.json --compact-output reports\text_rendering_qwen_baseline\promotion_status_compact.json --json
python -m gemmanima.cli text-preservation-release-gate --output reports\text_rendering_qwen_baseline\release_gate_status.json --json
```

## Stage 3: Bridge Training

When every target shard has a matching Gemma shard, run:

```powershell
.\scripts\run_hiddenstage_bridge_train_4070_ti_super.ps1
```

Output:

```text
E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt
```

Training uses the existing Anima adapter bridge objective from `08_train_stream_batched.py`.
Only the Gemma-facing k/v projection path is trainable.

Existing full bridge result outside the isolated PoC1 1k smoke pass:

- Epoch: `2`
- Val MSE: `0.001104317136865575`
- Gate: `0.004`
- Status: passed

## Stage 4: Backend Bridge Integration

The trained bridge checkpoint is registered in the GemmAnima backend:

```text
gemmanima.modules.hiddenstage_exit.HiddenStageExit
gemmanima.modules.bridge_runtime.TrainedBridgeRuntime
```

Smoke command:

```powershell
E:\ComfyUI_sage\python_embeded\python.exe scripts\smoke_hiddenstage_bridge_forward.py --checkpoint E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt
```

Expected smoke output:

```text
input  [1,16,1536]
t5_ids [1,32]
output [1,32,1024]
finite true
```

The normal backend dry-run now records `trained_hiddenstage_bridge` and the checkpoint audit in the manifest.

Run manifests also include observed-only conditioning metrics:

- `conditioning_metrics.measurement_policy`: `observed_only`
- `conditioning_metrics.run_conditioning_mse`: nullable, and `null` unless a run-level teacher/student conditioning distance is actually measured.
- `conditioning_metrics.bridge_val_mse`: nullable bridge checkpoint validation MSE when available.
- `conditioning_metrics.measured`: `false` when no run-level teacher/student conditioning distance exists.

## Renderer Wiring Status

The trained bridge has now been validated through the existing Anima chat-to-image renderer:

```powershell
E:\ComfyUI_sage\python_embeded\python.exe E:\anima_gemma_swap\scripts\core\18_hiddenstage_chat_generate.py --request "Draw Nahida from Genshin Impact as a bright forest anime illustration, gentle expression, detailed green-and-white outfit, soft sunlight." --adapter E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt --out C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\images\nahida_hiddenstage_bridge_real_smoke.png --size 512 --steps 12 --cfg 4.5 --unet-dtype fp8_e4m3fn_fast
```

Observed smoke result:

- GPU: `NVIDIA GeForce RTX 4070 Ti SUPER`
- Steps: `12`
- Output: `runs\images\nahida_hiddenstage_bridge_real_smoke.png`

The repo CLI can print the same command:

```powershell
python -m gemmanima.cli renderer-backends --json
python -m gemmanima.cli real-render-health --json
python -m gemmanima.cli real-render-command --json
```

The backend can also route generation requests through the real external renderer:

```powershell
python -m gemmanima.cli run "draw a bright forest" --renderer real --json
```

The current real renderer is a first-class backend adapter around the legacy external script. The in-process backend now has repo-native bootstrap diagnostics in `gemmanima.rendering.comfy_bootstrap`, repo-native Gemma hidden provider support in `gemmanima.rendering.gemma_hidden`, repo-native T5 tokenizer support in `gemmanima.rendering.t5_tokenizer`, and repo-native sampler/VAE decode support in `gemmanima.rendering.anima_sampler`.

The in-process renderer has passed a 4070 Ti SUPER smoke:

```powershell
$env:CUDA_VISIBLE_DEVICES='0'; $env:GEMMA_EMBED_ON_GPU='1'; E:\ComfyUI_sage\python_embeded\python.exe scripts\smoke_in_process_render.py --image-root runs\images --manifest-root runs\manifests --json
```

Observed output:

- Status: `completed`
- Renderer: `dry_run=false`
- Output: `runs\images\83722a408c6f4b12ab1ae3295c171b70.png`

## Status Commands

```powershell
python -m gemmanima.cli training-readiness --json
python -m gemmanima.cli prepare-gemma-cache --json
python -m gemmanima.cli prepare-bridge-training --json
python -m gemmanima.cli bridge-eval-status --json
python -m gemmanima.cli poc1-cache-plan --json
python -m gemmanima.cli poc1-cache-plan --limit 10000 --gpu-profile 4070-only --subset runs\teacher_targets\poc1_10k_teacher_subset.jsonl --target-dir runs\cache\poc1_10k\targets --gemma-dir runs\cache\poc1_10k\gemma --json
python -m gemmanima.cli poc1-bridge-plan --json
python -m gemmanima.cli poc1-status --json
python -m gemmanima.cli poc1-runtime-status --json
python -m gemmanima.cli text-rendering-eval-pack --json
python -m gemmanima.cli text-rendering-eval-plan --json
python -m gemmanima.cli text-rendering-eval-run-plan --json
python -m gemmanima.cli text-rendering-eval-status --json
python -m gemmanima.cli text-rendering-qwen-baseline-plan --json
python -m gemmanima.cli text-rendering-qwen-baseline-prompts --json
python -m gemmanima.cli text-preservation-bridge-plan --json
python -m gemmanima.cli text-preservation-bridge-status --json
python -m gemmanima.cli text-preservation-prompts --include-eval-cases --json
python -m gemmanima.cli text-preservation-blended-plan --json
python -m gemmanima.cli text-preservation-blended-status --json
python -m gemmanima.cli text-preservation-v5-plan --json
python -m gemmanima.cli text-preservation-v6-plan --json
python -m gemmanima.cli text-preservation-v7-plan --json
python -m gemmanima.cli text-preservation-v8-plan --json
python -m gemmanima.cli text-preservation-promotion-status --json
python -m gemmanima.cli text-preservation-release-gate --json
python -m gemmanima.cli text-preservation-heldout-eval-plan --json
python -m gemmanima.cli text-preservation-general-scene-eval-plan --json
python -m gemmanima.cli bridge-smoke-command --json
python -m gemmanima.cli gemma-hidden-smoke-command --json
python -m gemmanima.cli t5-tokenizer-smoke-command --json
python -m gemmanima.cli in-process-render-smoke-command --json
python -m gemmanima.cli renderer-backends --json
python -m gemmanima.cli real-render-health --json
python -m gemmanima.cli real-render-command --json
```
