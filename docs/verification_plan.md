# GEMMANIMA Verification Plan

## Current Baseline

The repository test suite is the first gate:

```powershell
python -m pytest -q
```

## PoC Gates

1. Environment smoke test
   - GPU detected.
   - VRAM usage recorded.
   - Anima fp16/bf16 load audited.
   - Gemma quantized load audited.
   - Dummy translator forward works.
   - Manifest saved.

2. TE distillation cache build
   - 1k sample cache generated.
   - Shape, dtype, and device recorded.
   - Failed sample list saved.
   - Cache manifest saved.

3. TextConditioningTranslator smoke train
   - 20 steps complete.
   - Loss decreases.
   - No OOM.
   - VRAM and step time recorded.

4. Tiny overfit
   - 200 steps complete.
   - Conditioning distance decreases.
   - Student conditioning can run Anima forward.

5. Teacher/student image compare
   - Teacher output image saved.
   - Student output image saved.
   - Same prompt and seed are recorded.
   - Conditioning distance is recorded only when it is actually measured.
   - Run manifests use observed-only `conditioning_metrics`; `run_conditioning_mse` is nullable and `measured=false` when no run-level teacher/student conditioning distance exists.

6. Conflict resolver demo
   - Silver-hair reference plus black-hair instruction creates a high-severity conflict.
   - `requires_user_confirmation` is true.
   - API response sets `clarification_required` to true.
   - Renderer is not executed.
   - A short follow-up clarification such as `Change it to black hair` resumes the blocked image request.
   - Field-specific clarification clears only the confirmed conflict and leaves unrelated conflicts blocked.
   - GUI preserve/change buttons send ordinary clarification messages through conversation history.

## Current Protocol Gate Checks

```powershell
python -m pytest tests\test_protocol_execution.py tests\test_protocol_contracts.py -q
```

This covers protocol construction, split conditioning axes, conflict blocking, and manifest schema validation.

Cache manifest validation:

```powershell
python -m pytest tests\test_cache_manifest.py -q
```

Current full regression gate:

```powershell
python -m pytest -q
python -m compileall -q gemmanima
```

Latest observed result: `163 passed`.

PoC1 runtime verification completed:

- `python -m gemmanima.cli poc1-cache-plan --json`
- `python -m gemmanima.cli prepare-teacher-targets --limit 1000 --json`
- `E:\ComfyUI_sage\python_embeded\python.exe E:\anima_gemma_swap\scripts\core\06_cache_targets.py ... --shard 1000 --resume`
- `E:\ComfyUI_sage\python_embeded\python.exe E:\anima_gemma_swap\scripts\core\07_cache_gemma_batched.py ... --batch-size 8 --resume`
- `E:\ComfyUI_sage\python_embeded\python.exe E:\anima_gemma_swap\scripts\core\08_train_stream_batched.py ... --limit-shards 1`
- `E:\ComfyUI_sage\python_embeded\python.exe scripts\smoke_hiddenstage_bridge_forward.py --checkpoint runs\cache\poc1_1k\bridge\poc1_bridge.pt`
- `E:\ComfyUI_sage\python_embeded\python.exe E:\anima_gemma_swap\scripts\core\18_hiddenstage_chat_generate.py ... --adapter runs\cache\poc1_1k\bridge\poc1_bridge.pt`
- `python -m gemmanima.cli write-compare-report ... --output reports\poc1_generation_compare_report.json`
- `python -m gemmanima.cli poc1-status --json`
- `python -m gemmanima.cli text-rendering-eval-pack --json`
- `python -m gemmanima.cli text-rendering-eval-plan --json`
- `python -m gemmanima.cli text-rendering-eval-status --json`
- `python -m gemmanima.cli text-rendering-eval-run-plan --json`

PoC1 10k planning and runtime-status verification:

- `python -m gemmanima.cli poc1-cache-plan --limit 10000 --gpu-profile 4070-only --subset runs\teacher_targets\poc1_10k_teacher_subset.jsonl --target-dir runs\cache\poc1_10k\targets --gemma-dir runs\cache\poc1_10k\gemma --json`
- The command is expected to print planned subset export, teacher target cache, Gemma hidden cache, and cache-manifest write commands.
- Generated runtime commands must target the RTX 4070 Ti SUPER only and must be run by the training manager with `CUDA_VISIBLE_DEVICES='0'`.
- The plan must not target CUDA device 1 / RTX 5060. The RTX 5060 is reserved for the user and unavailable to automation.
- The command must not run GPU cache generation, bridge training, rendering, or image comparison.
- 10k cache and bridge checkpoint completion may be marked only after the generated commands actually run and their artifacts are observed.
- `python -m gemmanima.cli poc1-runtime-status --json` is non-GPU-touching and may be used while the 4070-only training manager runs.
- The status command observes default 10k target, Gemma, and bridge checkpoint paths, then reports target shards, Gemma shards, paired/missing shards, bridge checkpoint audit, and `ready_for_bridge_training`.
- Latest observed 10k status: target shards `10`, Gemma shards `10`, paired shards `10`, missing Gemma shards `0`.
- Latest observed 10k bridge checkpoint: `runs\cache\poc1_10k\bridge\poc1_10k_bridge.pt`, `val_mse=0.0020450321631506085`, gate `0.004`, passed.

Text rendering eval status expectations:

- The command summarizes the deterministic prompt pack.
- Each case reports expected teacher image, student image, and compare report paths.
- The dry-run plan declares render and compare targets only; it does not execute GPU commands.
- Missing artifacts keep the case pending.
- The report does not fake OCR scores, image metrics, or pass/fail outcomes before artifacts exist.

Observed text rendering eval result:

- All 6 teacher/student image pairs and compare reports were generated under `runs\images\text_rendering_eval` and `reports\text_rendering_eval`.
- `python -m gemmanima.cli text-rendering-eval-status --json` reported `ready_cases=6`, `pending_cases=0`.
- Image metrics are observed from generated images; null conditioning distances are filtered out of status metrics.
- Contact sheet: `reports\text_rendering_eval\contact_sheet.png`.
- Manual visual review report: `reports\text_rendering_eval\visual_review.json`.
- Manual visual review result: target text legibility failed on all 6 cases.

Qwen baseline artifact check:

- `python -m gemmanima.cli text-rendering-qwen-baseline-plan --json`
- `python -m gemmanima.cli text-rendering-qwen-baseline-prompts --json`
- Qwen teacher baseline prompts exist at `reports\text_rendering_qwen_baseline\prompts.jsonl`.
- Qwen teacher and Gemma PoC1 10k student images exist for all 6 text-rendering cases under `runs\images\text_rendering_qwen_baseline`.
- Six compare reports, a contact sheet, and manual review exist under `reports\text_rendering_qwen_baseline`.
- Manual review result: Qwen teacher produced readable text for 5 cases and partial stylized ring text for 1 case; Gemma PoC1 10k produced 0 fully readable target-text cases. Image metrics are observational and do not replace OCR/manual legibility review.

Text-preservation micro-overfit check:

- `python -m gemmanima.cli text-preservation-bridge-plan --json`
- `python -m gemmanima.cli text-preservation-bridge-status --json`
- Target and Gemma text-preservation caches exist under `runs\cache\text_preservation_qwen` with 1 paired shard.
- Bridge checkpoint exists at `runs\cache\text_preservation_qwen\bridge\text_preservation_bridge.pt`.
- Latest observed text-preservation bridge Val MSE: `0.0006606672153187295`, gate `0.004`, passed.
- Qwen-vs-`gemma_text_preservation` images and compare reports exist for all 6 text-rendering cases.
- Contact sheet: `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation.png`.
- Manual review report: `reports\text_rendering_qwen_baseline\visual_review_text_preservation.json`.
- Manual review result: readable target text restored on 5 cases, partial stylized text on 1 case, 0 failed cases. Treat this as overfit evidence, not broad promotion evidence.

Text-preservation blended candidate check:

- `python -m gemmanima.cli text-preservation-prompts --include-eval-cases --json`
- `python -m gemmanima.cli text-preservation-blended-plan --json`
- `python -m gemmanima.cli text-preservation-blended-status --json`
- Best observed candidate path: `runs\cache\text_preservation_blended_v4\bridge\text_preservation_blended_v4_bridge.pt`.
- The v4 blended run used 518 text prompts, 8 text-shard hardlinks, 2 PoC1 10k general shard hardlinks, and 10 paired training shards.
- Latest observed v4 bridge Val MSE: `0.00019171485413001696`, gate `0.004`, passed.
- Qwen-vs-`gemma_text_preservation_blended_v4` images and compare reports exist for all 6 text-rendering cases.
- Contact sheet: `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v4.png`.
- Manual review report: `reports\text_rendering_qwen_baseline\visual_review_text_preservation_blended_v4.json`.
- Manual review result: readable target text on 5 cases, partial stylized text on 1 case, 0 failed cases. This is the current best blended text-preservation candidate, not yet a broad release checkpoint.

Text-preservation v5 held-out candidate check:

- `python -m gemmanima.cli text-preservation-v5-plan --json`
- `python -m gemmanima.cli text-preservation-heldout-eval-plan --count 64 --prompt-file reports\text_preservation_heldout_v5_clean\prompts.jsonl --out-root runs\images\text_preservation_heldout_v5_clean --report-root reports\text_preservation_heldout_v5_clean --student-checkpoint runs\cache\text_preservation_blended_v5\bridge\text_preservation_blended_v5_bridge.pt --student-name gemma_text_preservation_blended_v5 --prompt-index-offset 30000 --src-prefix text_preserve_heldout_clean --no-sample-marker --json`
- Latest observed v5 bridge checkpoint: `runs\cache\text_preservation_blended_v5\bridge\text_preservation_blended_v5_bridge.pt`, `val_mse=0.000139208897962817`, gate `0.004`, passed.
- Fixed 6-case Qwen-vs-v5 images and compare reports exist under `runs\images\text_rendering_qwen_baseline` and `reports\text_rendering_qwen_baseline`.
- Fixed 6-case mean image MSE vs Qwen: `0.013701042432027558`.
- Clean 64-case held-out Qwen-vs-v5 images and compare reports exist under `runs\images\text_preservation_heldout_v5_clean` and `reports\text_preservation_heldout_v5_clean`.
- Clean held-out manual review report: `reports\text_preservation_heldout_v5_clean\visual_review.json`.
- Clean held-out manual review result: 47 readable, 13 partial, 4 failed. Treat v5 as the current text-preservation candidate, with broader semantic regression still required before default promotion.
- General scene smoke artifacts: `runs\images\poc1_general_regression_v5.png`, `reports\poc1_general_regression_v5_compare.json`, and `reports\poc1_general_regression_v5_review.json`.

Text-preservation v6 hard-negative check:

- `python -m gemmanima.cli text-preservation-v6-plan --json`
- Latest observed v6 bridge checkpoint: `runs\cache\text_preservation_blended_v6\bridge\text_preservation_blended_v6_bridge.pt`, `val_mse=0.0001442893963030656`, gate `0.004`, passed.
- Fixed 6-case Qwen-vs-v6 compare reports exist under `reports\text_rendering_qwen_baseline`.
- Fixed 6-case mean image MSE vs Qwen: `0.021531414221196126`, worse than v5 `0.013701042432027558`.
- Clean 64-case held-out v6 compare reports exist under `reports\text_preservation_heldout_v6_clean`.
- Clean held-out mean image MSE vs Qwen: `0.07254247733180819`, worse than v5 `0.03992582718410631`.
- Verification decision: v6 is not a promotion candidate.

Text-preservation v7 balanced replay check:

- `python -m gemmanima.cli text-preservation-v7-plan --json`
- Latest observed v7 bridge checkpoint: `runs\cache\text_preservation_blended_v7\bridge\text_preservation_blended_v7_bridge.pt`, `val_mse=0.00011672825917230512`, gate `0.004`, passed.
- Fixed 6-case Qwen-vs-v7 compare reports exist under `reports\text_rendering_qwen_baseline`.
- Fixed 6-case mean image MSE vs Qwen: `0.024075733303713303`, worse than v5 `0.013701042432027558` and v6 `0.021531414221196126`.
- Verification decision: v7 is not a promotion candidate. Because the fixed text gate regressed, skip broader v7 held-out/general expansion and keep v5 as the current candidate.

Text-preservation v8 fixed-gate replay check:

- `python -m gemmanima.cli text-preservation-v8-plan --json`
- `python -m gemmanima.cli text-preservation-promotion-status --json`
- Latest observed v8 bridge checkpoint: `runs\cache\text_preservation_blended_v8\bridge\text_preservation_blended_v8_bridge.pt`, `val_mse=0.0006323225916275987`, gate `0.004`, passed.
- Fixed 6-case Qwen-vs-v8 compare reports exist under `reports\text_rendering_qwen_baseline`.
- Fixed 6-case mean image MSE vs Qwen: `0.025927293114364147`, worse than v5 `0.013701042432027558`.
- Verification decision: v8 is not a promotion candidate. Because the fixed text gate regressed, skip broader v8 held-out/general expansion and keep v5 as the current candidate.

Text-preservation promotion status check:

- `python -m gemmanima.cli text-preservation-promotion-status --json`
- `python -m gemmanima.cli text-preservation-promotion-status --output reports\text_rendering_qwen_baseline\promotion_status.json --compact-output reports\text_rendering_qwen_baseline\promotion_status_compact.json --json`
- The command must be non-GPU-touching and report `executes_gpu_commands=false`.
- Current observed fixed-gate decisions: v5 is the baseline; v6, v7, and v8 are rejected because their fixed 6-case mean image MSE is worse than v5.
- The fixed-gate guard compares required report count, mean MSE, max MSE, and per-case MSE against the v5 protected baseline.
- Durable reports: `reports\text_rendering_qwen_baseline\promotion_status.json` and `reports\text_rendering_qwen_baseline\promotion_status_compact.json`.

Text-preservation release gate check:

- `python -m gemmanima.cli text-preservation-release-gate --json`
- `python -m gemmanima.cli text-preservation-release-gate --output reports\text_rendering_qwen_baseline\release_gate_status.json --json`
- The command must be non-GPU-touching and report `executes_gpu_commands=false`.
- Current observed release gate: `pass` for protected baseline `v5`.
- Required evidence: fixed 6 reports, v5 held-out metrics/review, 50-case general scene metrics/review, and promotion status recommendation `protect_baseline`.
- Current observed v9 training gate: `blocked_until_objective_redesign`. Do not start another replay-weighting training run until a new objective or image/text-level training feedback path is designed.

Text-preservation v9 objective plan check:

- `python -m gemmanima.cli text-preservation-v9-objective-plan --json`
- `python -m gemmanima.cli text-preservation-v9-objective-plan --output reports\text_rendering_qwen_baseline\v9_objective_plan.json --json`
- The command must be non-GPU-touching and report `executes_gpu_commands=false`.
- The report must keep `training_plan.train_command=null` while the objective status is `blocked_until_artifact_gate_first_objective_redesign`.
- Expected next safe action before GPU work: implement the `artifact_gate_first` objective contract, then add tests that explicitly permit a candidate training plan.

Text-preservation v9 artifact-gate objective check:

- `python -m gemmanima.cli text-preservation-v9-artifact-gate-objective --json`
- `python -m gemmanima.cli text-preservation-v9-artifact-gate-objective --output reports\text_rendering_qwen_baseline\v9_artifact_gate_objective.json --json`
- The command must be non-GPU-touching and report `executes_gpu_commands=false`.
- Expected candidate planning status: `allowed`.
- Expected GPU training status: `blocked_until_trainer_supports_artifact_feedback`, with `train_command=null`.

Text-preservation v9 trainer support audit:

- `python -m gemmanima.cli text-preservation-v9-trainer-support-audit --json`
- `python -m gemmanima.cli text-preservation-v9-trainer-support-audit --output reports\text_rendering_qwen_baseline\v9_trainer_support_audit.json --json`
- The command must be non-GPU-touching and report `executes_gpu_commands=false`.
- Current observed trainer support after patching `08_train_stream_batched.py`: `supported`.
- The trainer now accepts `--artifact-feedback` and `--artifact-gate-loss-config` and can ingest post-render artifact feedback as sample weights.

Text-preservation v9 candidate check:

- `python -m gemmanima.cli text-preservation-v9-artifact-feedback --output reports\text_rendering_qwen_baseline\v9_artifact_feedback.jsonl --json`
- `python -m gemmanima.cli text-preservation-v9-artifact-gate-loss-config --output reports\text_rendering_qwen_baseline\v9_artifact_gate_loss_config.json --json`
- `python -m gemmanima.cli text-preservation-v9-candidate-plan --json`
- Candidate checkpoint: `runs\cache\text_preservation_blended_v9\bridge\text_preservation_blended_v9_bridge.pt`.
- Observed bridge Val MSE: `0.0005838303222844843`, gate `0.004`, passed.
- Fixed 6-case mean image MSE: `0.025934352500674624`; this is worse than v5 `0.013701042432027558`, so v9 is rejected for promotion.

Text-preservation v10 protected-anchor candidate check:

- `python -m gemmanima.cli text-preservation-v10-candidate-plan --json`
- v10 adds KV-anchor regularization against the protected v5 checkpoint while using artifact feedback.
- Training must target the RTX 4070 Ti SUPER only with `CUDA_VISIBLE_DEVICES='0'`; do not target the RTX 5060.
- Candidate checkpoint: `runs\cache\text_preservation_blended_v10\bridge\text_preservation_blended_v10_bridge.pt`.
- Observed bridge Val MSE: `0.0005838341403432423`, gate `0.004`, passed.
- Fixed 6-case mean image MSE: `0.026236171058068674`; this is worse than v5 `0.013701042432027558`, so v10 is rejected for promotion.
- Next verification before v11: audit whether artifact feedback weights align to the intended blended shard sample ids.

Text-preservation artifact feedback alignment audit:

- `python -m gemmanima.cli text-preservation-artifact-feedback-alignment-audit --json`
- Durable reports: `reports\text_rendering_qwen_baseline\v9_artifact_feedback_alignment_audit.json` and `reports\text_rendering_qwen_baseline\v10_artifact_feedback_alignment_audit.json`.
- Current observed result: feedback ids occurred 66 times in the v10 blend, including spillover into `10_v5_text` and `20_hard_negative` replay sources.

Text-preservation v11 source-filtered candidate check:

- `python -m gemmanima.cli text-preservation-v11-candidate-plan --json`
- v11 limits artifact feedback weighting to `00_fixed_gate` source buckets while keeping the v5 KV anchor.
- Candidate checkpoint: `runs\cache\text_preservation_blended_v11\bridge\text_preservation_blended_v11_bridge.pt`.
- Observed bridge Val MSE: `0.0006095190765336156`, gate `0.004`, passed.
- Fixed 6-case mean image MSE: `0.026408916998965044`; this is worse than v5 `0.013701042432027558`, so v11 is rejected for promotion.

Text-preservation KV delta audit:

- `python -m gemmanima.cli text-preservation-kv-delta-audit --output reports\text_rendering_qwen_baseline\kv_delta_audit_v9_v10_v11.json --json`
- Durable report: `reports\text_rendering_qwen_baseline\kv_delta_audit_v9_v10_v11.json`.
- Current observed result: v9/v10/v11 all share 12 comparable KV tensors against v5. Element-weighted MSEs are v9 `1.257022133371713e-07`, v10 `1.2565436002131657e-07`, and v11 `1.2578555395113333e-07`.
- Verification decision: v10 and v11 are not materially changed versus v9 in KV drift, so the v5 KV anchor/source filter did not provide a useful promotion path.

Text-preservation v12 surface redesign gate:

- `python -m gemmanima.cli text-preservation-v12-surface-plan --output reports\text_rendering_qwen_baseline\v12_training_surface_plan.json --json`
- Durable report: `reports\text_rendering_qwen_baseline\v12_training_surface_plan.json`.
- Current workflow position: `v12_surface_redesign`; next safe step: `build_render_readability_label_manifest`.
- Verification decision: GPU training remains blocked until the render/readability label manifest, surface curriculum manifest, Qwen target refresh manifest, fixed6 baseline map, held-out partial/failed case pack, and trainer surface contract audit exist.

Text-preservation v12 render/readability label manifest:

- `python -m gemmanima.cli text-preservation-render-readability-label-manifest --output reports\text_rendering_qwen_baseline\render_readability_label_manifest_v12.json --json`
- Durable report: `reports\text_rendering_qwen_baseline\render_readability_label_manifest_v12.json`.
- Current observed result: 70 labeled render records, with 6 accepted baseline fixed-gate cases, 47 readable held-out cases, 13 partial held-out cases, and 4 failed held-out cases.
- Verification decision: the 17 partial/failed held-out cases are now explicit `v12_priority_refresh` records. GPU training remains blocked until the surface curriculum manifest exists.

Text-preservation v12 surface curriculum manifest:

- `python -m gemmanima.cli text-preservation-surface-curriculum-manifest --output reports\text_rendering_qwen_baseline\surface_curriculum_manifest_v12.json --json`
- Durable report: `reports\text_rendering_qwen_baseline\surface_curriculum_manifest_v12.json`.
- Current observed result: 35 curriculum records, with 4 failed refresh, 13 partial refresh, 6 fixed-gate guards, and 12 readable replay guards.
- Verification decision: the curriculum is ready for the next non-GPU target refresh planning step. GPU training remains blocked until the Qwen target refresh manifest and trainer surface contract audit exist.

Text-preservation v12 Qwen target refresh manifest:

- `python -m gemmanima.cli text-preservation-qwen-target-refresh-manifest --output reports\text_rendering_qwen_baseline\qwen_target_refresh_manifest_v12.json --json`
- Durable report: `reports\text_rendering_qwen_baseline\qwen_target_refresh_manifest_v12.json`.
- Prompt subset: `reports\text_rendering_qwen_baseline\qwen_target_refresh_prompts_v12.jsonl`, with 35 records.
- Verification decision: the target cache command is 4070-only via `CUDA_VISIBLE_DEVICES='0'`. GPU training remains blocked until targets are cached and the trainer surface contract audit exists.

Text-preservation v12 trainer surface contract audit:

- `python -m gemmanima.cli text-preservation-v12-trainer-surface-contract-audit --output reports\text_rendering_qwen_baseline\v12_trainer_surface_contract_audit.json --json`
- Durable report: `reports\text_rendering_qwen_baseline\v12_trainer_surface_contract_audit.json`.
- Current observed result after patching `08_train_stream_batched.py`: `supported`.

Text-preservation v12 fixed6 render gate:

- Fixed6 Qwen-vs-v12 compare reports exist under `reports\text_rendering_qwen_baseline\text_eval_*_qwen_vs_gemma_text_preservation_blended_v12_compare.json`.
- Durable summary: `reports\text_rendering_qwen_baseline\metrics_summary_v12_fixed6.json`.
- Contact sheet: `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v12_fixed6.png`.
- Current observed result: mean image MSE `0.057429684209637344` versus protected v5 `0.013701042432027558`; all 6 protected fixed cases regressed.
- Verification decision: v12 is rejected. The release gate still protects v5 and pending candidates are empty.

Text-preservation v13 recovery plan:

- `python -m gemmanima.cli text-preservation-v13-recovery-plan --output reports\text_rendering_qwen_baseline\v13_recovery_plan.json --json`
- Durable report: `reports\text_rendering_qwen_baseline\v13_recovery_plan.json`.
- Current workflow position: `v13_recovery_planning`; next safe step: `build_guard_weighted_v13_manifest`.
- Verification decision: GPU training remains blocked until the v13 guard-weighted manifest exists. Any future GPU run must use `CUDA_VISIBLE_DEVICES='0'` and must not target the RTX 5060.

Text-preservation v13 guard-weighted tiny ablation:

- `python -m gemmanima.cli text-preservation-v13-guard-weighted-manifest --output reports\text_rendering_qwen_baseline\v13_guard_weighted_manifest.json --json`
- Durable manifest: `reports\text_rendering_qwen_baseline\v13_guard_weighted_manifest.json`.
- Prompt subset: `reports\text_rendering_qwen_baseline\v13_guard_weighted_prompts.jsonl`, with 12 records.
- v13 target/Gemma cache completed on the RTX 4070 Ti SUPER only: `runs\cache\text_preservation_blended_v13\targets\shard_0000.pt` and `runs\cache\text_preservation_blended_v13\gemma\shard_0000.pt`.
- v13 checkpoint: `runs\cache\text_preservation_blended_v13\bridge\text_preservation_blended_v13_bridge.pt`.
- Bridge Val MSE: `0.0002798435161821544`, gate `0.004`, passed.
- Fixed6 summary: `reports\text_rendering_qwen_baseline\metrics_summary_v13_fixed6.json`; contact sheet: `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v13_fixed6.png`.
- Verification decision: v13 is rejected despite recovering much of v12's drift. Mean image MSE is `0.016183895776824404` versus protected v5 `0.013701042432027558`, with remaining per-case regressions on LUNA GATE and TEA.

Text-preservation v14-v16 focus-only fixed gate ablations:

- v14 checkpoint: `runs\cache\text_preservation_blended_v14\bridge\text_preservation_blended_v14_bridge.pt`; fixed6 summary: `reports\text_rendering_qwen_baseline\metrics_summary_v14_fixed6.json`; contact sheet: `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v14_fixed6.png`.
- v14 decision: rejected. Mean image MSE `0.007978032362492135` beats protected v5 mean `0.013701042432027558`, but MEET AT DAWN and TEA still regress per-case.
- v15 checkpoint: `runs\cache\text_preservation_blended_v15\bridge\text_preservation_blended_v15_bridge.pt`; fixed6 summary: `reports\text_rendering_qwen_baseline\metrics_summary_v15_fixed6.json`; contact sheet: `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v15_fixed6.png`.
- v15 decision: rejected for the same MEET AT DAWN and TEA per-case regressions. Investigation found that the trainer's default artifact gate config clamps weights to `max_weight=4.0`, so the intended guard/focus weights `6` and `16` were both clipped.
- v16 checkpoint: `runs\cache\text_preservation_blended_v16\bridge\text_preservation_blended_v16_bridge.pt`; fixed6 summary: `reports\text_rendering_qwen_baseline\metrics_summary_v16_fixed6.json`; contact sheet: `reports\text_rendering_qwen_baseline\contact_sheet_text_preservation_blended_v16_fixed6.png`.
- v16 decision: rejected. The no-BOM JSON config `reports\text_rendering_qwen_baseline\v16_true_focus_gate_loss_config.json` raises `max_weight` to `16.0`, and the checkpoint hash differs from v15, but fixed6 mean image MSE `0.008126146024248252` still leaves MEET AT DAWN and TEA as per-case regressions.
- Current promotion observer: `reports\text_rendering_qwen_baseline\promotion_status.json` protects v5 and rejects v6-v16. Current release gate: `reports\text_rendering_qwen_baseline\release_gate_status.json` still passes for the protected v5 baseline and keeps the next training gate blocked until an objective redesign beats the fixed image/text gates.
- v12 target/Gemma cache status: 1 target shard, 1 Gemma shard, 1 paired shard, 35 examples.
- v12 candidate checkpoint: `runs\cache\text_preservation_blended_v12\bridge\text_preservation_blended_v12_bridge.pt`.
- Observed bridge Val MSE: `0.002338456588664225`; bridge MSE gate `0.004`; gate passed.
- Verification decision: v12 is only a pending bridge candidate. It must pass fixed6 Qwen-vs-v12 image/render comparison before any promotion consideration.

General scene regression check:

- `python -m gemmanima.cli text-preservation-general-scene-eval-plan --json`
- General scene prompt file: `reports\general_scene_regression_v5\prompts.jsonl`.
- Qwen-vs-v5 images and compare reports exist under `runs\images\general_scene_regression_v5` and `reports\general_scene_regression_v5`.
- Contact sheet: `reports\general_scene_regression_v5\contact_sheet_all15.png`.
- Manual review report: `reports\general_scene_regression_v5\visual_review.json`.
- Observed result: v5 generated nonblank general scenes across all 15 cases without text-only collapse. Treat this as a smoke pass, not full semantic promotion coverage.

Expanded general scene regression check:

- `python -m gemmanima.cli text-preservation-general-scene-eval-plan --count 50 --prompt-file reports\general_scene_regression_v5_50\prompts.jsonl --out-root runs\images\general_scene_regression_v5_50 --report-root reports\general_scene_regression_v5_50 --student-checkpoint runs\cache\text_preservation_blended_v5\bridge\text_preservation_blended_v5_bridge.pt --student-name gemma_text_preservation_blended_v5 --json`
- Qwen-vs-v5 images and 50 compare reports exist under `runs\images\general_scene_regression_v5_50` and `reports\general_scene_regression_v5_50`.
- Contact sheet: `reports\general_scene_regression_v5_50\contact_sheet_all50.png`.
- Review report: `reports\general_scene_regression_v5_50\visual_review.json`.
- Observed result: v5 generated nonblank general scenes across all 50 cases without text-only collapse. Mean image MSE vs Qwen: `0.06805774327367545`; median MSE: `0.044416142627596855`.

Text rendering eval prompt categories:

- sign
- book_cover
- magic_circle
- UI_panel
- handwritten_note
- label
