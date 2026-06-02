# GEMMANIMA Next Version Execution Plan

Source: `C:\Users\seine\Downloads\GEMMANIMA Next Version Plan.docx`

## Current Repo Reality

The repository is not at a blank v1 starting point. It already has:

- A dry-run backend path from chat routing to manifest writing.
- HiddenStage bridge isolation behind `HiddenStageExit`.
- A trained bridge checkpoint registered at `E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt`.
- Real renderer command wiring and in-process renderer smoke scaffolding.
- Tests covering training readiness, teacher targets, Gemma cache, bridge runtime, renderer adapters, API, GUI, and manifests.

The next-version work should therefore focus on standardizing the protocol and reproducibility contract before adding more training or renderer complexity.

## Immediate Priority

1. Lock `GEMMANIMA Protocol v0.1`.
2. Keep conditioning axes separate: semantic, reference, style, mood, spatial, negative.
3. Make every run manifest record protocol version, precision policy, memory policy, lineage, and frozen/trainable module lists.
4. Add schema files so downstream cache, training, conflict, and release tooling can validate the same contract.
5. Only then continue into cache build, distillation smoke train, tiny overfit, and teacher/student image comparison.

## Completed In This Pass

- Added protocol capsule dataclasses in `gemmanima.core.protocol`.
- Added `ConflictReport` and blocking logic for high-severity conflicts.
- Expanded `ConditioningBundle` to carry split conditioning axes.
- Expanded `Manifest` defaults with protocol, precision, memory, lineage, and module fields.
- Wired `GemmanimaProtocol` into `ContextRelevanceFilter`.
- Added a `ConflictResolver` gate before renderer execution.
- Updated `HiddenStageExit` to copy protocol axes into `ConditioningBundle`.
- Added run manifest schema validation in `ManifestStore.write()`.
- Expanded `ConflictResolver` to cover hair color, outfit, style, identity, and unresolved unsafe-content risk.
- Added schema validation helpers for protocol, conditioning bundle, conflict report, and run manifest payloads.
- Exposed `clarification_required` and `conflict` in API responses so the local GUI can surface blocked generations.
- Split protocol extraction into `ProtocolParser` so context filtering no longer owns protocol parsing heuristics.
- Added first-pass clarification resolution: an explicit user answer such as "change it to black hair" allows the previously blocked change.
- Added a PoC1 `CacheBuildManifest` contract and schema for Gemma/Anima cache build reporting.
- Added `poc1-cache-plan` and `poc1-bridge-plan` CLI commands.
- Ran the isolated PoC1 1k smoke path end to end: subset export, teacher target cache, Gemma hidden cache, 1-shard bridge training, forward smoke, and real render smoke.
- Added teacher/student generation comparison reporting and wrote `reports/poc1_generation_compare_report.json`.
- Added `text-rendering-eval-pack` with sign, book cover, magic circle, UI panel, handwritten note, and label prompts.
- Expanded `ConflictResolver` to return multi-conflict reports in one clarification response.
- Added field-specific clarification resume: short follow-up answers reuse the blocked image request, resolve only the confirmed conflict field, and keep unresolved fields blocked.
- Updated the local GUI to send recent conversation history so clarification replies can resume through the stateless API.
- Added run-manifest `conditioning_metrics` with observed-only policy, nullable `run_conditioning_mse`, nullable `bridge_val_mse`, and `measured=false` when no run-level teacher/student conditioning distance exists.
- Added `text-rendering-eval-plan --json` as a dry-run execution plan that declares deterministic teacher image, student image, and comparison report targets without running GPU commands or inventing metrics.
- Added GUI preserve/change buttons for conflict fields; the buttons send ordinary clarification messages through conversation history.
- Documented the PoC1 10k pilot as a command-ready planning path using explicit subset/cache directories and the `4070-only` GPU profile, while keeping it separate from the executed 1k smoke evidence.
- Added `poc1-runtime-status --json` as a non-GPU-touching observer for the 10k pilot target shards, Gemma shards, paired/missing shards, bridge checkpoint audit, and bridge-training readiness.
- Completed the PoC1 10k pilot cache and bridge training under the training manager with `CUDA_VISIBLE_DEVICES='0'`: target shards `10`, Gemma shards `10`, paired shards `10`, missing shards `0`, bridge Val MSE `0.0020450321631506085`, gate `0.004`, passed. The RTX 5060 remained reserved for the user.
- Added `text-rendering-eval-run-plan --json`, generated all 6 text-rendering teacher/student artifact pairs on the RTX 4070 Ti SUPER, wrote 6 compare reports, and recorded manual visual review in `reports\text_rendering_eval\visual_review.json`. Observed result: object-only prompting prevented person-focused drift, but target text legibility failed on all 6 cases.
- Added the Qwen teacher baseline comparison set for the same 6 text-rendering prompts: Qwen teacher images, Gemma PoC1 10k student images, 6 compare reports, contact sheet, and manual review under `reports\text_rendering_qwen_baseline`. Observed result: Qwen preserved readable text on 5 cases plus 1 partial stylized case, while Gemma PoC1 10k produced 0 fully readable target-text cases.
- Added `text-preservation-bridge-plan --json` and `text-preservation-bridge-status --json`, then ran the existing TE distillation path as a 6-case text-rendering micro-overfit on the RTX 4070 Ti SUPER only. Checkpoint: `runs\cache\text_preservation_qwen\bridge\text_preservation_bridge.pt`, Val MSE `0.0006606672153187295`, gate `0.004`, passed. Qwen-vs-`gemma_text_preservation` manual review restored readable text on 5 cases plus 1 partial stylized case.
- Added the blended text-preservation candidate path and expanded it after the sample-size concern: v4 uses 518 text prompts, 8 repeated text hardlinks, 2 PoC1 10k general hardlinks, and 10 paired training shards. Best observed candidate: `runs\cache\text_preservation_blended_v4\bridge\text_preservation_blended_v4_bridge.pt`, Val MSE `0.00019171485413001696`, gate `0.004`, passed. Manual review: 5 readable text cases, 1 partial stylized case, 0 failed cases.
- Added the v5 text-preservation candidate after held-out validation exposed prompt sample-marker leakage: v5 uses 1030 no-sample-marker text prompts, all 2 generated text shards, 6 text repeats, 4 PoC1 10k general shards, and 16 paired training shards. Checkpoint: `runs\cache\text_preservation_blended_v5\bridge\text_preservation_blended_v5_bridge.pt`, Val MSE `0.000139208897962817`, gate `0.004`, passed. Fixed 6-case mean image MSE vs Qwen improved to `0.013701042432027558`; clean 64-case held-out review is 47 readable, 13 partial, 4 failed.
- Added and evaluated a v6 hard-negative experiment. It trained successfully from v5 with 320 focused hard-negative prompts plus fixed eval replay, but regressed the fixed 6-case mean image MSE to `0.021531414221196126` and the clean held-out mean image MSE to `0.07254247733180819`; do not promote v6.
- Added `text-preservation-general-scene-prompts` and `text-preservation-general-scene-eval-plan`, then ran a 15-case Qwen-vs-v5 general scene smoke. v5 generated nonblank general scenes without text-only collapse; artifacts are under `reports\general_scene_regression_v5`.
- Expanded the v5 general scene regression pack to 50 unique Qwen-vs-Gemma cases. v5 generated nonblank scenes without text-only collapse; artifacts are under `reports\general_scene_regression_v5_50`, with mean image MSE `0.06805774327367545` and median MSE `0.044416142627596855`.
- Added and evaluated a v7 balanced replay experiment. It trained successfully from v5 with v5 text replay, lower-weight hard-negative replay, and all 10 PoC1 general shards, but regressed the fixed 6-case mean image MSE to `0.024075733303713303`; do not promote v7.
- Added and evaluated a v8 fixed-gate replay experiment. It trained successfully from v5 with 8 repeats of the original fixed 6-case cache plus v5 text, hard-negative, and general replay, but regressed the fixed 6-case mean image MSE to `0.025927293114364147`; do not promote v8.
- Added `text-preservation-promotion-status --json` as a non-GPU artifact observer. Current fixed-gate promotion state: v5 protected text-preservation baseline, v6 rejected, v7 rejected, v8 rejected. Durable full and compact reports are written under `reports\text_rendering_qwen_baseline`.
- Added `text-preservation-release-gate --json` as a non-GPU release gate. Current release gate state: v5 protected baseline passes fixed6, held-out, general-scene, and promotion evidence; v9 training is blocked until objective redesign.
- Added `text-preservation-v9-objective-plan --json` as a non-GPU design contract. Durable report: `reports\text_rendering_qwen_baseline\v9_objective_plan.json`; it keeps `training_plan.train_command=null` until an artifact-gate-first objective redesign is implemented and tested.
- Added `text-preservation-v9-artifact-gate-objective --json` as the non-GPU artifact-gate-first objective contract. Durable report: `reports\text_rendering_qwen_baseline\v9_artifact_gate_objective.json`; candidate planning is allowed, but GPU training remains blocked until trainer artifact feedback support exists.
- Added `text-preservation-v9-trainer-support-audit --json` as a non-GPU trainer readiness check. Durable report: `reports\text_rendering_qwen_baseline\v9_trainer_support_audit.json`; after patching `08_train_stream_batched.py`, current status is `supported`.
- Added v9 artifact feedback/config/candidate planning, then trained v9 on the RTX 4070 Ti SUPER only. Checkpoint: `runs\cache\text_preservation_blended_v9\bridge\text_preservation_blended_v9_bridge.pt`; bridge Val MSE `0.0005838303222844843` passed, but fixed 6-case mean image MSE regressed to `0.025934352500674624`; do not promote v9.
- Added v10 protected-anchor candidate planning and trainer KV-anchor regularization, then trained v10 on the RTX 4070 Ti SUPER only. Checkpoint: `runs\cache\text_preservation_blended_v10\bridge\text_preservation_blended_v10_bridge.pt`; bridge Val MSE `0.0005838341403432423` passed, but fixed 6-case mean image MSE regressed to `0.026236171058068674`; do not promote v10.
- Added artifact feedback alignment auditing. v9/v10 feedback ids spilled into v5 text replay and hard-negative replay sources, so v11 added source-bucket filtering.
- Added v11 source-filtered candidate planning and trainer source-bucket feedback filtering, then trained v11 on the RTX 4070 Ti SUPER only. Checkpoint: `runs\cache\text_preservation_blended_v11\bridge\text_preservation_blended_v11_bridge.pt`; bridge Val MSE `0.0006095190765336156` passed, but fixed 6-case mean image MSE regressed to `0.026408916998965044`; do not promote v11.
- Added `text-preservation-kv-delta-audit --json` and generated `reports\text_rendering_qwen_baseline\kv_delta_audit_v9_v10_v11.json`. v9/v10/v11 KV drift against protected v5 is nearly identical: v9 `1.257022133371713e-07`, v10 `1.2565436002131657e-07`, v11 `1.2578555395113333e-07`; the anchor/source-filter path is not enough.
- Added `text-preservation-v12-surface-plan --json` and generated `reports\text_rendering_qwen_baseline\v12_training_surface_plan.json`. Current workflow position is `v12_surface_redesign`; GPU training stays blocked until render/readability-conditioned target refresh artifacts and a trainer surface contract audit exist.
- Added `text-preservation-render-readability-label-manifest --json` and generated `reports\text_rendering_qwen_baseline\render_readability_label_manifest_v12.json`. It contains 70 labeled render records: 6 accepted baseline, 47 readable, 13 partial, and 4 failed; 17 partial/failed held-out records are marked as v12 priority refresh cases.
- Added `text-preservation-surface-curriculum-manifest --json` and generated `reports\text_rendering_qwen_baseline\surface_curriculum_manifest_v12.json`. It contains 35 curriculum records: 4 failed refresh, 13 partial refresh, 6 fixed-gate guards, and 12 readable replay guards.
- Added `text-preservation-qwen-target-refresh-manifest --json` and generated `reports\text_rendering_qwen_baseline\qwen_target_refresh_manifest_v12.json` plus `reports\text_rendering_qwen_baseline\qwen_target_refresh_prompts_v12.jsonl`. The manifest declares a 4070-only Qwen target cache command for 35 curriculum records.
- Added `text-preservation-v12-trainer-surface-contract-audit --json`, patched `08_train_stream_batched.py` for v12 surface curriculum support, per-case gate loss budget, and pre-train promotion assertions, then regenerated `reports\text_rendering_qwen_baseline\v12_trainer_surface_contract_audit.json` with status `supported`.
- Cached v12 Qwen targets and Gemma hidden states on the RTX 4070 Ti SUPER only, then trained `runs\cache\text_preservation_blended_v12\bridge\text_preservation_blended_v12_bridge.pt` from protected v5. Bridge Val MSE `0.002338456588664225` passed the `0.004` bridge gate, but fixed6 Qwen-vs-v12 image comparison rejected v12: mean image MSE `0.057429684209637344` versus protected v5 `0.013701042432027558`, with 6/6 per-case regressions.
- Added `text-preservation-v13-recovery-plan --json` and generated `reports\text_rendering_qwen_baseline\v13_recovery_plan.json`. Current workflow position is `v13_recovery_planning`; GPU training stays blocked until a guard-weighted v13 manifest exists.
- Added `text-preservation-v13-guard-weighted-manifest --json` and generated `reports\text_rendering_qwen_baseline\v13_guard_weighted_manifest.json` plus `reports\text_rendering_qwen_baseline\v13_guard_weighted_prompts.jsonl`. Cached 12 v13 records, trained `runs\cache\text_preservation_blended_v13\bridge\text_preservation_blended_v13_bridge.pt` from protected v5 on the RTX 4070 Ti SUPER only, and observed bridge Val MSE `0.0002798435161821544`.
- Fixed6 Qwen-vs-v13 image comparison rejected v13. Mean image MSE improved from v12 `0.057429684209637344` to `0.016183895776824404`, but still regressed versus protected v5 `0.013701042432027558`; remaining per-case regressions are LUNA GATE and TEA.
- Added and evaluated v14-v16 focus-only fixed-gate ablations from protected v5 on the RTX 4070 Ti SUPER only. v14/v15 lowered fixed6 mean image MSE to `0.007978032362492135`, but both failed MEET AT DAWN and TEA per-case gates. v16 used `reports\text_rendering_qwen_baseline\v16_true_focus_gate_loss_config.json` with `max_weight=16.0` to avoid the trainer's default weight clamp, but still failed the same two per-case gates with fixed6 mean image MSE `0.008126146024248252`.
- Added JSON schemas:
  - `schemas/gemmanima_protocol.schema.json`
  - `schemas/conditioning_bundle.schema.json`
  - `schemas/conflict_report.schema.json`
  - `schemas/run_manifest.schema.json`
  - `schemas/cache_build_manifest.schema.json`

## Next Work Order

1. Replace parser heuristics with model/adapter-backed protocol extraction when the upstream Gemma parser contract is ready.
2. Keep blended v5 as the protected text-preservation baseline. The 50-case general-scene smoke is now complete, but v5 still needs broader semantic validation and renderer/user-facing checks before replacing PoC1 10k or v4 as the default bridge.
3. Use the completed PoC1 10k bridge checkpoint as the next observed pilot baseline:
   `runs\cache\poc1_10k\bridge\poc1_10k_bridge.pt`.
4. Before another large text-preservation run, treat v9 through v16 as rejected evidence. v5 remains protected. The active next step is a targeted objective/teacher refresh for MEET AT DAWN and TEA, not another replay-weight-only ablation.
5. Keep future GPU automation on the RTX 4070 Ti SUPER with `CUDA_VISIBLE_DEVICES='0'` unless the user explicitly changes the reservation. The RTX 5060 is reserved for the user and must not be targeted by automation.
6. Recheck pilot artifacts with `python -m gemmanima.cli poc1-runtime-status --json` before any promotion or renderer comparison work; treat the output as observational only.
