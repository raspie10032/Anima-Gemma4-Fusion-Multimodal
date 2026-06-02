# GEMMANIMA Version Roadmap

## v1: Text-to-Anima Conditioning Distillation

Goal: open the basic path from Gemma-side text representation to Anima-compatible conditioning using the existing Anima text encoder path as teacher.

Done or partially done in this repo:

- HiddenStage bridge training and checkpoint audit.
- Teacher target and Gemma cache utilities.
- Teacher/student smoke command scaffolding.
- Manifest writing for generated runs.
- Initial `GEMMANIMA Protocol v0.1` dataclasses and protocol-backed conditioning payloads.
- Schema validation for protocol, conditioning bundle, conflict report, run manifest, and cache-build manifest payloads.
- PoC1 1k cache, bridge-train, forward-smoke, render-smoke, and comparison-report path.
- Field-specific clarification resume for blocked reference/text conflicts.
- Run-manifest `conditioning_metrics` with observed-only policy, nullable run-level conditioning distance, nullable bridge validation MSE, and `measured=false` when no run-level teacher/student conditioning distance exists.
- Dry-run text rendering eval plan and artifact-based status reporting.
- GUI preserve/change controls for conflict fields, implemented as ordinary clarification messages through conversation history.
- PoC1 10k pilot planning command documented with explicit subset, teacher-cache, Gemma-cache paths, and the 4070-only GPU profile.
- PoC1 10k pilot cache and bridge training completed under the training manager with `CUDA_VISIBLE_DEVICES='0'`: target shards `10`, Gemma shards `10`, paired shards `10`, missing shards `0`, bridge Val MSE `0.0020450321631506085`, gate `0.004`, passed.
- Non-GPU-touching `poc1-runtime-status --json` observer for 10k target shards, Gemma shards, paired/missing shards, bridge checkpoint audit, and bridge-training readiness.
- Qwen teacher baseline vs Gemma PoC1 10k comparison artifacts for all 6 text-rendering cases, including prompts, teacher/student images, compare reports, contact sheet, and manual visual review.
- Text-preservation micro-overfit using the existing TE distillation scripts on the 6 Qwen baseline text prompts: bridge Val MSE `0.0006606672153187295`, 1 paired shard, readable text restored on 5 cases and partial stylized text on 1 case.
- Text-preservation blended v4 candidate using 518 text prompts plus 2 PoC1 10k general shards: bridge Val MSE `0.00019171485413001696`, 10 paired hardlinked training shards, readable text on 5 cases and partial stylized text on 1 case, best observed mean image MSE vs Qwen in this pass.
- Text-preservation blended v5 candidate using 1030 no-sample-marker text prompts plus 4 PoC1 10k general shards: bridge Val MSE `0.000139208897962817`, 16 paired hardlinked training shards, fixed 6-case mean image MSE `0.013701042432027558`, and clean 64-case held-out review of 47 readable, 13 partial, 4 failed.
- Text-preservation blended v6 hard-negative experiment completed and rejected for promotion: bridge Val MSE `0.0001442893963030656` passed, but fixed 6-case mean image MSE regressed to `0.021531414221196126` and clean held-out mean image MSE regressed to `0.07254247733180819`.
- General scene regression smoke for v5 completed on 15 Qwen-vs-Gemma cases; v5 produced nonblank general scenes without text-only collapse.
- Expanded v5 general scene regression completed on 50 Qwen-vs-Gemma cases; v5 remained nonblank without text-only collapse. Mean image MSE `0.06805774327367545`, median MSE `0.044416142627596855`.
- Text-preservation blended v7 balanced replay experiment completed and rejected for promotion: bridge Val MSE improved to `0.00011672825917230512`, but fixed 6-case mean image MSE regressed to `0.024075733303713303`.
- Text-preservation blended v8 fixed-gate replay experiment completed and rejected for promotion: bridge Val MSE `0.0006323225916275987` passed, but fixed 6-case mean image MSE regressed to `0.025927293114364147`.
- Non-GPU `text-preservation-promotion-status --json` now compares v5/v6/v7/v8 fixed-gate artifacts and marks v6-v8 as rejected against the v5 protected text-preservation baseline. Durable full and compact reports live under `reports\text_rendering_qwen_baseline`.
- Non-GPU `text-preservation-release-gate --json` now validates the v5 protected baseline across fixed6, held-out, general-scene, and promotion evidence. Current release gate is `pass`, while the v9 training gate is `blocked_until_objective_redesign`.
- Non-GPU `text-preservation-v9-objective-plan --json` now writes the v9 design contract and keeps training blocked with `training_plan.train_command=null` until an artifact-gate-first objective redesign exists.
- Non-GPU `text-preservation-v9-artifact-gate-objective --json` now defines the artifact-gate-first objective contract. Candidate planning is allowed, but GPU training remains blocked until the trainer supports artifact feedback.
- Non-GPU `text-preservation-v9-trainer-support-audit --json` now confirms the external bridge trainer supports artifact feedback after the `08_train_stream_batched.py` patch.
- Text-preservation blended v9 artifact-gate candidate trained on the RTX 4070 Ti SUPER only: bridge Val MSE `0.0005838303222844843` passed, but fixed 6-case mean image MSE regressed to `0.025934352500674624`; do not promote v9.
- Text-preservation blended v10 protected-anchor candidate trained on the RTX 4070 Ti SUPER only: bridge Val MSE `0.0005838341403432423` passed, but fixed 6-case mean image MSE regressed to `0.026236171058068674`; do not promote v10.
- Artifact feedback alignment audit found v9/v10 feedback ids occurred outside fixed-gate shards, so v11 added source-bucket filtering.
- Text-preservation blended v11 source-filtered candidate trained on the RTX 4070 Ti SUPER only: bridge Val MSE `0.0006095190765336156` passed, but fixed 6-case mean image MSE regressed to `0.026408916998965044`; do not promote v11.
- Non-GPU KV delta audit now compares v9/v10/v11 checkpoints against the protected v5 KV tensors. Current element-weighted MSEs are nearly identical: v9 `1.257022133371713e-07`, v10 `1.2565436002131657e-07`, v11 `1.2578555395113333e-07`; the KV anchor/source filter did not materially reduce checkpoint drift.
- Non-GPU `text-preservation-v12-surface-plan --json` now marks the workflow at `v12_surface_redesign`, blocks GPU training, and requires a render/readability-conditioned target refresh before any new candidate run.
- Non-GPU `text-preservation-render-readability-label-manifest --json` now builds the first v12 surface artifact: 70 labeled render records with 17 partial/failed held-out records marked for priority refresh.
- Non-GPU `text-preservation-surface-curriculum-manifest --json` now builds the second v12 surface artifact: 35 curriculum records covering failed refresh, partial refresh, fixed-gate guards, and readable replay guards.
- Non-GPU `text-preservation-qwen-target-refresh-manifest --json` now writes the v12 prompt subset and declares the 4070-only Qwen target cache command for 35 curriculum records.
- `08_train_stream_batched.py` now supports the v12 surface contract, including surface curriculum weights, per-case gate loss budget, and pre-train promotion assertions.
- Text-preservation blended v12 target/Gemma cache and candidate bridge train completed on the RTX 4070 Ti SUPER only. Bridge Val MSE `0.002338456588664225` passed the `0.004` bridge gate, but fixed6 Qwen-vs-v12 image comparison rejected v12: mean image MSE `0.057429684209637344` versus protected v5 `0.013701042432027558`, with 6/6 per-case regressions.
- Added `text-preservation-v13-recovery-plan --json` and generated `reports\text_rendering_qwen_baseline\v13_recovery_plan.json`. v13 is a protected-baseline, low-LR, guard-weighted recovery path; GPU training is blocked until a v13 guard manifest exists.
- Added `text-preservation-v13-guard-weighted-manifest --json`, cached 12 v13 records, and trained a tiny v13 ablation from protected v5 on the RTX 4070 Ti SUPER only. Bridge Val MSE `0.0002798435161821544` passed, but fixed6 mean image MSE `0.016183895776824404` still regressed versus v5 `0.013701042432027558`; v13 is rejected, with remaining LUNA GATE and TEA per-case regressions.
- Trained and evaluated v14, v15, and v16 focus-only fixed-gate ablations from protected v5 on the RTX 4070 Ti SUPER only. v14/v15 fixed6 mean image MSE reached `0.007978032362492135`, but both were rejected by MEET AT DAWN and TEA per-case regressions. v16 enabled true focus weighting with `max_weight=16.0`, produced fixed6 mean image MSE `0.008126146024248252`, and was still rejected on the same two per-case regressions.

Still needed:

- Text rendering remediation promotion: blended v5 is the protected text-preservation baseline. The expanded 50-case general-scene check passed its nonblank/no-collapse smoke gate, but v5 still needs broader semantic and user-facing renderer validation before replacing PoC1 10k or v4 as the default bridge.
- Next large text-preservation training should not rely on replay weighting alone. v6, v7, and v8 show that lower bridge Val MSE, hard-negative focus, or fixed-gate replay can still damage the fixed text-rendering gate.
- v9 through v16 trained successfully where attempted but failed the fixed image-level promotion gate. v5 remains the protected text-preservation baseline; the active next step is a targeted objective/teacher refresh for MEET AT DAWN and TEA rather than another small replay-weight-only ablation.
- Qwen baseline comparison separates teacher capability from bridge regression: Qwen rendered readable target text on 5/6 cases plus one partial stylized-text case, while Gemma PoC1 10k preserved 0 fully readable target-text cases. The micro-overfit checkpoint restored the same 5/6 plus one partial pattern.
- 20-step smoke train and 200-step tiny overfit reports under the new manifest contract.
- Promotion decision for whether the successful PoC1 10k bridge should stay as a pilot checkpoint or feed the next renderer/text-preservation evaluation pass; the RTX 5060 remains reserved for the user.

## v2: Gemma Conditioning Translator

Goal: separate text semantic conditioning from image reference conditioning, then combine them through a small translator/mixer stack.

Required modules:

- `TextConditioningTranslator`
- `SceneConditioningTranslator`
- `VisionReferenceTranslator`
- `ConditioningMixer`
- `ConflictResolver` (first high-severity gate implemented)

Completion criteria:

- Text-only semantic path is stable.
- Image reference smoke test works.
- Image + text split-conditioning demo works.
- Reference drift is measurably lower than text-only.
- High-severity unresolved conflicts stop generation and return a clarification request.

## v3: Protocol-Native Multimodal Conditioning Bridge

Goal: support protocol-to-conditioning generation without depending on prompt strings as the internal bridge format.

Required outcomes:

- `GEMMANIMA Protocol v0.1` is stable.
- Scene, Character, Style, Mood, Reference, Instruction, Conflict, Conditioning, and RunManifest contracts are standardized.
- T5XXL/runtime text encoder dependency becomes optional or fallback.
- External model adapters can change parser inputs without retraining the translator.
- Safe public lineage release path is documented and enforceable.
