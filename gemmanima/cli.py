from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemmanima import GemmAnimaConductor
from gemmanima.core.manifest import ManifestStore
from gemmanima.core.model_registry import ModelRegistry
from gemmanima.training.readiness import build_training_readiness_report, write_training_readiness_report
from gemmanima.training.gemma_cache import audit_cache_pairing, default_split_gemma_cache_plans
from gemmanima.training.cache_manifest import CacheBuildManifest, write_cache_build_manifest
from gemmanima.training.comparison import GenerationCompareReport, write_compare_report
from gemmanima.training.candidate_workflow import (
    build_candidate_objective_manifest,
    build_candidate_promotion_bundle,
    build_candidate_workflow_status,
    write_candidate_objective_manifest,
    write_candidate_promotion_bundle,
    write_candidate_workflow_status,
)
from gemmanima.training.image_state_conditioning import (
    build_image_state_conditioning_plan,
    build_image_state_subset,
    write_image_state_conditioning_plan,
)
from gemmanima.training.image_state_fusion_diagnostics import write_conditioning_fusion_guard_manifest
from gemmanima.training.image_state_fusion_diagnostics import write_conditioning_fusion_preflight_manifest
from gemmanima.training.image_state_fusion_diagnostics import write_image_state_replay_training_objective
from gemmanima.training.poc1_cache import (
    DEFAULT_POC1_GEMMA_DIR,
    DEFAULT_POC1_BRIDGE_OUT,
    DEFAULT_POC1_SUBSET,
    DEFAULT_POC1_TARGET_DIR,
    build_poc1_bridge_plan,
    build_poc1_cache_plan,
)
from gemmanima.training.poc1_status import (
    DEFAULT_COMPARE_REPORT,
    DEFAULT_POC1_10K_BRIDGE_CHECKPOINT,
    DEFAULT_POC1_10K_GEMMA_DIR,
    DEFAULT_POC1_10K_TARGET_DIR,
    DEFAULT_RUNTIME_REPORT,
    build_poc1_runtime_status,
    build_poc1_status,
)
from gemmanima.training.text_rendering_eval import (
    build_text_rendering_eval_execution_plan,
    build_text_rendering_eval_pack,
    build_text_rendering_eval_run_plan,
    build_text_rendering_eval_status,
    build_text_rendering_qwen_baseline_plan,
    write_text_rendering_qwen_prompt_file,
)
from gemmanima.training.text_preservation import (
    build_general_scene_regression_eval_plan,
    build_general_scene_regression_prompt_records,
    build_text_preservation_artifact_feedback_alignment_audit,
    build_text_preservation_heldout_eval_plan,
    build_text_preservation_heldout_prompt_records,
    build_text_preservation_blended_plan,
    build_text_preservation_bridge_plan,
    build_text_preservation_bridge_status,
    build_text_preservation_compact_promotion_status,
    build_text_preservation_v9_artifact_feedback_dataset,
    build_text_preservation_v9_artifact_gate_loss_config,
    build_text_preservation_v10_candidate_plan,
    build_text_preservation_v11_artifact_gate_loss_config,
    build_text_preservation_v11_candidate_plan,
    build_text_preservation_kv_delta_audit,
    build_text_preservation_v12_surface_plan,
    build_text_preservation_render_readability_label_manifest,
    build_text_preservation_surface_curriculum_manifest,
    build_text_preservation_qwen_target_refresh_manifest,
    build_text_preservation_v12_trainer_surface_contract_audit,
    build_text_preservation_v13_recovery_plan,
    build_text_preservation_v13_guard_weighted_manifest,
    build_text_preservation_v14_focus_fixed_gate_manifest,
    build_text_preservation_v17_targeted_teacher_refresh_manifest,
    build_text_preservation_v18_tea_micro_refresh_manifest,
    build_text_preservation_v19_dual_guard_refresh_manifest,
    build_text_preservation_v23_hard_heldout_refresh_manifest,
    build_text_preservation_v24_fixed_gate_protected_heldout_refresh_manifest,
    build_text_preservation_prompt_records,
    build_text_preservation_promotion_status,
    build_text_preservation_release_gate_status,
    build_text_preservation_v9_artifact_gate_objective,
    build_text_preservation_v9_candidate_plan,
    build_text_preservation_v9_objective_plan,
    build_text_preservation_v9_trainer_support_audit,
    build_text_preservation_v5_plan,
    build_text_preservation_v8_fixed_gate_plan,
    build_text_preservation_v7_balanced_plan,
    build_text_preservation_v6_hard_negative_plan,
    build_text_preservation_v6_prompt_records,
    write_text_preservation_promotion_status,
    write_text_preservation_release_gate_status,
    write_text_preservation_v9_artifact_feedback_dataset,
    write_text_preservation_v9_artifact_gate_loss_config,
    write_text_preservation_v11_artifact_gate_loss_config,
    write_text_preservation_kv_delta_audit,
    write_text_preservation_v12_surface_plan,
    write_text_preservation_render_readability_label_manifest,
    write_text_preservation_surface_curriculum_manifest,
    write_text_preservation_qwen_target_refresh_manifest,
    write_text_preservation_v12_trainer_surface_contract_audit,
    write_text_preservation_v13_recovery_plan,
    write_text_preservation_v13_guard_weighted_manifest,
    write_text_preservation_v14_focus_fixed_gate_manifest,
    write_text_preservation_v17_targeted_teacher_refresh_manifest,
    write_text_preservation_v18_tea_micro_refresh_manifest,
    write_text_preservation_v19_dual_guard_refresh_manifest,
    write_text_preservation_v23_hard_heldout_refresh_manifest,
    write_text_preservation_v24_fixed_gate_protected_heldout_refresh_manifest,
    write_text_preservation_v9_artifact_gate_objective,
    write_text_preservation_v9_objective_plan,
    write_text_preservation_v9_trainer_support_audit,
    write_text_preservation_v6_prompt_file,
    write_general_scene_regression_prompt_file,
    write_text_preservation_artifact_feedback_alignment_audit,
    write_text_preservation_heldout_prompt_file,
    write_text_preservation_prompt_file,
)
from gemmanima.training.bridge_training import BridgeTrainingPlan
from gemmanima.training.teacher_targets import audit_target_cache, export_teacher_subset
from gemmanima.training.orchestrator import pipeline_status, write_pipeline_status
from gemmanima.training.evaluation import audit_bridge_checkpoint
from gemmanima.training.rebalance import build_rebalance_subsets
from gemmanima.training.real_render import audit_real_render_dependencies, build_real_render_command
from gemmanima.modules.real_anima_renderer import ExternalAnimaRendererAdapter
from gemmanima.modules.tipo_runtime import DEFAULT_TAG_PROMPT, run_tipo_vision_tag
from gemmanima.rendering.backends import audit_renderer_backend, renderer_backend_profile
from gemmanima.rendering.image_state_engine import image_state_engine_status
from gemmanima.training.real_render import DEFAULT_EMBEDDED_PYTHON
from gemmanima.modules.in_process_anima_renderer import InProcessAnimaRendererAdapter
from gemmanima.core.config import EngineConfig, ModelConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the GemmAnima control backend.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Route one user message through the engine.")
    run_parser.add_argument("message", help="User message to route through the engine.")
    run_parser.add_argument("--session-id", default=None)
    run_parser.add_argument("--manifest-root", default="runs/manifests")
    run_parser.add_argument("--image-root", default="runs/images")
    run_parser.add_argument(
        "--renderer",
        choices=["dry-run", "real", "in-process", "external-script"],
        default="dry-run",
    )
    run_parser.add_argument("--steps", type=int, default=None)
    run_parser.add_argument("--size", type=int, default=None)
    run_parser.add_argument("--cfg", type=float, default=None)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--unet-dtype", default="fp8_e4m3fn_fast")
    run_parser.add_argument("--anima-dm", default=None, help="Override the Anima diffusion model safetensors path.")
    run_parser.add_argument("--hiddenstage-bridge", default=None, help="Override the HiddenStage bridge checkpoint path.")
    run_parser.add_argument("--json", action="store_true", help="Print machine-readable response JSON.")

    tag_parser = subparsers.add_parser("tag-image", help="Run the TIPO vision tagger on one image.")
    tag_parser.add_argument("image_path", help="Image file to tag.")
    tag_parser.add_argument("--prompt", default=DEFAULT_TAG_PROMPT)
    tag_parser.add_argument("--json", action="store_true")

    health_parser = subparsers.add_parser("health", help="Print model registry health.")
    health_parser.add_argument("--json", action="store_true")

    download_plan_parser = subparsers.add_parser("model-download-plan", help="Print first-run model download sources.")
    download_plan_parser.add_argument("--json", action="store_true")

    ensure_assets_parser = subparsers.add_parser("ensure-model-assets", help="Download missing model assets from their configured sources.")
    ensure_assets_parser.add_argument("--name", action="append", default=None, help="Only ensure the named model asset.")
    ensure_assets_parser.add_argument("--overwrite", action="store_true")
    ensure_assets_parser.add_argument("--json", action="store_true")

    latest_parser = subparsers.add_parser("latest-manifest", help="Print the latest manifest path.")
    latest_parser.add_argument("--manifest-root", default="runs/manifests")

    readiness_parser = subparsers.add_parser("training-readiness", help="Audit planner and bridge training readiness.")
    readiness_parser.add_argument("--planner-out", default=None)
    readiness_parser.add_argument("--train-manifest", default=None)
    readiness_parser.add_argument("--eval-manifest", default=None)
    readiness_parser.add_argument("--manifest-limit", type=int, default=0)
    readiness_parser.add_argument("--check-image-embeds", action="store_true")
    readiness_parser.add_argument("--output", default=None)
    readiness_parser.add_argument("--json", action="store_true")

    teacher_parser = subparsers.add_parser(
        "prepare-teacher-targets",
        help="Export a teacher prompt subset for Anima/Qwen crossattn target caching.",
    )
    teacher_parser.add_argument(
        "--manifest",
        default=r"E:\anima_gemma_swap\dataset_manifests\hiddenstage_multimodal_planner_anima_v2.jsonl",
    )
    teacher_parser.add_argument(
        "--output-subset",
        default=r"runs\teacher_targets\hiddenstage_multimodal_planner_anima_v2_teacher_subset.jsonl",
    )
    teacher_parser.add_argument("--limit", type=int, default=0)
    teacher_parser.add_argument(
        "--target-dir",
        default=r"E:\anima_gemma_swap\cache_hiddenstage_planner_v2\targets",
    )
    teacher_parser.add_argument("--json", action="store_true")

    gemma_parser = subparsers.add_parser("prepare-gemma-cache", help="Print Gemma hidden cache commands and pairing state.")
    gemma_parser.add_argument("--json", action="store_true")

    bridge_parser = subparsers.add_parser("prepare-bridge-training", help="Print HiddenStage bridge training command.")
    bridge_parser.add_argument("--json", action="store_true")

    pipe_parser = subparsers.add_parser("pipeline-status", help="Inspect target/Gemma/bridge pipeline state.")
    pipe_parser.add_argument("--output", default=None)
    pipe_parser.add_argument("--json", action="store_true")

    eval_parser = subparsers.add_parser("bridge-eval-status", help="Inspect HiddenStage bridge checkpoint status.")
    eval_parser.add_argument("--checkpoint", default=None)
    eval_parser.add_argument("--json", action="store_true")

    smoke_parser = subparsers.add_parser("bridge-smoke-command", help="Print the heavy bridge forward smoke command.")
    smoke_parser.add_argument("--json", action="store_true")

    gemma_smoke_parser = subparsers.add_parser("gemma-hidden-smoke-command", help="Print the Gemma hidden provider smoke command.")
    gemma_smoke_parser.add_argument("--json", action="store_true")

    t5_smoke_parser = subparsers.add_parser("t5-tokenizer-smoke-command", help="Print the T5 tokenizer provider smoke command.")
    t5_smoke_parser.add_argument("--json", action="store_true")

    in_process_smoke_parser = subparsers.add_parser(
        "in-process-render-smoke-command",
        help="Print a short in-process renderer smoke command.",
    )
    in_process_smoke_parser.add_argument("--json", action="store_true")

    external_render_parser = subparsers.add_parser(
        "external-render-command",
        help="Print the legacy external-script Anima render command.",
    )
    for render_parser in (external_render_parser,):
        render_parser.add_argument("--request", default=None)
        render_parser.add_argument("--out", default=None)
        render_parser.add_argument("--seed", type=int, default=19375672098)
        render_parser.add_argument("--size", type=int, default=512)
        render_parser.add_argument("--steps", type=int, default=12)
        render_parser.add_argument("--cfg", type=float, default=4.5)
        render_parser.add_argument("--unet-dtype", default="fp8_e4m3fn_fast")
        render_parser.add_argument("--hiddenstage-bridge", default=None)
        render_parser.add_argument("--json", action="store_true")

    real_render_parser = subparsers.add_parser(
        "real-render-command",
        help="Compatibility alias for external-render-command.",
    )
    real_render_parser.add_argument("--request", default=None)
    real_render_parser.add_argument("--out", default=None)
    real_render_parser.add_argument("--seed", type=int, default=19375672098)
    real_render_parser.add_argument("--size", type=int, default=512)
    real_render_parser.add_argument("--steps", type=int, default=12)
    real_render_parser.add_argument("--cfg", type=float, default=4.5)
    real_render_parser.add_argument("--unet-dtype", default="fp8_e4m3fn_fast")
    real_render_parser.add_argument("--hiddenstage-bridge", default=None)
    real_render_parser.add_argument("--json", action="store_true")

    real_health_parser = subparsers.add_parser("real-render-health", help="Audit real Anima renderer dependencies.")
    real_health_parser.add_argument("--json", action="store_true")

    renderer_backends_parser = subparsers.add_parser("renderer-backends", help="Audit renderer backend readiness.")
    renderer_backends_parser.add_argument("--json", action="store_true")

    gui_parser = subparsers.add_parser("gui-command", help="Print the local GUI server command.")
    gui_parser.add_argument("--host", default="127.0.0.1")
    gui_parser.add_argument("--port", type=int, default=8765)
    gui_parser.add_argument("--base-dir", default="runs")
    gui_parser.add_argument("--json", action="store_true")

    rebalance_parser = subparsers.add_parser("rebalance-targets", help="Create rebalanced target subsets from unfinished 4070 work.")
    rebalance_parser.add_argument("--completed-4070-shards", type=int, required=True)
    rebalance_parser.add_argument("--json", action="store_true")

    cache_manifest_parser = subparsers.add_parser("write-cache-manifest", help="Write a validated cache build manifest.")
    cache_manifest_parser.add_argument("--stage", default="poc1_1k_smoke")
    cache_manifest_parser.add_argument(
        "--cache-kind",
        choices=["gemma_text_state", "gemma_vision_projection", "anima_te_conditioning"],
        required=True,
    )
    cache_manifest_parser.add_argument("--sample-count", type=int, required=True)
    cache_manifest_parser.add_argument("--source-manifest", required=True)
    cache_manifest_parser.add_argument("--output-dir", required=True)
    cache_manifest_parser.add_argument("--success-count", type=int, default=0)
    cache_manifest_parser.add_argument("--failure-count", type=int, default=0)
    cache_manifest_parser.add_argument("--shape", default="")
    cache_manifest_parser.add_argument("--dtype", default="")
    cache_manifest_parser.add_argument("--device", default="")
    cache_manifest_parser.add_argument("--manifest-out", required=True)
    cache_manifest_parser.add_argument("--json", action="store_true")

    poc1_parser = subparsers.add_parser("poc1-cache-plan", help="Print the PoC1 1k cache build plan.")
    poc1_parser.add_argument(
        "--manifest",
        default=r"E:\anima_gemma_swap\dataset_manifests\hiddenstage_multimodal_planner_anima_v2.jsonl",
    )
    poc1_parser.add_argument("--subset", default=str(DEFAULT_POC1_SUBSET))
    poc1_parser.add_argument("--target-dir", default=str(DEFAULT_POC1_TARGET_DIR))
    poc1_parser.add_argument("--gemma-dir", default=str(DEFAULT_POC1_GEMMA_DIR))
    poc1_parser.add_argument("--limit", type=int, default=1000)
    poc1_parser.add_argument("--gpu-profile", choices=["all", "4070-only"], default="all")
    poc1_parser.add_argument("--json", action="store_true")

    poc1_bridge_parser = subparsers.add_parser("poc1-bridge-plan", help="Print the PoC1 bridge smoke train plan.")
    poc1_bridge_parser.add_argument("--target-dir", default=str(DEFAULT_POC1_TARGET_DIR))
    poc1_bridge_parser.add_argument("--gemma-dir", default=str(DEFAULT_POC1_GEMMA_DIR))
    poc1_bridge_parser.add_argument("--output", default=str(DEFAULT_POC1_BRIDGE_OUT))
    poc1_bridge_parser.add_argument("--limit-shards", type=int, default=1, help="Use 0 to train across all paired shards.")
    poc1_bridge_parser.add_argument("--json", action="store_true")

    compare_parser = subparsers.add_parser("write-compare-report", help="Write a teacher/student generation compare report.")
    compare_parser.add_argument("--prompt", required=True)
    compare_parser.add_argument("--seed", type=int, required=True)
    compare_parser.add_argument("--teacher-image", required=True)
    compare_parser.add_argument("--student-image", required=True)
    compare_parser.add_argument("--student-checkpoint", required=True)
    compare_parser.add_argument("--conditioning-mse", type=float, default=None)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--json", action="store_true")

    poc1_status_parser = subparsers.add_parser("poc1-status", help="Summarize PoC1 runtime and comparison reports.")
    poc1_status_parser.add_argument("--runtime-report", default=str(DEFAULT_RUNTIME_REPORT))
    poc1_status_parser.add_argument("--compare-report", default=str(DEFAULT_COMPARE_REPORT))
    poc1_status_parser.add_argument("--json", action="store_true")

    poc1_runtime_status_parser = subparsers.add_parser(
        "poc1-runtime-status",
        help="Summarize PoC1 10k cache and bridge checkpoint runtime state.",
    )
    poc1_runtime_status_parser.add_argument("--target-dir", default=str(DEFAULT_POC1_10K_TARGET_DIR))
    poc1_runtime_status_parser.add_argument("--gemma-dir", default=str(DEFAULT_POC1_10K_GEMMA_DIR))
    poc1_runtime_status_parser.add_argument("--bridge-checkpoint", default=str(DEFAULT_POC1_10K_BRIDGE_CHECKPOINT))
    poc1_runtime_status_parser.add_argument("--json", action="store_true")

    candidate_workflow_parser = subparsers.add_parser(
        "candidate-workflow-status",
        help="Summarize a candidate checkpoint across smoke, fixed6, and general-quality artifacts.",
    )
    candidate_workflow_parser.add_argument("--candidate-name", required=True)
    candidate_workflow_parser.add_argument("--checkpoint", required=True)
    candidate_workflow_parser.add_argument("--fixed6-summary", default=None)
    candidate_workflow_parser.add_argument("--general-quality-report", default=None)
    candidate_workflow_parser.add_argument("--smoke-report", default=None)
    candidate_workflow_parser.add_argument("--protected-baseline", default="v5")
    candidate_workflow_parser.add_argument(
        "--baseline-checkpoint",
        default=r"runs\cache\text_preservation_blended_v5\bridge\text_preservation_blended_v5_bridge.pt",
    )
    candidate_workflow_parser.add_argument("--output", default=None)
    candidate_workflow_parser.add_argument("--json", action="store_true")

    candidate_objective_parser = subparsers.add_parser(
        "candidate-objective-manifest",
        help="Convert candidate fixed6 regressions into the next protected training objective manifest.",
    )
    candidate_objective_parser.add_argument("--candidate-name", required=True)
    candidate_objective_parser.add_argument("--donor-checkpoint", required=True)
    candidate_objective_parser.add_argument("--fixed6-summary", required=True)
    candidate_objective_parser.add_argument("--protected-baseline", default="v5")
    candidate_objective_parser.add_argument(
        "--baseline-checkpoint",
        default=r"runs\cache\text_preservation_blended_v5\bridge\text_preservation_blended_v5_bridge.pt",
    )
    candidate_objective_parser.add_argument("--target-sample-count", type=int, default=10000)
    candidate_objective_parser.add_argument("--output", default=None)
    candidate_objective_parser.add_argument("--json", action="store_true")

    candidate_promotion_parser = subparsers.add_parser(
        "candidate-promotion-bundle",
        help="Build a default-promotion/rollback gate bundle from candidate workflow status.",
    )
    candidate_promotion_parser.add_argument("--workflow-status", required=True)
    candidate_promotion_parser.add_argument("--candidate-name", default=None)
    candidate_promotion_parser.add_argument("--protected-baseline", default="v5")
    candidate_promotion_parser.add_argument(
        "--baseline-checkpoint",
        default=r"runs\cache\text_preservation_blended_v5\bridge\text_preservation_blended_v5_bridge.pt",
    )
    candidate_promotion_parser.add_argument("--output", default=None)
    candidate_promotion_parser.add_argument("--json", action="store_true")

    image_state_subset_parser = subparsers.add_parser(
        "image-state-conditioning-subset",
        help="Write an image-state conditioning subset from a multimodal manifest.",
    )
    image_state_subset_parser.add_argument("--source-manifest", required=True)
    image_state_subset_parser.add_argument("--output", required=True)
    image_state_subset_parser.add_argument("--limit", type=int, default=10000)
    image_state_subset_parser.add_argument("--start", type=int, default=0)
    image_state_subset_parser.add_argument("--allow-missing-image-embed", action="store_true")
    image_state_subset_parser.add_argument("--json", action="store_true")

    image_state_plan_parser = subparsers.add_parser(
        "image-state-conditioning-plan",
        help="Print or write the 4070-only image-state to Anima conditioning training plan.",
    )
    image_state_plan_parser.add_argument(
        "--source-manifest",
        default=r"E:\anima_gemma_swap\dataset_manifests\hiddenstage_multimodal_planner_anima_v2.jsonl",
    )
    image_state_plan_parser.add_argument("--subset", default=None)
    image_state_plan_parser.add_argument("--output-root", default=r"runs\cache\image_state_conditioning_v1")
    image_state_plan_parser.add_argument(
        "--text-translator",
        default=r"E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt",
    )
    image_state_plan_parser.add_argument("--sample-count", type=int, default=10000)
    image_state_plan_parser.add_argument("--stage", default="image_state_conditioning_v1")
    image_state_plan_parser.add_argument("--target-shard", type=int, default=1000)
    image_state_plan_parser.add_argument("--batch-size", type=int, default=4)
    image_state_plan_parser.add_argument("--epochs", type=int, default=1)
    image_state_plan_parser.add_argument("--lr", type=float, default=2e-4)
    image_state_plan_parser.add_argument("--output", default=None)
    image_state_plan_parser.add_argument("--json", action="store_true")

    image_state_engine_parser = subparsers.add_parser(
        "image-state-engine-status",
        help="Audit the image-state runtime engine checkpoint and fusion-mode readiness.",
    )
    image_state_engine_parser.add_argument(
        "--checkpoint",
        default=r"runs\cache\image_state_conditioning_v2_full\bridge\image_state_conditioning_v2_full_image_translator.pt",
    )
    image_state_engine_parser.add_argument("--subset", default=None)
    image_state_engine_parser.add_argument("--train-report", default=None)
    image_state_engine_parser.add_argument("--json", action="store_true")

    image_state_guard_parser = subparsers.add_parser(
        "image-state-fusion-guard",
        help="Write guard diagnostics and replay rows from a conditioning_fusion sweep report.",
    )
    image_state_guard_parser.add_argument("--sweep-report", required=True)
    image_state_guard_parser.add_argument("--subset", required=True)
    image_state_guard_parser.add_argument("--failed-idx", nargs="+", type=int, required=True)
    image_state_guard_parser.add_argument("--output", required=True)
    image_state_guard_parser.add_argument("--replay-output", required=True)
    image_state_guard_parser.add_argument("--json", action="store_true")

    image_state_preflight_parser = subparsers.add_parser(
        "image-state-fusion-preflight",
        help="Split conditioning_fusion failures into text-path and image-state replay candidates.",
    )
    image_state_preflight_parser.add_argument("--fusion-report", required=True)
    image_state_preflight_parser.add_argument("--text-only-report", required=True)
    image_state_preflight_parser.add_argument("--subset", required=True)
    image_state_preflight_parser.add_argument("--fusion-failed-idx", nargs="+", type=int, required=True)
    image_state_preflight_parser.add_argument("--text-only-failed-idx", nargs="*", type=int, default=[])
    image_state_preflight_parser.add_argument("--output", required=True)
    image_state_preflight_parser.add_argument("--image-replay-output", required=True)
    image_state_preflight_parser.add_argument("--json", action="store_true")

    image_state_objective_parser = subparsers.add_parser(
        "image-state-replay-objective",
        help="Write the next guarded image-state translator objective with conditioning_fusion replay oversampling.",
    )
    image_state_objective_parser.add_argument("--base-subset", required=True)
    image_state_objective_parser.add_argument("--guard-replay", required=True)
    image_state_objective_parser.add_argument("--current-checkpoint", default=None)
    image_state_objective_parser.add_argument("--target-dir", default=None)
    image_state_objective_parser.add_argument("--output-root", default=None)
    image_state_objective_parser.add_argument("--stage", default="image_state_conditioning_v3_guarded")
    image_state_objective_parser.add_argument("--replay-weight", type=int, default=12)
    image_state_objective_parser.add_argument("--epochs", type=int, default=2)
    image_state_objective_parser.add_argument("--batch-size", type=int, default=32)
    image_state_objective_parser.add_argument("--lr", type=float, default=1e-4)
    image_state_objective_parser.add_argument("--image-cache-gb", type=float, default=56.0)
    image_state_objective_parser.add_argument("--output", required=True)
    image_state_objective_parser.add_argument("--json", action="store_true")

    text_eval_parser = subparsers.add_parser("text-rendering-eval-pack", help="Print text rendering preservation eval prompts.")
    text_eval_parser.add_argument("--json", action="store_true")

    text_eval_status_parser = subparsers.add_parser(
        "text-rendering-eval-status",
        help="Summarize text rendering eval artifacts without inventing metrics.",
    )
    text_eval_status_parser.add_argument("--json", action="store_true")

    text_eval_plan_parser = subparsers.add_parser(
        "text-rendering-eval-plan",
        help="Print a dry-run text rendering eval execution plan.",
    )
    text_eval_plan_parser.add_argument("--json", action="store_true")

    text_eval_run_plan_parser = subparsers.add_parser(
        "text-rendering-eval-run-plan",
        help="Print executable 4070-only text rendering eval commands.",
    )
    text_eval_run_plan_parser.add_argument("--max-cases", type=int, default=None)
    text_eval_run_plan_parser.add_argument("--size", type=int, default=512)
    text_eval_run_plan_parser.add_argument("--steps", type=int, default=12)
    text_eval_run_plan_parser.add_argument("--cfg", type=float, default=4.5)
    text_eval_run_plan_parser.add_argument("--unet-dtype", default="fp8_e4m3fn_fast")
    text_eval_run_plan_parser.add_argument("--teacher-checkpoint", default=None)
    text_eval_run_plan_parser.add_argument("--student-checkpoint", default=None)
    text_eval_run_plan_parser.add_argument("--json", action="store_true")

    qwen_prompt_parser = subparsers.add_parser(
        "text-rendering-qwen-baseline-prompts",
        help="Write the Qwen baseline text-rendering prompt jsonl.",
    )
    qwen_prompt_parser.add_argument("--output", default=None)
    qwen_prompt_parser.add_argument("--max-cases", type=int, default=None)
    qwen_prompt_parser.add_argument("--json", action="store_true")

    qwen_plan_parser = subparsers.add_parser(
        "text-rendering-qwen-baseline-plan",
        help="Print executable 4070-only Qwen baseline vs Gemma bridge commands.",
    )
    qwen_plan_parser.add_argument("--max-cases", type=int, default=None)
    qwen_plan_parser.add_argument("--size", type=int, default=512)
    qwen_plan_parser.add_argument("--steps", type=int, default=20)
    qwen_plan_parser.add_argument("--cfg", type=float, default=4.5)
    qwen_plan_parser.add_argument("--sampler", default="euler")
    qwen_plan_parser.add_argument("--scheduler", default="normal")
    qwen_plan_parser.add_argument("--unet-dtype", default="default")
    qwen_plan_parser.add_argument("--student-checkpoint", default=None)
    qwen_plan_parser.add_argument("--student-name", default="gemma_poc1_10k")
    qwen_plan_parser.add_argument("--json", action="store_true")

    text_preserve_plan_parser = subparsers.add_parser(
        "text-preservation-bridge-plan",
        help="Print the 4070-only text-preservation micro-overfit bridge plan.",
    )
    text_preserve_plan_parser.add_argument("--json", action="store_true")

    text_preserve_status_parser = subparsers.add_parser(
        "text-preservation-bridge-status",
        help="Summarize text-preservation cache and bridge artifacts without touching the GPU.",
    )
    text_preserve_status_parser.add_argument("--json", action="store_true")

    text_preserve_prompts_parser = subparsers.add_parser(
        "text-preservation-prompts",
        help="Write expanded text-preservation prompt jsonl.",
    )
    text_preserve_prompts_parser.add_argument("--output", default=None)
    text_preserve_prompts_parser.add_argument("--count", type=int, default=48)
    text_preserve_prompts_parser.add_argument("--include-eval-cases", action="store_true")
    text_preserve_prompts_parser.add_argument("--prompt-index-offset", type=int, default=0)
    text_preserve_prompts_parser.add_argument("--src-prefix", default="text_preserve_blend")
    text_preserve_prompts_parser.add_argument("--no-sample-marker", action="store_true")
    text_preserve_prompts_parser.add_argument("--json", action="store_true")

    text_preserve_blended_parser = subparsers.add_parser(
        "text-preservation-blended-plan",
        help="Print the 4070-only blended text-preservation bridge plan.",
    )
    text_preserve_blended_parser.add_argument("--root", default=None)
    text_preserve_blended_parser.add_argument("--prompt-file", default=None)
    text_preserve_blended_parser.add_argument("--output", default=None)
    text_preserve_blended_parser.add_argument("--resume-kv", default=None)
    text_preserve_blended_parser.add_argument("--count", type=int, default=48)
    text_preserve_blended_parser.add_argument("--prompt-index-offset", type=int, default=0)
    text_preserve_blended_parser.add_argument("--src-prefix", default="text_preserve_blend")
    text_preserve_blended_parser.add_argument("--no-sample-marker", action="store_true")
    text_preserve_blended_parser.add_argument("--text-repeat", type=int, default=32)
    text_preserve_blended_parser.add_argument("--general-shards", type=int, default=1)
    text_preserve_blended_parser.add_argument("--epochs", type=int, default=2)
    text_preserve_blended_parser.add_argument("--include-eval-cases", action="store_true", default=True)
    text_preserve_blended_parser.add_argument("--exclude-eval-cases", action="store_true")
    text_preserve_blended_parser.add_argument("--json", action="store_true")

    text_preserve_blended_status_parser = subparsers.add_parser(
        "text-preservation-blended-status",
        help="Summarize blended text-preservation cache and bridge artifacts.",
    )
    text_preserve_blended_status_parser.add_argument("--target-dir", default=None)
    text_preserve_blended_status_parser.add_argument("--gemma-dir", default=None)
    text_preserve_blended_status_parser.add_argument("--bridge-checkpoint", default=None)
    text_preserve_blended_status_parser.add_argument("--json", action="store_true")

    text_preserve_v5_parser = subparsers.add_parser(
        "text-preservation-v5-plan",
        help="Print the 4070-only v5 no-sample-marker blended text-preservation bridge plan.",
    )
    text_preserve_v5_parser.add_argument("--count", type=int, default=1024)
    text_preserve_v5_parser.add_argument("--text-repeat", type=int, default=6)
    text_preserve_v5_parser.add_argument("--general-shards", type=int, default=4)
    text_preserve_v5_parser.add_argument("--epochs", type=int, default=2)
    text_preserve_v5_parser.add_argument("--lr", type=float, default=3e-5)
    text_preserve_v5_parser.add_argument("--json", action="store_true")

    text_preserve_v6_prompts_parser = subparsers.add_parser(
        "text-preservation-v6-prompts",
        help="Write hard-negative no-sample-marker v6 text-preservation prompt jsonl.",
    )
    text_preserve_v6_prompts_parser.add_argument("--output", default=None)
    text_preserve_v6_prompts_parser.add_argument("--count", type=int, default=320)
    text_preserve_v6_prompts_parser.add_argument("--exclude-eval-cases", action="store_true")
    text_preserve_v6_prompts_parser.add_argument("--json", action="store_true")

    text_preserve_v6_parser = subparsers.add_parser(
        "text-preservation-v6-plan",
        help="Print the 4070-only v6 hard-negative blended text-preservation bridge plan.",
    )
    text_preserve_v6_parser.add_argument("--count", type=int, default=320)
    text_preserve_v6_parser.add_argument("--text-repeat", type=int, default=10)
    text_preserve_v6_parser.add_argument("--general-shards", type=int, default=4)
    text_preserve_v6_parser.add_argument("--epochs", type=int, default=2)
    text_preserve_v6_parser.add_argument("--lr", type=float, default=2e-5)
    text_preserve_v6_parser.add_argument("--json", action="store_true")

    text_preserve_v7_parser = subparsers.add_parser(
        "text-preservation-v7-plan",
        help="Print the 4070-only v7 balanced replay text-preservation bridge plan.",
    )
    text_preserve_v7_parser.add_argument("--v5-text-repeats", type=int, default=4)
    text_preserve_v7_parser.add_argument("--hard-negative-repeats", type=int, default=1)
    text_preserve_v7_parser.add_argument("--general-shards", type=int, default=10)
    text_preserve_v7_parser.add_argument("--epochs", type=int, default=2)
    text_preserve_v7_parser.add_argument("--lr", type=float, default=1e-5)
    text_preserve_v7_parser.add_argument("--json", action="store_true")

    text_preserve_v8_parser = subparsers.add_parser(
        "text-preservation-v8-plan",
        help="Print the 4070-only v8 fixed-gate preserving text-preservation bridge plan.",
    )
    text_preserve_v8_parser.add_argument("--fixed-gate-repeats", type=int, default=8)
    text_preserve_v8_parser.add_argument("--v5-text-repeats", type=int, default=2)
    text_preserve_v8_parser.add_argument("--hard-negative-repeats", type=int, default=1)
    text_preserve_v8_parser.add_argument("--general-shards", type=int, default=4)
    text_preserve_v8_parser.add_argument("--epochs", type=int, default=1)
    text_preserve_v8_parser.add_argument("--lr", type=float, default=5e-6)
    text_preserve_v8_parser.add_argument("--json", action="store_true")

    promotion_status_parser = subparsers.add_parser(
        "text-preservation-promotion-status",
        help="Summarize text-preservation promotion gates from existing reports without GPU work.",
    )
    promotion_status_parser.add_argument("--fixed-report-root", default=None)
    promotion_status_parser.add_argument("--baseline", default="v5")
    promotion_status_parser.add_argument("--output", default=None)
    promotion_status_parser.add_argument("--compact-output", default=None)
    promotion_status_parser.add_argument("--json", action="store_true")

    release_gate_parser = subparsers.add_parser(
        "text-preservation-release-gate",
        help="Evaluate the non-GPU v5 protected-baseline release gate and v9 training gate.",
    )
    release_gate_parser.add_argument("--fixed-report-root", default=None)
    release_gate_parser.add_argument("--baseline", default="v5")
    release_gate_parser.add_argument("--output", default=None)
    release_gate_parser.add_argument("--json", action="store_true")

    v9_objective_parser = subparsers.add_parser(
        "text-preservation-v9-objective-plan",
        help="Write the non-GPU v9 objective-redesign contract before any new training run.",
    )
    v9_objective_parser.add_argument("--baseline", default="v5")
    v9_objective_parser.add_argument("--output", default=None)
    v9_objective_parser.add_argument("--json", action="store_true")

    v9_artifact_objective_parser = subparsers.add_parser(
        "text-preservation-v9-artifact-gate-objective",
        help="Write the non-GPU artifact-gate-first v9 objective contract.",
    )
    v9_artifact_objective_parser.add_argument("--baseline", default="v5")
    v9_artifact_objective_parser.add_argument("--output", default=None)
    v9_artifact_objective_parser.add_argument("--json", action="store_true")

    v9_trainer_audit_parser = subparsers.add_parser(
        "text-preservation-v9-trainer-support-audit",
        help="Audit whether the bridge trainer supports the artifact-gate-first objective.",
    )
    v9_trainer_audit_parser.add_argument("--train-script", default=None)
    v9_trainer_audit_parser.add_argument("--output", default=None)
    v9_trainer_audit_parser.add_argument("--json", action="store_true")

    v9_artifact_feedback_parser = subparsers.add_parser(
        "text-preservation-v9-artifact-feedback",
        help="Write the fixed-gate artifact feedback JSONL for v9 candidate training.",
    )
    v9_artifact_feedback_parser.add_argument("--fixed-report-root", default=None)
    v9_artifact_feedback_parser.add_argument("--student-name", default="gemma_text_preservation_blended_v5")
    v9_artifact_feedback_parser.add_argument("--output", default=None)
    v9_artifact_feedback_parser.add_argument("--json", action="store_true")

    v9_loss_config_parser = subparsers.add_parser(
        "text-preservation-v9-artifact-gate-loss-config",
        help="Write the v9 artifact gate loss config JSON.",
    )
    v9_loss_config_parser.add_argument("--output", default=None)
    v9_loss_config_parser.add_argument("--json", action="store_true")

    v9_candidate_parser = subparsers.add_parser(
        "text-preservation-v9-candidate-plan",
        help="Print the 4070-only v9 artifact-gate candidate bridge plan.",
    )
    v9_candidate_parser.add_argument("--root", default=None)
    v9_candidate_parser.add_argument("--output", default=None)
    v9_candidate_parser.add_argument("--artifact-feedback", default=None)
    v9_candidate_parser.add_argument("--artifact-gate-loss-config", default=None)
    v9_candidate_parser.add_argument("--json", action="store_true")

    v10_candidate_parser = subparsers.add_parser(
        "text-preservation-v10-candidate-plan",
        help="Print the 4070-only v10 protected-anchor artifact-gate bridge plan.",
    )
    v10_candidate_parser.add_argument("--root", default=None)
    v10_candidate_parser.add_argument("--output", default=None)
    v10_candidate_parser.add_argument("--artifact-feedback", default=None)
    v10_candidate_parser.add_argument("--artifact-gate-loss-config", default=None)
    v10_candidate_parser.add_argument("--anchor-checkpoint", default=None)
    v10_candidate_parser.add_argument("--anchor-lambda", type=float, default=0.1)
    v10_candidate_parser.add_argument("--json", action="store_true")

    v11_loss_config_parser = subparsers.add_parser(
        "text-preservation-v11-artifact-gate-loss-config",
        help="Write the v11 fixed-gate-only artifact gate loss config JSON.",
    )
    v11_loss_config_parser.add_argument("--output", default=None)
    v11_loss_config_parser.add_argument("--json", action="store_true")

    v11_candidate_parser = subparsers.add_parser(
        "text-preservation-v11-candidate-plan",
        help="Print the 4070-only v11 source-filtered artifact-gate bridge plan.",
    )
    v11_candidate_parser.add_argument("--root", default=None)
    v11_candidate_parser.add_argument("--output", default=None)
    v11_candidate_parser.add_argument("--artifact-feedback", default=None)
    v11_candidate_parser.add_argument("--artifact-gate-loss-config", default=None)
    v11_candidate_parser.add_argument("--anchor-checkpoint", default=None)
    v11_candidate_parser.add_argument("--anchor-lambda", type=float, default=0.1)
    v11_candidate_parser.add_argument("--json", action="store_true")

    feedback_alignment_parser = subparsers.add_parser(
        "text-preservation-artifact-feedback-alignment-audit",
        help="Audit whether artifact feedback ids align to the intended blended shard sources.",
    )
    feedback_alignment_parser.add_argument("--blend-target-dir", default=None)
    feedback_alignment_parser.add_argument("--artifact-feedback", default=None)
    feedback_alignment_parser.add_argument("--output", default=None)
    feedback_alignment_parser.add_argument("--json", action="store_true")

    kv_delta_parser = subparsers.add_parser(
        "text-preservation-kv-delta-audit",
        help="Measure v9-v11 KV checkpoint drift against the protected v5 baseline without GPU work.",
    )
    kv_delta_parser.add_argument("--baseline-checkpoint", default=None)
    kv_delta_parser.add_argument(
        "--candidate-checkpoint",
        action="append",
        default=None,
        help="Candidate checkpoint as version=path. May be repeated.",
    )
    kv_delta_parser.add_argument("--output", default=None)
    kv_delta_parser.add_argument("--json", action="store_true")

    v12_surface_parser = subparsers.add_parser(
        "text-preservation-v12-surface-plan",
        help="Write the non-GPU v12 training-surface redesign plan after v9-v11 rejection evidence.",
    )
    v12_surface_parser.add_argument("--baseline", default="v5")
    v12_surface_parser.add_argument("--output", default=None)
    v12_surface_parser.add_argument("--json", action="store_true")

    readability_manifest_parser = subparsers.add_parser(
        "text-preservation-render-readability-label-manifest",
        help="Build the non-GPU v12 render/readability label manifest from existing review artifacts.",
    )
    readability_manifest_parser.add_argument("--fixed-review", default=None)
    readability_manifest_parser.add_argument("--heldout-review", default=None)
    readability_manifest_parser.add_argument("--heldout-report-root", default=None)
    readability_manifest_parser.add_argument("--general-review", default=None)
    readability_manifest_parser.add_argument("--general-metrics", default=None)
    readability_manifest_parser.add_argument("--output", default=None)
    readability_manifest_parser.add_argument("--json", action="store_true")

    surface_curriculum_parser = subparsers.add_parser(
        "text-preservation-surface-curriculum-manifest",
        help="Build the non-GPU v12 surface curriculum manifest from render/readability labels.",
    )
    surface_curriculum_parser.add_argument("--label-manifest", default=None)
    surface_curriculum_parser.add_argument("--max-readable-guards", type=int, default=12)
    surface_curriculum_parser.add_argument("--output", default=None)
    surface_curriculum_parser.add_argument("--json", action="store_true")

    qwen_refresh_parser = subparsers.add_parser(
        "text-preservation-qwen-target-refresh-manifest",
        help="Build the non-GPU v12 Qwen target refresh manifest and prompt JSONL from the surface curriculum.",
    )
    qwen_refresh_parser.add_argument("--curriculum-manifest", default=None)
    qwen_refresh_parser.add_argument("--prompt-file", default=None)
    qwen_refresh_parser.add_argument("--target-dir", default=None)
    qwen_refresh_parser.add_argument("--gpu-index", type=int, default=0)
    qwen_refresh_parser.add_argument("--output", default=None)
    qwen_refresh_parser.add_argument("--json", action="store_true")

    v12_trainer_contract_parser = subparsers.add_parser(
        "text-preservation-v12-trainer-surface-contract-audit",
        help="Audit whether the external trainer supports the v12 surface curriculum contract without GPU work.",
    )
    v12_trainer_contract_parser.add_argument("--train-script", default=None)
    v12_trainer_contract_parser.add_argument("--output", default=None)
    v12_trainer_contract_parser.add_argument("--json", action="store_true")

    v13_recovery_parser = subparsers.add_parser(
        "text-preservation-v13-recovery-plan",
        help="Plan the non-GPU v13 recovery path after v12 fixed-gate rejection.",
    )
    v13_recovery_parser.add_argument("--promotion-status", default=None)
    v13_recovery_parser.add_argument("--baseline", default="v5")
    v13_recovery_parser.add_argument("--rejected-version", default="v12")
    v13_recovery_parser.add_argument("--gpu-index", type=int, default=0)
    v13_recovery_parser.add_argument("--output", default=None)
    v13_recovery_parser.add_argument("--json", action="store_true")

    v13_guard_parser = subparsers.add_parser(
        "text-preservation-v13-guard-weighted-manifest",
        help="Build the non-GPU guard-weighted v13 tiny-ablation manifest and prompt JSONL.",
    )
    v13_guard_parser.add_argument("--source-manifest", default=None)
    v13_guard_parser.add_argument("--prompt-file", default=None)
    v13_guard_parser.add_argument("--target-dir", default=None)
    v13_guard_parser.add_argument("--max-refresh-records", type=int, default=2)
    v13_guard_parser.add_argument("--max-readable-guards", type=int, default=4)
    v13_guard_parser.add_argument("--gpu-index", type=int, default=0)
    v13_guard_parser.add_argument("--output", default=None)
    v13_guard_parser.add_argument("--json", action="store_true")

    v14_focus_parser = subparsers.add_parser(
        "text-preservation-v14-focus-fixed-gate-manifest",
        help="Build the non-GPU v14 focus-only fixed-gate manifest for remaining v13 regressions.",
    )
    v14_focus_parser.add_argument("--source-manifest", default=None)
    v14_focus_parser.add_argument("--prompt-file", default=None)
    v14_focus_parser.add_argument("--target-dir", default=None)
    v14_focus_parser.add_argument("--focus-case-id", action="append", default=None)
    v14_focus_parser.add_argument("--gpu-index", type=int, default=0)
    v14_focus_parser.add_argument("--output", default=None)
    v14_focus_parser.add_argument("--json", action="store_true")

    v17_refresh_parser = subparsers.add_parser(
        "text-preservation-v17-targeted-teacher-refresh-manifest",
        help="Build the non-GPU v17 targeted teacher-refresh manifest for MEET AT DAWN and TEA regressions.",
    )
    v17_refresh_parser.add_argument("--source-manifest", default=None)
    v17_refresh_parser.add_argument("--prompt-file", default=None)
    v17_refresh_parser.add_argument("--target-dir", default=None)
    v17_refresh_parser.add_argument("--focus-case-id", action="append", default=None)
    v17_refresh_parser.add_argument("--focus-variant-count", type=int, default=4)
    v17_refresh_parser.add_argument("--gpu-index", type=int, default=0)
    v17_refresh_parser.add_argument("--output", default=None)
    v17_refresh_parser.add_argument("--json", action="store_true")

    v18_refresh_parser = subparsers.add_parser(
        "text-preservation-v18-tea-micro-refresh-manifest",
        help="Build the non-GPU v18 TEA-only micro-refresh manifest after the v17 single-case rejection.",
    )
    v18_refresh_parser.add_argument("--source-manifest", default=None)
    v18_refresh_parser.add_argument("--prompt-file", default=None)
    v18_refresh_parser.add_argument("--target-dir", default=None)
    v18_refresh_parser.add_argument("--focus-case-id", action="append", default=None)
    v18_refresh_parser.add_argument("--v17-gain-guard-case-id", action="append", default=None)
    v18_refresh_parser.add_argument("--focus-variant-count", type=int, default=6)
    v18_refresh_parser.add_argument("--gpu-index", type=int, default=0)
    v18_refresh_parser.add_argument("--output", default=None)
    v18_refresh_parser.add_argument("--json", action="store_true")

    v19_refresh_parser = subparsers.add_parser(
        "text-preservation-v19-dual-guard-refresh-manifest",
        help="Build the non-GPU v19 dual MEET/TEA refresh manifest from the protected v5 baseline.",
    )
    v19_refresh_parser.add_argument("--source-manifest", default=None)
    v19_refresh_parser.add_argument("--prompt-file", default=None)
    v19_refresh_parser.add_argument("--target-dir", default=None)
    v19_refresh_parser.add_argument("--focus-case-id", action="append", default=None)
    v19_refresh_parser.add_argument("--stability-guard-case-id", action="append", default=None)
    v19_refresh_parser.add_argument("--focus-variant-count", type=int, default=4)
    v19_refresh_parser.add_argument("--gpu-index", type=int, default=0)
    v19_refresh_parser.add_argument("--output", default=None)
    v19_refresh_parser.add_argument("--json", action="store_true")

    v23_refresh_parser = subparsers.add_parser(
        "text-preservation-v23-hard-heldout-refresh-manifest",
        help="Build the non-GPU v23 hard-heldout refresh manifest after the v22 alpha28 heldout rejection.",
    )
    v23_refresh_parser.add_argument("--source-manifest", default=None)
    v23_refresh_parser.add_argument("--heldout-review", default=None)
    v23_refresh_parser.add_argument("--heldout-prompts", default=None)
    v23_refresh_parser.add_argument("--prompt-file", default=None)
    v23_refresh_parser.add_argument("--target-dir", default=None)
    v23_refresh_parser.add_argument("--max-partial-records", type=int, default=12)
    v23_refresh_parser.add_argument("--gpu-index", type=int, default=0)
    v23_refresh_parser.add_argument("--output", default=None)
    v23_refresh_parser.add_argument("--json", action="store_true")

    v24_refresh_parser = subparsers.add_parser(
        "text-preservation-v24-fixed-gate-protected-heldout-refresh-manifest",
        help="Build the non-GPU v24 heldout refresh manifest with fixed-gate regression protection after v23.",
    )
    v24_refresh_parser.add_argument("--source-manifest", default=None)
    v24_refresh_parser.add_argument("--heldout-review", default=None)
    v24_refresh_parser.add_argument("--heldout-prompts", default=None)
    v24_refresh_parser.add_argument("--prompt-file", default=None)
    v24_refresh_parser.add_argument("--target-dir", default=None)
    v24_refresh_parser.add_argument("--regression-case-id", action="append", default=None)
    v24_refresh_parser.add_argument("--max-partial-records", type=int, default=8)
    v24_refresh_parser.add_argument("--focus-variant-count", type=int, default=4)
    v24_refresh_parser.add_argument("--gpu-index", type=int, default=0)
    v24_refresh_parser.add_argument("--output", default=None)
    v24_refresh_parser.add_argument("--json", action="store_true")

    heldout_prompts_parser = subparsers.add_parser(
        "text-preservation-heldout-prompts",
        help="Write held-out text-preservation prompt jsonl.",
    )
    heldout_prompts_parser.add_argument("--output", default=None)
    heldout_prompts_parser.add_argument("--count", type=int, default=64)
    heldout_prompts_parser.add_argument("--prompt-index-offset", type=int, default=10000)
    heldout_prompts_parser.add_argument("--src-prefix", default="text_preserve_heldout")
    heldout_prompts_parser.add_argument("--no-sample-marker", action="store_true")
    heldout_prompts_parser.add_argument("--json", action="store_true")

    heldout_eval_parser = subparsers.add_parser(
        "text-preservation-heldout-eval-plan",
        help="Print the 4070-only held-out Qwen-vs-v4 text preservation eval plan.",
    )
    heldout_eval_parser.add_argument("--count", type=int, default=64)
    heldout_eval_parser.add_argument("--prompt-file", default=None)
    heldout_eval_parser.add_argument("--out-root", default=None)
    heldout_eval_parser.add_argument("--report-root", default=None)
    heldout_eval_parser.add_argument("--student-checkpoint", default=None)
    heldout_eval_parser.add_argument("--student-name", default="gemma_text_preservation_blended_v4")
    heldout_eval_parser.add_argument("--prompt-index-offset", type=int, default=10000)
    heldout_eval_parser.add_argument("--src-prefix", default="text_preserve_heldout")
    heldout_eval_parser.add_argument("--no-sample-marker", action="store_true")
    heldout_eval_parser.add_argument("--json", action="store_true")

    general_scene_prompts_parser = subparsers.add_parser(
        "text-preservation-general-scene-prompts",
        help="Write general-scene regression prompt jsonl.",
    )
    general_scene_prompts_parser.add_argument("--output", default=None)
    general_scene_prompts_parser.add_argument("--count", type=int, default=15)
    general_scene_prompts_parser.add_argument("--json", action="store_true")

    general_scene_eval_parser = subparsers.add_parser(
        "text-preservation-general-scene-eval-plan",
        help="Print the 4070-only general-scene Qwen-vs-Gemma regression eval plan.",
    )
    general_scene_eval_parser.add_argument("--count", type=int, default=15)
    general_scene_eval_parser.add_argument("--prompt-file", default=None)
    general_scene_eval_parser.add_argument("--out-root", default=None)
    general_scene_eval_parser.add_argument("--report-root", default=None)
    general_scene_eval_parser.add_argument("--student-checkpoint", default=None)
    general_scene_eval_parser.add_argument("--student-name", default="gemma_text_preservation_blended_v5")
    general_scene_eval_parser.add_argument("--json", action="store_true")

    if argv and argv[0] not in {
        "run",
        "tag-image",
        "health",
        "model-download-plan",
        "ensure-model-assets",
        "latest-manifest",
        "training-readiness",
        "prepare-teacher-targets",
        "prepare-gemma-cache",
        "prepare-bridge-training",
        "pipeline-status",
        "bridge-eval-status",
        "bridge-smoke-command",
        "gemma-hidden-smoke-command",
        "t5-tokenizer-smoke-command",
        "in-process-render-smoke-command",
        "external-render-command",
        "real-render-command",
        "real-render-health",
        "renderer-backends",
        "gui-command",
        "rebalance-targets",
        "write-cache-manifest",
        "poc1-cache-plan",
        "poc1-bridge-plan",
        "write-compare-report",
        "poc1-status",
        "poc1-runtime-status",
        "candidate-workflow-status",
        "candidate-objective-manifest",
        "candidate-promotion-bundle",
        "image-state-conditioning-subset",
        "image-state-conditioning-plan",
        "image-state-engine-status",
        "image-state-fusion-guard",
        "image-state-fusion-preflight",
        "image-state-replay-objective",
        "text-rendering-eval-pack",
        "text-rendering-eval-status",
        "text-rendering-eval-plan",
        "text-rendering-eval-run-plan",
        "text-rendering-qwen-baseline-prompts",
        "text-rendering-qwen-baseline-plan",
        "text-preservation-bridge-plan",
        "text-preservation-bridge-status",
        "text-preservation-prompts",
        "text-preservation-blended-plan",
        "text-preservation-blended-status",
        "text-preservation-v5-plan",
        "text-preservation-v6-prompts",
        "text-preservation-v6-plan",
        "text-preservation-v7-plan",
        "text-preservation-v8-plan",
        "text-preservation-promotion-status",
        "text-preservation-release-gate",
        "text-preservation-v9-objective-plan",
        "text-preservation-v9-artifact-gate-objective",
        "text-preservation-v9-trainer-support-audit",
        "text-preservation-v9-artifact-feedback",
        "text-preservation-v9-artifact-gate-loss-config",
        "text-preservation-v9-candidate-plan",
        "text-preservation-v10-candidate-plan",
        "text-preservation-v11-artifact-gate-loss-config",
        "text-preservation-v11-candidate-plan",
        "text-preservation-artifact-feedback-alignment-audit",
        "text-preservation-kv-delta-audit",
        "text-preservation-v12-surface-plan",
        "text-preservation-render-readability-label-manifest",
        "text-preservation-surface-curriculum-manifest",
        "text-preservation-qwen-target-refresh-manifest",
        "text-preservation-v12-trainer-surface-contract-audit",
        "text-preservation-v13-recovery-plan",
        "text-preservation-v13-guard-weighted-manifest",
        "text-preservation-v14-focus-fixed-gate-manifest",
        "text-preservation-v17-targeted-teacher-refresh-manifest",
        "text-preservation-v18-tea-micro-refresh-manifest",
        "text-preservation-v19-dual-guard-refresh-manifest",
        "text-preservation-v23-hard-heldout-refresh-manifest",
        "text-preservation-v24-fixed-gate-protected-heldout-refresh-manifest",
        "text-preservation-heldout-prompts",
        "text-preservation-heldout-eval-plan",
        "text-preservation-general-scene-prompts",
        "text-preservation-general-scene-eval-plan",
        "-h",
        "--help",
    }:
        argv = ["run", *argv]

    args = parser.parse_args(argv)

    if args.command == "health":
        health = ModelRegistry().health()
        if args.json:
            print(json.dumps(health, ensure_ascii=False, indent=2))
        else:
            for name, item in health.items():
                marker = "ok" if item["exists"] else "missing"
                print(f"{marker} {name}: {item['path']}")
        return 0

    if args.command == "model-download-plan":
        payload = ModelRegistry().download_plan()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for asset in payload["assets"]:
                print(f"{asset['name']}: {asset['source']['url']} -> {asset['path']}")
        return 0

    if args.command == "ensure-model-assets":
        names = set(args.name) if args.name else None
        payload = ModelRegistry().ensure_assets(overwrite=args.overwrite, names=names)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for asset in payload["assets"]:
                print(f"{asset['status']} {asset['name']}: {asset['path']}")
        return 0

    if args.command == "tag-image":
        result = run_tipo_vision_tag(image_path=Path(args.image_path), prompt=args.prompt)
        payload = {
            "mode": "tag_image",
            "status": result.get("status", "failed"),
            "tags": result.get("tags", ""),
            "message": result.get("tags", ""),
            "seconds": result.get("seconds"),
            "model": result.get("model"),
            "mmproj": result.get("mmproj"),
            "device": result.get("device"),
            "error": result.get("error", ""),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["tags"] or payload["error"])
        return 0 if payload["status"] == "completed" else 1

    if args.command == "bridge-eval-status":
        payload = audit_bridge_checkpoint(args.checkpoint) if args.checkpoint else audit_bridge_checkpoint()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "bridge-smoke-command":
        command = (
            r"E:\ComfyUI_sage\python_embeded\python.exe "
            r"scripts\smoke_hiddenstage_bridge_forward.py "
            r"--checkpoint E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt"
        )
        payload = {"command": command}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(command)
        return 0

    if args.command == "gemma-hidden-smoke-command":
        command = f"{DEFAULT_EMBEDDED_PYTHON} scripts\\smoke_gemma_hidden_provider.py --json"
        payload = {
            "command": command,
            "gpu": "RTX 4070 Ti SUPER",
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(command)
        return 0

    if args.command == "t5-tokenizer-smoke-command":
        command = f"{DEFAULT_EMBEDDED_PYTHON} scripts\\smoke_t5_tokenizer_provider.py --json"
        payload = {
            "command": command,
            "gpu": "RTX 4070 Ti SUPER",
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(command)
        return 0

    if args.command == "in-process-render-smoke-command":
        command = (
            f"$env:CUDA_VISIBLE_DEVICES='0'; $env:GEMMA_EMBED_ON_GPU='1'; "
            f"{DEFAULT_EMBEDDED_PYTHON} scripts\\smoke_in_process_render.py "
            "--image-root runs\\images --manifest-root runs\\manifests "
            "--steps 8 --size 512 --cfg 4.5 --json"
        )
        payload = {
            "command": command,
            "gpu": "RTX 4070 Ti SUPER",
            "cuda_visible_devices": "0",
            "gemma_embed_on_gpu": "1",
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(command)
        return 0

    if args.command in {"external-render-command", "real-render-command"}:
        command_kwargs = {
            "size": args.size,
            "steps": args.steps,
            "cfg": args.cfg,
            "seed": args.seed,
            "unet_dtype": args.unet_dtype,
        }
        if args.request is not None:
            command_kwargs["request"] = args.request
        if args.out is not None:
            command_kwargs["output"] = args.out
        if args.hiddenstage_bridge is not None:
            command_kwargs["config"] = _config_with_model_overrides(hiddenstage_bridge=args.hiddenstage_bridge)
        payload = build_real_render_command(
            **command_kwargs,
        ).to_json_dict()
        payload["backend"] = "external_script"
        if args.command == "real-render-command":
            payload["deprecated_alias"] = "real-render-command"
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["command"])
        return 0

    if args.command == "real-render-health":
        payload = audit_real_render_dependencies()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "renderer-backends":
        payload = audit_renderer_backend()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "gui-command":
        command = (
            "$env:CUDA_VISIBLE_DEVICES='0'; $env:GEMMA_EMBED_ON_GPU='1'; "
            f"python -m gemmanima.server --host {args.host} --port {args.port} --base-dir {args.base_dir}"
        )
        payload = {
            "command": command,
            "url": f"http://{args.host}:{args.port}",
            "gpu": "RTX 4070 Ti SUPER",
            "cuda_visible_devices": "0",
            "gemma_embed_on_gpu": "1",
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(command)
            print(payload["url"])
        return 0

    if args.command == "rebalance-targets":
        payload = build_rebalance_subsets(completed_4070_shards=args.completed_4070_shards).to_json_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "write-cache-manifest":
        shape = tuple(int(part) for part in args.shape.split(",") if part)
        manifest = CacheBuildManifest(
            stage=args.stage,
            cache_kind=args.cache_kind,
            sample_count=args.sample_count,
            source_manifest=Path(args.source_manifest),
            output_dir=Path(args.output_dir),
            success_count=args.success_count,
            failure_count=args.failure_count,
            shape=shape,
            dtype=args.dtype,
            device=args.device,
        )
        path = write_cache_build_manifest(manifest, args.manifest_out)
        payload = {"manifest_path": str(path), "manifest": manifest.to_json_dict()}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(path)
        return 0

    if args.command == "poc1-cache-plan":
        payload = build_poc1_cache_plan(
            manifest=args.manifest,
            subset=args.subset,
            target_dir=args.target_dir,
            gemma_dir=args.gemma_dir,
            limit=args.limit,
            gpu_profile=args.gpu_profile,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["prepare_subset_command"])
            print(payload["teacher_target_command"])
            print(payload["teacher_target_manifest_command"])
            for plan in payload["gemma_cache_plans"]:
                print(plan["command"])
                print(plan["cache_manifest_command"])
        return 0

    if args.command == "poc1-bridge-plan":
        payload = build_poc1_bridge_plan(
            target_dir=args.target_dir,
            gemma_dir=args.gemma_dir,
            output=args.output,
            limit_shards=None if args.limit_shards == 0 else args.limit_shards,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["train_command"])
            print(payload["eval_command"])
            print(payload["forward_smoke_command"])
        return 0

    if args.command == "write-compare-report":
        report = GenerationCompareReport(
            prompt=args.prompt,
            seed=args.seed,
            teacher_image=Path(args.teacher_image),
            student_image=Path(args.student_image),
            student_checkpoint=Path(args.student_checkpoint),
            conditioning_mse=args.conditioning_mse,
        )
        path = write_compare_report(report, args.output)
        payload = {"report_path": str(path), "report": report.to_json_dict()}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(path)
        return 0

    if args.command == "poc1-status":
        payload = build_poc1_status(runtime_report=args.runtime_report, compare_report=args.compare_report)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "poc1-runtime-status":
        payload = build_poc1_runtime_status(
            target_dir=args.target_dir,
            gemma_dir=args.gemma_dir,
            bridge_checkpoint=args.bridge_checkpoint,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "candidate-workflow-status":
        kwargs = {
            "candidate_name": args.candidate_name,
            "checkpoint": args.checkpoint,
            "fixed6_summary": args.fixed6_summary,
            "general_quality_report": args.general_quality_report,
            "smoke_report": args.smoke_report,
            "protected_baseline": args.protected_baseline,
            "baseline_checkpoint": args.baseline_checkpoint,
        }
        if args.output:
            path = write_candidate_workflow_status(output=args.output, **kwargs)
            payload = {"status_path": str(path), "status": build_candidate_workflow_status(**kwargs)}
        else:
            payload = build_candidate_workflow_status(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "candidate-objective-manifest":
        kwargs = {
            "candidate_name": args.candidate_name,
            "donor_checkpoint": args.donor_checkpoint,
            "fixed6_summary": args.fixed6_summary,
            "protected_baseline": args.protected_baseline,
            "baseline_checkpoint": args.baseline_checkpoint,
            "target_sample_count": args.target_sample_count,
        }
        if args.output:
            path = write_candidate_objective_manifest(output=args.output, **kwargs)
            payload = {"manifest_path": str(path), "manifest": build_candidate_objective_manifest(**kwargs)}
        else:
            payload = build_candidate_objective_manifest(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "candidate-promotion-bundle":
        kwargs = {
            "workflow_status": args.workflow_status,
            "candidate_name": args.candidate_name,
            "protected_baseline": args.protected_baseline,
            "baseline_checkpoint": args.baseline_checkpoint,
        }
        if args.output:
            path = write_candidate_promotion_bundle(output=args.output, **kwargs)
            payload = {"bundle_path": str(path), "bundle": build_candidate_promotion_bundle(**kwargs)}
        else:
            payload = build_candidate_promotion_bundle(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "image-state-conditioning-subset":
        payload = build_image_state_subset(
            source_manifest=args.source_manifest,
            output=args.output,
            limit=args.limit,
            start=args.start,
            require_image_embed=not args.allow_missing_image_embed,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "image-state-conditioning-plan":
        kwargs = {
            "source_manifest": args.source_manifest,
            "subset": args.subset,
            "output_root": args.output_root,
            "text_translator": args.text_translator,
            "sample_count": args.sample_count,
            "stage": args.stage,
            "target_shard": args.target_shard,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
        }
        if args.output:
            path = write_image_state_conditioning_plan(output=args.output, **kwargs)
            payload = {"plan_path": str(path), "plan": build_image_state_conditioning_plan(**kwargs)}
        else:
            payload = build_image_state_conditioning_plan(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "image-state-engine-status":
        payload = image_state_engine_status(
            checkpoint=args.checkpoint,
            subset=args.subset,
            train_report=args.train_report,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "image-state-fusion-guard":
        payload = write_conditioning_fusion_guard_manifest(
            output=args.output,
            replay_output=args.replay_output,
            sweep_report=args.sweep_report,
            subset=args.subset,
            failed_indices=args.failed_idx,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "image-state-fusion-preflight":
        payload = write_conditioning_fusion_preflight_manifest(
            output=args.output,
            image_replay_output=args.image_replay_output,
            fusion_report=args.fusion_report,
            text_only_report=args.text_only_report,
            subset=args.subset,
            fusion_failed_indices=args.fusion_failed_idx,
            text_only_failed_indices=args.text_only_failed_idx,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "image-state-replay-objective":
        kwargs = {
            "base_subset": args.base_subset,
            "guard_replay": args.guard_replay,
            "stage": args.stage,
            "replay_weight": args.replay_weight,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "image_cache_gb": args.image_cache_gb,
        }
        if args.current_checkpoint is not None:
            kwargs["current_checkpoint"] = args.current_checkpoint
        if args.target_dir is not None:
            kwargs["target_dir"] = args.target_dir
        if args.output_root is not None:
            kwargs["output_root"] = args.output_root
        payload = write_image_state_replay_training_objective(output=args.output, **kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-rendering-eval-pack":
        payload = build_text_rendering_eval_pack()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for item in payload["prompts"]:
                print(f"[{item['category']}] {item['target_text']}: {item['prompt']}")
        return 0

    if args.command == "text-rendering-eval-status":
        payload = build_text_rendering_eval_status()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-rendering-eval-plan":
        payload = build_text_rendering_eval_execution_plan()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for case in payload["cases"]:
                print(f"[{case['id']}] seed={case['seed']} -> {case['comparison']['compare_report']}")
        return 0

    if args.command == "text-rendering-eval-run-plan":
        kwargs = {
            "max_cases": args.max_cases,
            "size": args.size,
            "steps": args.steps,
            "cfg": args.cfg,
            "unet_dtype": args.unet_dtype,
        }
        if args.teacher_checkpoint is not None:
            kwargs["teacher_checkpoint"] = args.teacher_checkpoint
        if args.student_checkpoint is not None:
            kwargs["student_checkpoint"] = args.student_checkpoint
        payload = build_text_rendering_eval_run_plan(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for command in payload["setup_commands"]:
                print(command)
            for case in payload["cases"]:
                print(case["teacher"]["command"])
                print(case["student"]["command"])
                print(case["comparison"]["command"])
        return 0

    if args.command == "text-rendering-qwen-baseline-prompts":
        kwargs = {"max_cases": args.max_cases}
        if args.output is not None:
            kwargs["output"] = args.output
        path = write_text_rendering_qwen_prompt_file(**kwargs)
        payload = {"prompt_file": str(path)}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(path)
        return 0

    if args.command == "text-rendering-qwen-baseline-plan":
        kwargs = {
            "max_cases": args.max_cases,
            "size": args.size,
            "steps": args.steps,
            "cfg": args.cfg,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "unet_dtype": args.unet_dtype,
        }
        if args.student_checkpoint is not None:
            kwargs["student_checkpoint"] = args.student_checkpoint
        kwargs["student_name"] = args.student_name
        payload = build_text_rendering_qwen_baseline_plan(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["prompt_write_command"])
            print(payload["qwen_command"])
            print(payload["gemma_command"])
            for case in payload["cases"]:
                for command in case["rename_commands"]:
                    print(command)
                print(case["compare_command"])
        return 0

    if args.command == "text-preservation-bridge-plan":
        payload = build_text_preservation_bridge_plan()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for command in payload["setup_commands"]:
                print(command)
            print(payload["target_cache_command"])
            print(payload["gemma_cache_command"])
            print(payload["train_command"])
            print(payload["status_command"])
            print(payload["post_train_qwen_eval_plan_command"])
        return 0

    if args.command == "text-preservation-bridge-status":
        payload = build_text_preservation_bridge_status()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-prompts":
        kwargs = {
            "count": args.count,
            "include_eval_cases": args.include_eval_cases,
            "prompt_index_offset": args.prompt_index_offset,
            "src_prefix": args.src_prefix,
            "include_sample_marker": not args.no_sample_marker,
        }
        if args.output is not None:
            kwargs["output"] = args.output
        path = write_text_preservation_prompt_file(**kwargs)
        payload = {
            "prompt_file": str(path),
            "count": len(
                build_text_preservation_prompt_records(
                    count=args.count,
                    include_eval_cases=args.include_eval_cases,
                    prompt_index_offset=args.prompt_index_offset,
                    src_prefix=args.src_prefix,
                    include_sample_marker=not args.no_sample_marker,
                )
            ),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(path)
        return 0

    if args.command == "text-preservation-blended-plan":
        kwargs = {
            "text_repeat": args.text_repeat,
            "general_shards": args.general_shards,
            "epochs": args.epochs,
            "sample_count": args.count,
            "include_eval_cases": args.include_eval_cases and not args.exclude_eval_cases,
            "prompt_index_offset": args.prompt_index_offset,
            "src_prefix": args.src_prefix,
            "include_sample_marker": not args.no_sample_marker,
        }
        if args.root is not None:
            kwargs["root"] = args.root
        if args.prompt_file is not None:
            kwargs["prompt_file"] = args.prompt_file
        if args.output is not None:
            kwargs["output"] = args.output
        if args.resume_kv is not None:
            kwargs["resume_kv"] = args.resume_kv
        payload = build_text_preservation_blended_plan(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for command in payload["setup_commands"]:
                print(command)
            print(payload["prompt_write_command"])
            print(payload["text_target_cache_command"])
            print(payload["text_gemma_cache_command"])
            for command in payload["blend_link_commands"]:
                print(command)
            print(payload["train_command"])
            print(payload["status_command"])
            print(payload["post_train_qwen_eval_plan_command"])
        return 0

    if args.command == "text-preservation-v5-plan":
        payload = build_text_preservation_v5_plan(
            sample_count=args.count,
            text_repeat=args.text_repeat,
            general_shards=args.general_shards,
            epochs=args.epochs,
            lr=args.lr,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for command in payload["setup_commands"]:
                print(command)
            print(payload["prompt_write_command"])
            print(payload["text_target_cache_command"])
            print(payload["text_gemma_cache_command"])
            for command in payload["blend_link_commands"]:
                print(command)
            print(payload["train_command"])
            print(payload["post_train_heldout_eval_plan_command"])
        return 0

    if args.command == "text-preservation-v6-prompts":
        kwargs = {
            "count": args.count,
            "include_eval_cases": not args.exclude_eval_cases,
        }
        if args.output is not None:
            kwargs["output"] = args.output
        path = write_text_preservation_v6_prompt_file(**kwargs)
        count = len(build_text_preservation_v6_prompt_records(count=args.count))
        if not args.exclude_eval_cases:
            count += 6
        payload = {"prompt_file": str(path), "count": count}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(path)
        return 0

    if args.command == "text-preservation-v6-plan":
        payload = build_text_preservation_v6_hard_negative_plan(
            sample_count=args.count,
            text_repeat=args.text_repeat,
            general_shards=args.general_shards,
            epochs=args.epochs,
            lr=args.lr,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for command in payload["setup_commands"]:
                print(command)
            print(payload["prompt_write_command"])
            print(payload["text_target_cache_command"])
            print(payload["text_gemma_cache_command"])
            for command in payload["blend_link_commands"]:
                print(command)
            print(payload["train_command"])
            print(payload["post_train_heldout_eval_plan_command"])
        return 0

    if args.command == "text-preservation-v7-plan":
        payload = build_text_preservation_v7_balanced_plan(
            v5_text_repeats=args.v5_text_repeats,
            hard_negative_repeats=args.hard_negative_repeats,
            general_shards=args.general_shards,
            epochs=args.epochs,
            lr=args.lr,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for command in payload["setup_commands"]:
                print(command)
            for command in payload["blend_link_commands"]:
                print(command)
            print(payload["train_command"])
            print(payload["eval_command"])
            print(payload["post_train_qwen_eval_plan_command"])
        return 0

    if args.command == "text-preservation-v8-plan":
        payload = build_text_preservation_v8_fixed_gate_plan(
            fixed_gate_repeats=args.fixed_gate_repeats,
            v5_text_repeats=args.v5_text_repeats,
            hard_negative_repeats=args.hard_negative_repeats,
            general_shards=args.general_shards,
            epochs=args.epochs,
            lr=args.lr,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for command in payload["setup_commands"]:
                print(command)
            for command in payload["blend_link_commands"]:
                print(command)
            print(payload["train_command"])
            print(payload["eval_command"])
            print(payload["post_train_qwen_eval_plan_command"])
        return 0

    if args.command == "text-preservation-promotion-status":
        kwargs = {"baseline": args.baseline}
        if args.fixed_report_root is not None:
            kwargs["fixed_report_root"] = args.fixed_report_root
        if args.output is not None:
            payload = write_text_preservation_promotion_status(args.output, **kwargs)
        else:
            payload = build_text_preservation_promotion_status(**kwargs)
        if args.compact_output is not None:
            compact_path = Path(args.compact_output)
            compact_path.parent.mkdir(parents=True, exist_ok=True)
            compact_payload = build_text_preservation_compact_promotion_status(payload)
            compact_path.write_text(json.dumps(compact_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            payload["compact_output"] = str(compact_path)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-release-gate":
        kwargs = {"baseline": args.baseline}
        if args.fixed_report_root is not None:
            kwargs["fixed_report_root"] = args.fixed_report_root
        if args.output is not None:
            payload = write_text_preservation_release_gate_status(args.output, **kwargs)
        else:
            payload = build_text_preservation_release_gate_status(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v9-objective-plan":
        kwargs = {"baseline": args.baseline}
        if args.output is not None:
            payload = write_text_preservation_v9_objective_plan(args.output, **kwargs)
        else:
            payload = build_text_preservation_v9_objective_plan(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v9-artifact-gate-objective":
        kwargs = {"baseline": args.baseline}
        if args.output is not None:
            payload = write_text_preservation_v9_artifact_gate_objective(args.output, **kwargs)
        else:
            payload = build_text_preservation_v9_artifact_gate_objective(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v9-trainer-support-audit":
        kwargs = {}
        if args.train_script is not None:
            kwargs["train_script"] = args.train_script
        if args.output is not None:
            payload = write_text_preservation_v9_trainer_support_audit(args.output, **kwargs)
        else:
            payload = build_text_preservation_v9_trainer_support_audit(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v9-artifact-feedback":
        kwargs = {"student_name": args.student_name}
        if args.fixed_report_root is not None:
            kwargs["fixed_report_root"] = args.fixed_report_root
        if args.output is not None:
            payload = write_text_preservation_v9_artifact_feedback_dataset(args.output, **kwargs)
        else:
            payload = build_text_preservation_v9_artifact_feedback_dataset(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v9-artifact-gate-loss-config":
        if args.output is not None:
            payload = write_text_preservation_v9_artifact_gate_loss_config(args.output)
        else:
            payload = build_text_preservation_v9_artifact_gate_loss_config()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v9-candidate-plan":
        kwargs = {}
        if args.root is not None:
            kwargs["root"] = args.root
        if args.output is not None:
            kwargs["output"] = args.output
        if args.artifact_feedback is not None:
            kwargs["artifact_feedback"] = args.artifact_feedback
        if args.artifact_gate_loss_config is not None:
            kwargs["artifact_gate_loss_config"] = args.artifact_gate_loss_config
        payload = build_text_preservation_v9_candidate_plan(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload.get("artifact_feedback_write_command", ""))
            print(payload.get("artifact_gate_loss_config_write_command", ""))
            for command in payload.get("setup_commands", []):
                print(command)
            for command in payload.get("blend_link_commands", []):
                print(command)
            print(payload.get("train_command"))
            print(payload.get("post_train_qwen_eval_plan_command"))
        return 0

    if args.command == "text-preservation-v10-candidate-plan":
        kwargs = {"anchor_lambda": args.anchor_lambda}
        if args.root is not None:
            kwargs["root"] = args.root
        if args.output is not None:
            kwargs["output"] = args.output
        if args.artifact_feedback is not None:
            kwargs["artifact_feedback"] = args.artifact_feedback
        if args.artifact_gate_loss_config is not None:
            kwargs["artifact_gate_loss_config"] = args.artifact_gate_loss_config
        if args.anchor_checkpoint is not None:
            kwargs["anchor_checkpoint"] = args.anchor_checkpoint
        payload = build_text_preservation_v10_candidate_plan(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload.get("artifact_feedback_write_command", ""))
            print(payload.get("artifact_gate_loss_config_write_command", ""))
            for command in payload.get("setup_commands", []):
                print(command)
            for command in payload.get("blend_link_commands", []):
                print(command)
            print(payload.get("train_command"))
            print(payload.get("post_train_qwen_eval_plan_command"))
        return 0

    if args.command == "text-preservation-v11-artifact-gate-loss-config":
        if args.output is not None:
            payload = write_text_preservation_v11_artifact_gate_loss_config(args.output)
        else:
            payload = build_text_preservation_v11_artifact_gate_loss_config()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v11-candidate-plan":
        kwargs = {"anchor_lambda": args.anchor_lambda}
        if args.root is not None:
            kwargs["root"] = args.root
        if args.output is not None:
            kwargs["output"] = args.output
        if args.artifact_feedback is not None:
            kwargs["artifact_feedback"] = args.artifact_feedback
        if args.artifact_gate_loss_config is not None:
            kwargs["artifact_gate_loss_config"] = args.artifact_gate_loss_config
        if args.anchor_checkpoint is not None:
            kwargs["anchor_checkpoint"] = args.anchor_checkpoint
        payload = build_text_preservation_v11_candidate_plan(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload.get("artifact_feedback_write_command", ""))
            print(payload.get("artifact_gate_loss_config_write_command", ""))
            for command in payload.get("setup_commands", []):
                print(command)
            for command in payload.get("blend_link_commands", []):
                print(command)
            print(payload.get("train_command"))
            print(payload.get("post_train_qwen_eval_plan_command"))
        return 0

    if args.command == "text-preservation-artifact-feedback-alignment-audit":
        kwargs = {}
        if args.blend_target_dir is not None:
            kwargs["blend_target_dir"] = args.blend_target_dir
        if args.artifact_feedback is not None:
            kwargs["artifact_feedback"] = args.artifact_feedback
        if args.output is not None:
            payload = write_text_preservation_artifact_feedback_alignment_audit(args.output, **kwargs)
        else:
            payload = build_text_preservation_artifact_feedback_alignment_audit(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-kv-delta-audit":
        kwargs = {}
        if args.baseline_checkpoint is not None:
            kwargs["baseline_checkpoint"] = args.baseline_checkpoint
        if args.candidate_checkpoint is not None:
            candidates = {}
            for item in args.candidate_checkpoint:
                if "=" not in item:
                    parser.error("--candidate-checkpoint must use version=path")
                version, checkpoint = item.split("=", 1)
                candidates[version] = checkpoint
            kwargs["candidate_checkpoints"] = candidates
        if args.output is not None:
            payload = write_text_preservation_kv_delta_audit(args.output, **kwargs)
        else:
            payload = build_text_preservation_kv_delta_audit(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v12-surface-plan":
        kwargs = {"baseline": args.baseline}
        if args.output is not None:
            payload = write_text_preservation_v12_surface_plan(args.output, **kwargs)
        else:
            payload = build_text_preservation_v12_surface_plan(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-render-readability-label-manifest":
        kwargs = {}
        if args.fixed_review is not None:
            kwargs["fixed_review"] = args.fixed_review
        if args.heldout_review is not None:
            kwargs["heldout_review"] = args.heldout_review
        if args.heldout_report_root is not None:
            kwargs["heldout_report_root"] = args.heldout_report_root
        if args.general_review is not None:
            kwargs["general_review"] = args.general_review
        if args.general_metrics is not None:
            kwargs["general_metrics"] = args.general_metrics
        if args.output is not None:
            payload = write_text_preservation_render_readability_label_manifest(args.output, **kwargs)
        else:
            payload = build_text_preservation_render_readability_label_manifest(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-surface-curriculum-manifest":
        kwargs = {"max_readable_guards": args.max_readable_guards}
        if args.label_manifest is not None:
            kwargs["label_manifest"] = args.label_manifest
        if args.output is not None:
            payload = write_text_preservation_surface_curriculum_manifest(args.output, **kwargs)
        else:
            payload = build_text_preservation_surface_curriculum_manifest(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-qwen-target-refresh-manifest":
        kwargs = {"gpu_index": args.gpu_index}
        if args.curriculum_manifest is not None:
            kwargs["curriculum_manifest"] = args.curriculum_manifest
        if args.prompt_file is not None:
            kwargs["prompt_file"] = args.prompt_file
        if args.target_dir is not None:
            kwargs["target_dir"] = args.target_dir
        if args.output is not None:
            payload = write_text_preservation_qwen_target_refresh_manifest(args.output, **kwargs)
        else:
            payload = build_text_preservation_qwen_target_refresh_manifest(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v12-trainer-surface-contract-audit":
        kwargs = {}
        if args.train_script is not None:
            kwargs["train_script"] = args.train_script
        if args.output is not None:
            payload = write_text_preservation_v12_trainer_surface_contract_audit(args.output, **kwargs)
        else:
            payload = build_text_preservation_v12_trainer_surface_contract_audit(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v13-recovery-plan":
        kwargs = {
            "baseline": args.baseline,
            "rejected_version": args.rejected_version,
            "gpu_index": args.gpu_index,
        }
        if args.promotion_status is not None:
            kwargs["promotion_status"] = args.promotion_status
        if args.output is not None:
            payload = write_text_preservation_v13_recovery_plan(args.output, **kwargs)
        else:
            payload = build_text_preservation_v13_recovery_plan(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v13-guard-weighted-manifest":
        kwargs = {
            "max_refresh_records": args.max_refresh_records,
            "max_readable_guards": args.max_readable_guards,
            "gpu_index": args.gpu_index,
        }
        if args.source_manifest is not None:
            kwargs["source_manifest"] = args.source_manifest
        if args.prompt_file is not None:
            kwargs["prompt_file"] = args.prompt_file
        if args.target_dir is not None:
            kwargs["target_dir"] = args.target_dir
        if args.output is not None:
            payload = write_text_preservation_v13_guard_weighted_manifest(args.output, **kwargs)
        else:
            payload = build_text_preservation_v13_guard_weighted_manifest(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v14-focus-fixed-gate-manifest":
        kwargs = {"gpu_index": args.gpu_index}
        if args.source_manifest is not None:
            kwargs["source_manifest"] = args.source_manifest
        if args.prompt_file is not None:
            kwargs["prompt_file"] = args.prompt_file
        if args.target_dir is not None:
            kwargs["target_dir"] = args.target_dir
        if args.focus_case_id is not None:
            kwargs["focus_case_ids"] = tuple(args.focus_case_id)
        if args.output is not None:
            payload = write_text_preservation_v14_focus_fixed_gate_manifest(args.output, **kwargs)
        else:
            payload = build_text_preservation_v14_focus_fixed_gate_manifest(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-blended-status":
        plan = build_text_preservation_blended_plan()
        target_dir = args.target_dir if args.target_dir is not None else plan["blend_target_dir"]
        gemma_dir = args.gemma_dir if args.gemma_dir is not None else plan["blend_gemma_dir"]
        bridge_checkpoint = args.bridge_checkpoint if args.bridge_checkpoint is not None else plan["output"]
        payload = build_text_preservation_bridge_status(
            target_dir=target_dir,
            gemma_dir=gemma_dir,
            bridge_checkpoint=bridge_checkpoint,
        )
        payload["stage"] = "text_preservation_blended_candidate"
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v17-targeted-teacher-refresh-manifest":
        kwargs = {
            "focus_variant_count": args.focus_variant_count,
            "gpu_index": args.gpu_index,
        }
        if args.source_manifest is not None:
            kwargs["source_manifest"] = args.source_manifest
        if args.prompt_file is not None:
            kwargs["prompt_file"] = args.prompt_file
        if args.target_dir is not None:
            kwargs["target_dir"] = args.target_dir
        if args.focus_case_id is not None:
            kwargs["focus_case_ids"] = tuple(args.focus_case_id)
        if args.output is not None:
            payload = write_text_preservation_v17_targeted_teacher_refresh_manifest(args.output, **kwargs)
        else:
            payload = build_text_preservation_v17_targeted_teacher_refresh_manifest(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v18-tea-micro-refresh-manifest":
        kwargs = {
            "focus_variant_count": args.focus_variant_count,
            "gpu_index": args.gpu_index,
        }
        if args.source_manifest is not None:
            kwargs["source_manifest"] = args.source_manifest
        if args.prompt_file is not None:
            kwargs["prompt_file"] = args.prompt_file
        if args.target_dir is not None:
            kwargs["target_dir"] = args.target_dir
        if args.focus_case_id is not None:
            kwargs["focus_case_ids"] = tuple(args.focus_case_id)
        if args.v17_gain_guard_case_id is not None:
            kwargs["v17_gain_guard_case_ids"] = tuple(args.v17_gain_guard_case_id)
        if args.output is not None:
            payload = write_text_preservation_v18_tea_micro_refresh_manifest(args.output, **kwargs)
        else:
            payload = build_text_preservation_v18_tea_micro_refresh_manifest(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v19-dual-guard-refresh-manifest":
        kwargs = {
            "focus_variant_count": args.focus_variant_count,
            "gpu_index": args.gpu_index,
        }
        if args.source_manifest is not None:
            kwargs["source_manifest"] = args.source_manifest
        if args.prompt_file is not None:
            kwargs["prompt_file"] = args.prompt_file
        if args.target_dir is not None:
            kwargs["target_dir"] = args.target_dir
        if args.focus_case_id is not None:
            kwargs["focus_case_ids"] = tuple(args.focus_case_id)
        if args.stability_guard_case_id is not None:
            kwargs["stability_guard_case_ids"] = tuple(args.stability_guard_case_id)
        if args.output is not None:
            payload = write_text_preservation_v19_dual_guard_refresh_manifest(args.output, **kwargs)
        else:
            payload = build_text_preservation_v19_dual_guard_refresh_manifest(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v23-hard-heldout-refresh-manifest":
        kwargs = {
            "max_partial_records": args.max_partial_records,
            "gpu_index": args.gpu_index,
        }
        if args.source_manifest is not None:
            kwargs["source_manifest"] = args.source_manifest
        if args.heldout_review is not None:
            kwargs["heldout_review"] = args.heldout_review
        if args.heldout_prompts is not None:
            kwargs["heldout_prompts"] = args.heldout_prompts
        if args.prompt_file is not None:
            kwargs["prompt_file"] = args.prompt_file
        if args.target_dir is not None:
            kwargs["target_dir"] = args.target_dir
        if args.output is not None:
            payload = write_text_preservation_v23_hard_heldout_refresh_manifest(args.output, **kwargs)
        else:
            payload = build_text_preservation_v23_hard_heldout_refresh_manifest(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-v24-fixed-gate-protected-heldout-refresh-manifest":
        kwargs = {
            "max_partial_records": args.max_partial_records,
            "focus_variant_count": args.focus_variant_count,
            "gpu_index": args.gpu_index,
        }
        if args.source_manifest is not None:
            kwargs["source_manifest"] = args.source_manifest
        if args.heldout_review is not None:
            kwargs["heldout_review"] = args.heldout_review
        if args.heldout_prompts is not None:
            kwargs["heldout_prompts"] = args.heldout_prompts
        if args.prompt_file is not None:
            kwargs["prompt_file"] = args.prompt_file
        if args.target_dir is not None:
            kwargs["target_dir"] = args.target_dir
        if args.regression_case_id is not None:
            kwargs["regression_case_ids"] = tuple(args.regression_case_id)
        if args.output is not None:
            payload = write_text_preservation_v24_fixed_gate_protected_heldout_refresh_manifest(args.output, **kwargs)
        else:
            payload = build_text_preservation_v24_fixed_gate_protected_heldout_refresh_manifest(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-preservation-heldout-prompts":
        kwargs = {
            "count": args.count,
            "prompt_index_offset": args.prompt_index_offset,
            "src_prefix": args.src_prefix,
            "include_sample_marker": not args.no_sample_marker,
        }
        if args.output is not None:
            kwargs["output"] = args.output
        path = write_text_preservation_heldout_prompt_file(**kwargs)
        payload = {
            "prompt_file": str(path),
            "count": len(
                build_text_preservation_heldout_prompt_records(
                    count=args.count,
                    prompt_index_offset=args.prompt_index_offset,
                    src_prefix=args.src_prefix,
                    include_sample_marker=not args.no_sample_marker,
                )
            ),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(path)
        return 0

    if args.command == "text-preservation-heldout-eval-plan":
        kwargs = {
            "count": args.count,
            "student_name": args.student_name,
            "prompt_index_offset": args.prompt_index_offset,
            "src_prefix": args.src_prefix,
            "include_sample_marker": not args.no_sample_marker,
        }
        if args.prompt_file is not None:
            kwargs["prompt_file"] = args.prompt_file
        if args.out_root is not None:
            kwargs["out_root"] = args.out_root
        if args.report_root is not None:
            kwargs["report_root"] = args.report_root
        if args.student_checkpoint is not None:
            kwargs["student_checkpoint"] = args.student_checkpoint
        payload = build_text_preservation_heldout_eval_plan(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["prompt_write_command"])
            print(payload["qwen_command"])
            print(payload["gemma_command"])
            for case in payload["cases"]:
                for command in case["rename_commands"]:
                    print(command)
                print(case["compare_command"])
        return 0

    if args.command == "text-preservation-general-scene-prompts":
        kwargs = {"count": args.count}
        if args.output is not None:
            kwargs["output"] = args.output
        path = write_general_scene_regression_prompt_file(**kwargs)
        payload = {
            "prompt_file": str(path),
            "count": len(build_general_scene_regression_prompt_records(count=args.count)),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(path)
        return 0

    if args.command == "text-preservation-general-scene-eval-plan":
        kwargs = {"count": args.count, "student_name": args.student_name}
        if args.prompt_file is not None:
            kwargs["prompt_file"] = args.prompt_file
        if args.out_root is not None:
            kwargs["out_root"] = args.out_root
        if args.report_root is not None:
            kwargs["report_root"] = args.report_root
        if args.student_checkpoint is not None:
            kwargs["student_checkpoint"] = args.student_checkpoint
        payload = build_general_scene_regression_eval_plan(**kwargs)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["prompt_write_command"])
            print(payload["qwen_command"])
            print(payload["gemma_command"])
            for case in payload["cases"]:
                for command in case["rename_commands"]:
                    print(command)
                print(case["compare_command"])
        return 0

    if args.command == "pipeline-status":
        payload = pipeline_status()
        if args.output:
            write_pipeline_status(args.output)
        if args.json or not args.output:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"wrote pipeline status -> {args.output}")
        return 0

    if args.command == "prepare-gemma-cache":
        payload = {
            "plans": [plan.to_json_dict() for plan in default_split_gemma_cache_plans()],
            "pairing": audit_cache_pairing(),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for plan in payload["plans"]:
                print(plan["command"])
            print(json.dumps(payload["pairing"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "prepare-bridge-training":
        payload = BridgeTrainingPlan().to_json_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["command"])
            print(json.dumps(payload["cache_pairing"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "latest-manifest":
        latest = ManifestStore(args.manifest_root).latest()
        print(str(latest) if latest else "")
        return 0

    if args.command == "training-readiness":
        kwargs = {
            "manifest_limit": args.manifest_limit,
            "check_image_embed_exists": args.check_image_embeds,
        }
        if args.planner_out:
            kwargs["planner_out"] = args.planner_out
        if args.train_manifest:
            kwargs["train_manifest"] = args.train_manifest
        if args.eval_manifest:
            kwargs["eval_manifest"] = args.eval_manifest
        report = build_training_readiness_report(**kwargs)
        if args.output:
            write_training_readiness_report(report, args.output)
        if args.json or not args.output:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"wrote training readiness report -> {args.output}")
        return 0

    if args.command == "prepare-teacher-targets":
        export = export_teacher_subset(args.manifest, args.output_subset, limit=args.limit, target_dir=args.target_dir)
        payload = {
            "export": export.to_json_dict(),
            "target_cache": audit_target_cache(args.target_dir),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"wrote {export.rows_written} rows -> {export.output_subset}")
            print(export.command)
        return 0

    if args.command is None:
        parser.print_help()
        return 2

    renderer = None
    run_config = _run_config(args)
    if args.renderer == "external-script":
        renderer = ExternalAnimaRendererAdapter(args.image_root)
    elif args.renderer in {"real", "in-process"}:
        renderer = InProcessAnimaRendererAdapter(args.image_root, config=run_config, unet_dtype=args.unet_dtype)
    conductor = GemmAnimaConductor(
        session_id=args.session_id,
        manifest_root=Path(args.manifest_root),
        image_root=Path(args.image_root),
        renderer=renderer,
        config=run_config,
        plan_overrides=_run_plan_overrides(args),
    )
    response = conductor.handle_user_message(args.message)
    payload = {
        "mode": response.mode.value,
        "status": response.status.value,
        "message": response.message,
        "prompt": response.prompt,
        "manifest_path": str(response.manifest_path) if response.manifest_path else None,
        "output_path": str(response.output_path) if response.output_path else None,
        "progress": list(response.progress),
        "clarification_required": response.clarification_required,
        "conflict": response.conflict,
        "job_id": response.job_id,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload["message"])
        if payload["manifest_path"]:
            print(f"manifest: {payload['manifest_path']}")
        if payload["output_path"]:
            print(f"output: {payload['output_path']}")
    return 0

def _run_plan_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if getattr(args, "steps", None) is not None:
        overrides["steps"] = args.steps
    if getattr(args, "size", None) is not None:
        overrides["width"] = args.size
        overrides["height"] = args.size
    if getattr(args, "cfg", None) is not None:
        overrides["cfg"] = args.cfg
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    return overrides


def _run_config(args: argparse.Namespace) -> EngineConfig:
    return _config_with_model_overrides(
        anima_dm=getattr(args, "anima_dm", None),
        hiddenstage_bridge=getattr(args, "hiddenstage_bridge", None),
    )


def _config_with_model_overrides(
    *,
    anima_dm: str | None = None,
    hiddenstage_bridge: str | None = None,
) -> EngineConfig:
    config = EngineConfig()
    if not anima_dm and not hiddenstage_bridge:
        return config
    models = ModelConfig(
        gemma_planner_adapter=config.models.gemma_planner_adapter,
        gemma_vision_embedding=config.models.gemma_vision_embedding,
        anima_diffusion_model=Path(anima_dm) if anima_dm else config.models.anima_diffusion_model,
        anima_vae=config.models.anima_vae,
        hiddenstage_bridge=Path(hiddenstage_bridge) if hiddenstage_bridge else config.models.hiddenstage_bridge,
    )
    return EngineConfig(models=models, hardware=config.hardware, renderer_profiles=config.renderer_profiles)


if __name__ == "__main__":
    raise SystemExit(main())
