from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemmanima import GemmAnimaConductor
from gemmanima.core.manifest import ManifestStore
from gemmanima.core.model_registry import ModelRegistry
from gemmanima.training.readiness import build_training_readiness_report, write_training_readiness_report
from gemmanima.training.gemma_cache import audit_cache_pairing, default_split_gemma_cache_plans
from gemmanima.training.bridge_training import BridgeTrainingPlan
from gemmanima.training.teacher_targets import audit_target_cache, export_teacher_subset
from gemmanima.training.orchestrator import pipeline_status, write_pipeline_status
from gemmanima.training.evaluation import audit_bridge_checkpoint
from gemmanima.training.rebalance import build_rebalance_subsets
from gemmanima.training.real_render import audit_real_render_dependencies, build_real_render_command
from gemmanima.modules.real_anima_renderer import ExternalAnimaRendererAdapter
from gemmanima.rendering.backends import audit_renderer_backend, renderer_backend_profile
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
    run_parser.add_argument("--json", action="store_true", help="Print machine-readable response JSON.")

    health_parser = subparsers.add_parser("health", help="Print model registry health.")
    health_parser.add_argument("--json", action="store_true")

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

    if argv and argv[0] not in {
        "run",
        "health",
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
        export = export_teacher_subset(args.manifest, args.output_subset, limit=args.limit)
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
    config = EngineConfig()
    if not getattr(args, "anima_dm", None):
        return config
    models = ModelConfig(
        gemma_planner_adapter=config.models.gemma_planner_adapter,
        gemma_vision_embedding=config.models.gemma_vision_embedding,
        anima_diffusion_model=Path(args.anima_dm),
        anima_text_encoder=config.models.anima_text_encoder,
        anima_vae=config.models.anima_vae,
        hiddenstage_bridge=config.models.hiddenstage_bridge,
    )
    return EngineConfig(models=models, hardware=config.hardware, renderer_profiles=config.renderer_profiles)


if __name__ == "__main__":
    raise SystemExit(main())
