from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemmanima import GemmAnimaConductor
from gemmanima.api import (
    apply_payload_generation_preset,
    build_config,
    build_plan_overrides,
    classify_auto_intent,
    generation_plan_from_chat_contract,
)
from gemmanima.core.schemas import ChatTurn, GenerationPlan
from gemmanima.modules.in_process_anima_renderer import InProcessAnimaRendererAdapter
from gemmanima.modules.prompt_fallback import (
    build_safe_generation_prompt,
    build_safe_negative_prompt,
    enrich_generation_prompt,
)
from gemmanima.modules.tipo_runtime import (
    DEFAULT_TAG_PROMPT,
    TipoVisionConfig,
    clean_vision_tags,
    initialize_tipo_text_runtime,
    parse_image_generation_contract,
    run_tipo_text_chat,
    run_tipo_vision_tag,
)
from gemmanima.modules.wd_tagger import run_wd_vision_tag


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    message: str
    expected_terms: tuple[str, ...]
    expected_tags: tuple[str, ...]
    seed: int
    resolution_preset: str = "square_1024"
    orientation: str = ""


DEFAULT_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="forest_lantern",
        message="숲속에서 작은 랜턴을 들고 서 있는 소녀 일러스트 한 장 그려줘.",
        expected_terms=("forest", "lantern", "girl"),
        expected_tags=("1girl", "forest", "lantern", "standing"),
        seed=41001,
    ),
    EvalCase(
        case_id="rainy_cafe",
        message="비 오는 밤, 카페 창가에 앉아 있는 캐릭터를 따뜻한 분위기로 그려줘.",
        expected_terms=("rain", "night", "cafe", "window"),
        expected_tags=("rain", "night", "cafe", "sitting", "window"),
        seed=41002,
    ),
    EvalCase(
        case_id="pose_jump",
        message="역동적으로 점프하는 포즈의 애니 캐릭터, 파란 재킷과 운동화가 보이게.",
        expected_terms=("jump", "dynamic pose", "blue jacket", "sneakers"),
        expected_tags=("jumping", "dynamic pose", "blue jacket", "sneakers"),
        seed=41003,
    ),
    EvalCase(
        case_id="witch_library",
        message="마법 도서관에서 책을 펼친 작은 마녀를 1024 정사각형으로 그려줘.",
        expected_terms=("magic library", "book", "witch"),
        expected_tags=("witch", "book", "library", "hat"),
        seed=41004,
    ),
    EvalCase(
        case_id="snow_portrait",
        message="눈 내리는 거리에서 빨간 목도리를 한 소녀 초상화 만들어줘.",
        expected_terms=("snow", "street", "red scarf", "portrait"),
        expected_tags=("snow", "street", "red scarf", "portrait", "1girl"),
        seed=41005,
    ),
    EvalCase(
        case_id="artist_marker",
        message="@pottsness 느낌의 부드러운 색감으로 꽃밭에 앉은 캐릭터 한 장.",
        expected_terms=("@pottsness", "soft colors", "flower field", "sitting"),
        expected_tags=("flower field", "sitting", "soft colors", "1girl"),
        seed=41006,
    ),
)


def _case_payload(case: EvalCase, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": "auto",
        "message": case.message,
        "language": "ko",
        "renderer": args.renderer,
        "generation_preset": args.generation_preset,
        "bridge_profile": args.bridge_profile,
        "resolution_preset": case.resolution_preset,
        "orientation": case.orientation,
        "steps": args.steps,
        "cfg": args.cfg,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "seed": case.seed,
        "unet_dtype": args.unet_dtype,
        "tiled_vae": True,
        "memory_mode": args.memory_mode,
        "reserve_vram": args.reserve_vram,
        "session_id": f"korean-chat-image-quality-{case.case_id}",
    }
    if args.cpu_vae:
        payload["cpu_vae"] = True
    return payload


def _score_terms(text: str, expected: tuple[str, ...]) -> dict[str, Any]:
    lowered = text.lower()
    hits = [term for term in expected if term.lower() in lowered]
    return {
        "hits": hits,
        "misses": [term for term in expected if term not in hits],
        "score": round(len(hits) / max(1, len(expected)), 3),
    }


def _score_tags(tags: str, expected_tags: tuple[str, ...]) -> dict[str, Any]:
    normalized = {tag.strip().lower().replace("_", " ") for tag in tags.split(",") if tag.strip()}
    hits: list[str] = []
    for expected in expected_tags:
        needle = expected.lower().replace("_", " ")
        if any(needle == tag or needle in tag for tag in normalized):
            hits.append(expected)
    return {
        "hits": hits,
        "misses": [tag for tag in expected_tags if tag not in hits],
        "score": round(len(hits) / max(1, len(expected_tags)), 3),
        "tag_count": len(normalized),
    }


def _load_manifest(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Korean Chat Image Quality Eval",
        "",
        f"- started_at: `{report['started_at']}`",
        f"- renderer: `{report['settings']['renderer']}`",
        f"- preset: `{report['settings']['generation_preset']}`",
        f"- sampler/scheduler: `{report['settings']['sampler']}` / `{report['settings']['scheduler']}`",
        f"- steps/cfg: `{report['settings']['steps']}` / `{report['settings']['cfg']}`",
        f"- mean_intent_prompt_score: `{report['summary']['mean_intent_prompt_score']}`",
        f"- mean_vision_tag_score: `{report['summary']['mean_vision_tag_score']}`",
        "",
    ]
    for case in report["cases"]:
        lines.extend(
            [
                f"## {case['case_id']}",
                "",
                f"- Korean chat: {case['message']}",
                f"- status: `{case['status']}`",
                f"- auto_route: `{case.get('auto_route', {}).get('intent', '')}` confidence `{case.get('auto_route', {}).get('confidence', '')}`",
                f"- prompt: `{case.get('prompt', '')}`",
                f"- intent_prompt_score: `{case['intent_prompt_score']['score']}` hits `{', '.join(case['intent_prompt_score']['hits'])}`",
                f"- vision_tag_score: `{case['vision_tag_score']['score']}` hits `{', '.join(case['vision_tag_score']['hits'])}`",
                f"- vision_tags: `{case.get('vision_tags', '')}`",
            ]
        )
        image_path = case.get("output_path")
        if image_path:
            lines.extend(["", f"![{case['case_id']}]({Path(image_path).resolve().as_posix()})"])
        lines.append("")
    return "\n".join(lines) + "\n"


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["GEMMA_EMBED_ON_GPU"] = "1"
    if args.comfy_root:
        os.environ["GEMMANIMA_COMFY_ROOT"] = str(Path(args.comfy_root))
    if args.comfy_site_packages:
        os.environ["GEMMANIMA_COMFY_SITE_PACKAGES"] = str(Path(args.comfy_site_packages))
    if args.gemma_hf_dir:
        os.environ["GEMMANIMA_GEMMA_HF_DIR"] = str(Path(args.gemma_hf_dir))
    if args.gemma_hidden_device:
        os.environ["GEMMANIMA_GEMMA_HIDDEN_DEVICE"] = args.gemma_hidden_device
    if args.gemma_hidden_dtype:
        os.environ["GEMMANIMA_GEMMA_HIDDEN_DTYPE"] = args.gemma_hidden_dtype
    if args.gemma_embed_on_gpu is not None:
        os.environ["GEMMA_EMBED_ON_GPU"] = "1" if args.gemma_embed_on_gpu else "0"
    if args.swap_project_root:
        os.environ["GEMMANIMA_SWAP_PROJECT_ROOT"] = str(Path(args.swap_project_root))
    started = time.strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = Path(args.out_dir) / started
    base_dir = out_dir / "runs"
    base_dir.mkdir(parents=True, exist_ok=True)

    config = build_config({"bridge_profile": args.bridge_profile} if args.bridge_profile else {})
    text_init = initialize_tipo_text_runtime()
    renderer = InProcessAnimaRendererAdapter(
        base_dir / "images",
        config=config,
        unet_dtype=args.unet_dtype,
        tiled_vae=True,
        comfy_args=tuple([f"--{args.memory_mode}"] if args.memory_mode else ()),
    )
    cases: list[dict[str, Any]] = []
    for case in DEFAULT_CASES[: args.limit]:
        payload = _case_payload(case, args)
        auto_route = classify_auto_intent(payload, message=case.message, language="ko")
        contract_result = run_tipo_text_chat(
            message=case.message,
            language="ko",
            chat_mode="image_generation_request",
            history=[],
        )
        contract = contract_result.get("image_generation")
        if not isinstance(contract, dict) or contract.get("status") != "completed":
            contract = parse_image_generation_contract(
                str(contract_result.get("raw") or contract_result.get("message") or "")
            )
        if contract.get("status") == "completed":
            plan = apply_payload_generation_preset(generation_plan_from_chat_contract(contract), payload)
            plan = GenerationPlan.from_dict(
                {**plan.to_json_dict(), "prompt": enrich_generation_prompt(case.message, plan.prompt)}
            )
        else:
            plan = apply_payload_generation_preset(
                GenerationPlan(
                    prompt=build_safe_generation_prompt(case.message),
                    negative_prompt=build_safe_negative_prompt(),
                    seed=case.seed,
                ),
                payload,
            )
        plan = GenerationPlan.from_dict({**plan.to_json_dict(), **build_plan_overrides(payload)})
        conductor = GemmAnimaConductor(
            session_id=str(payload["session_id"]),
            manifest_root=base_dir / "manifests",
            image_root=base_dir / "images",
            renderer=renderer,
            config=config,
            plan_overrides=build_plan_overrides(payload),
        )
        conductor.history.append(ChatTurn(role="user", content=case.message))
        response = conductor.handle_generation_plan(case.message, plan)
        image_path = response.output_path
        tag_result: dict[str, Any] = {}
        vision_tags = ""
        if image_path and image_path.is_file():
            tag_result = run_wd_vision_tag(image_path=image_path)
            if tag_result.get("status") != "completed":
                tag_result = run_tipo_vision_tag(
                    image_path=image_path,
                    prompt=DEFAULT_TAG_PROMPT,
                    config=TipoVisionConfig(),
                )
            vision_tags = clean_vision_tags(str(tag_result.get("tags") or tag_result.get("raw") or ""))
        manifest = _load_manifest(str(response.manifest_path) if response.manifest_path else None)
        prompt_text = plan.prompt
        cases.append(
            {
                "case_id": case.case_id,
                "message": case.message,
                "status": response.status.value,
                "progress": list(response.progress),
                "auto_route": auto_route,
                "contract_status": contract.get("status"),
                "contract_error": contract.get("error", ""),
                "prompt": prompt_text,
                "negative_prompt": plan.negative_prompt,
                "plan": plan.to_json_dict(),
                "output_path": str(image_path) if image_path else "",
                "manifest_path": str(response.manifest_path) if response.manifest_path else "",
                "manifest": manifest,
                "vision_tags": vision_tags,
                "vision_tagger": tag_result.get("tagger", "tipo:vision"),
                "vision_raw": tag_result.get("raw", ""),
                "vision_error": tag_result.get("error", ""),
                "intent_prompt_score": _score_terms(prompt_text, case.expected_terms),
                "vision_tag_score": _score_tags(vision_tags, case.expected_tags),
            }
        )
    mean_prompt = round(
        sum(item["intent_prompt_score"]["score"] for item in cases) / max(1, len(cases)),
        3,
    )
    mean_vision = round(
        sum(item["vision_tag_score"]["score"] for item in cases) / max(1, len(cases)),
        3,
    )
    report = {
        "started_at": started,
        "out_dir": str(out_dir),
        "text_runtime_init": text_init,
        "settings": {
            "renderer": args.renderer,
            "generation_preset": args.generation_preset,
            "steps": args.steps,
            "cfg": args.cfg,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "unet_dtype": args.unet_dtype,
            "memory_mode": args.memory_mode,
            "cuda_visible_devices": args.cuda_visible_devices,
        },
        "summary": {
            "case_count": len(cases),
            "mean_intent_prompt_score": mean_prompt,
            "mean_vision_tag_score": mean_vision,
            "pass": mean_prompt >= args.min_prompt_score and mean_vision >= args.min_vision_score,
            "thresholds": {
                "min_prompt_score": args.min_prompt_score,
                "min_vision_score": args.min_vision_score,
            },
        },
        "cases": cases,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Korean chat-to-image and vision tag relevance.")
    parser.add_argument("--out-dir", default="runs/evals/korean_chat_image_quality")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--renderer", default="in-process")
    parser.add_argument("--generation-preset", default="anima_balanced")
    parser.add_argument("--bridge-profile", default="")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--sampler", default="euler_ancestral")
    parser.add_argument("--scheduler", default="sgm_uniform")
    parser.add_argument("--unet-dtype", default="fp8_e4m3fn_fast")
    parser.add_argument("--memory-mode", default="lowvram")
    parser.add_argument("--reserve-vram", default="")
    parser.add_argument("--cpu-vae", action="store_true")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--comfy-root", default="")
    parser.add_argument("--comfy-site-packages", default="")
    parser.add_argument("--gemma-hf-dir", default="")
    parser.add_argument("--gemma-hidden-device", default="")
    parser.add_argument("--gemma-hidden-dtype", default="")
    parser.add_argument("--gemma-embed-on-gpu", dest="gemma_embed_on_gpu", action="store_true", default=None)
    parser.add_argument("--no-gemma-embed-on-gpu", dest="gemma_embed_on_gpu", action="store_false")
    parser.add_argument("--swap-project-root", default="")
    parser.add_argument("--min-prompt-score", type=float, default=0.65)
    parser.add_argument("--min-vision-score", type=float, default=0.45)
    args = parser.parse_args()
    report = run_eval(args)
    print(json.dumps({"report": str(Path(report["out_dir"]) / "report.json"), "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0 if report["summary"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
