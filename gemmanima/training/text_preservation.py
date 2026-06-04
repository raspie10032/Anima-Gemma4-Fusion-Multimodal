from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from gemmanima.training.bridge_training import DEFAULT_PYTHON, DEFAULT_TRAIN_SCRIPT
from gemmanima.training.comparison import compare_text_roi_if_available
from gemmanima.training.evaluation import audit_bridge_checkpoint
from gemmanima.training.gemma_cache import GemmaCachePlan, audit_cache_pairing
from gemmanima.training.teacher_targets import build_cache_targets_command
from gemmanima.training.text_rendering_eval import (
    QWEN_BASELINE_PROMPT_FILE,
    build_text_rendering_qwen_prompt_records,
    _compare_command,
    _eval_generate_command,
    _quote_pwsh,
)


DEFAULT_TEXT_PRESERVATION_ROOT = Path(r"runs\cache\text_preservation_qwen")
DEFAULT_TEXT_PRESERVATION_TARGET_DIR = DEFAULT_TEXT_PRESERVATION_ROOT / "targets"
DEFAULT_TEXT_PRESERVATION_GEMMA_DIR = DEFAULT_TEXT_PRESERVATION_ROOT / "gemma"
DEFAULT_TEXT_PRESERVATION_BRIDGE = DEFAULT_TEXT_PRESERVATION_ROOT / "bridge" / "text_preservation_bridge.pt"
DEFAULT_TEXT_PRESERVATION_RESUME = Path(r"runs\cache\poc1_10k\bridge\poc1_10k_bridge.pt")
DEFAULT_TEXT_PRESERVATION_BLEND_ROOT = Path(r"runs\cache\text_preservation_blended")
DEFAULT_TEXT_PRESERVATION_BLEND_PROMPTS = Path(r"reports\text_preservation_blended\prompts.jsonl")
DEFAULT_TEXT_PRESERVATION_BLEND_BRIDGE = DEFAULT_TEXT_PRESERVATION_BLEND_ROOT / "bridge" / "text_preservation_blended_bridge.pt"
DEFAULT_TEXT_PRESERVATION_HELDOUT_PROMPTS = Path(r"reports\text_preservation_heldout_v4\prompts.jsonl")
DEFAULT_TEXT_PRESERVATION_HELDOUT_IMAGE_DIR = Path(r"runs\images\text_preservation_heldout_v4")
DEFAULT_TEXT_PRESERVATION_HELDOUT_REPORT_DIR = Path(r"reports\text_preservation_heldout_v4")
DEFAULT_TEXT_PRESERVATION_V4_BRIDGE = Path(r"runs\cache\text_preservation_blended_v4\bridge\text_preservation_blended_v4_bridge.pt")
DEFAULT_TEXT_PRESERVATION_BLEND_V5_ROOT = Path(r"runs\cache\text_preservation_blended_v5")
DEFAULT_TEXT_PRESERVATION_BLEND_V5_PROMPTS = Path(r"reports\text_preservation_blended_v5\prompts.jsonl")
DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V5_ROOT / "bridge" / "text_preservation_blended_v5_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V6_ROOT = Path(r"runs\cache\text_preservation_blended_v6")
DEFAULT_TEXT_PRESERVATION_BLEND_V6_PROMPTS = Path(r"reports\text_preservation_blended_v6\prompts.jsonl")
DEFAULT_TEXT_PRESERVATION_BLEND_V6_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V6_ROOT / "bridge" / "text_preservation_blended_v6_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V7_ROOT = Path(r"runs\cache\text_preservation_blended_v7")
DEFAULT_TEXT_PRESERVATION_BLEND_V7_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V7_ROOT / "bridge" / "text_preservation_blended_v7_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V8_ROOT = Path(r"runs\cache\text_preservation_blended_v8")
DEFAULT_TEXT_PRESERVATION_BLEND_V8_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V8_ROOT / "bridge" / "text_preservation_blended_v8_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V9_ROOT = Path(r"runs\cache\text_preservation_blended_v9")
DEFAULT_TEXT_PRESERVATION_BLEND_V9_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V9_ROOT / "bridge" / "text_preservation_blended_v9_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V10_ROOT = Path(r"runs\cache\text_preservation_blended_v10")
DEFAULT_TEXT_PRESERVATION_BLEND_V10_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V10_ROOT / "bridge" / "text_preservation_blended_v10_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V11_ROOT = Path(r"runs\cache\text_preservation_blended_v11")
DEFAULT_TEXT_PRESERVATION_BLEND_V11_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V11_ROOT / "bridge" / "text_preservation_blended_v11_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V12_ROOT = Path(r"runs\cache\text_preservation_blended_v12")
DEFAULT_TEXT_PRESERVATION_BLEND_V12_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V12_ROOT / "bridge" / "text_preservation_blended_v12_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_V9_ARTIFACT_FEEDBACK = Path(
    r"reports\text_rendering_qwen_baseline\v9_artifact_feedback.jsonl"
)
DEFAULT_TEXT_PRESERVATION_V9_LOSS_CONFIG = Path(
    r"reports\text_rendering_qwen_baseline\v9_artifact_gate_loss_config.json"
)
DEFAULT_TEXT_PRESERVATION_V10_ARTIFACT_FEEDBACK = Path(
    r"reports\text_rendering_qwen_baseline\v10_artifact_feedback.jsonl"
)
DEFAULT_TEXT_PRESERVATION_V10_LOSS_CONFIG = Path(
    r"reports\text_rendering_qwen_baseline\v10_artifact_gate_loss_config.json"
)
DEFAULT_TEXT_PRESERVATION_V11_ARTIFACT_FEEDBACK = Path(
    r"reports\text_rendering_qwen_baseline\v11_artifact_feedback.jsonl"
)
DEFAULT_TEXT_PRESERVATION_V11_LOSS_CONFIG = Path(
    r"reports\text_rendering_qwen_baseline\v11_artifact_gate_loss_config.json"
)
DEFAULT_TEXT_PRESERVATION_KV_DELTA_REPORT = Path(
    r"reports\text_rendering_qwen_baseline\kv_delta_audit_v9_v10_v11.json"
)
DEFAULT_TEXT_PRESERVATION_V12_SURFACE_PLAN_REPORT = Path(
    r"reports\text_rendering_qwen_baseline\v12_training_surface_plan.json"
)
DEFAULT_TEXT_PRESERVATION_RENDER_READABILITY_MANIFEST = Path(
    r"reports\text_rendering_qwen_baseline\render_readability_label_manifest_v12.json"
)
DEFAULT_TEXT_PRESERVATION_SURFACE_CURRICULUM_MANIFEST = Path(
    r"reports\text_rendering_qwen_baseline\surface_curriculum_manifest_v12.json"
)
DEFAULT_TEXT_PRESERVATION_QWEN_TARGET_REFRESH_MANIFEST = Path(
    r"reports\text_rendering_qwen_baseline\qwen_target_refresh_manifest_v12.json"
)
DEFAULT_TEXT_PRESERVATION_QWEN_TARGET_REFRESH_PROMPTS = Path(
    r"reports\text_rendering_qwen_baseline\qwen_target_refresh_prompts_v12.jsonl"
)
DEFAULT_TEXT_PRESERVATION_V12_TRAINER_CONTRACT_AUDIT = Path(
    r"reports\text_rendering_qwen_baseline\v12_trainer_surface_contract_audit.json"
)
DEFAULT_TEXT_PRESERVATION_V13_RECOVERY_PLAN_REPORT = Path(
    r"reports\text_rendering_qwen_baseline\v13_recovery_plan.json"
)
DEFAULT_TEXT_PRESERVATION_V13_GUARD_MANIFEST = Path(
    r"reports\text_rendering_qwen_baseline\v13_guard_weighted_manifest.json"
)
DEFAULT_TEXT_PRESERVATION_V13_GUARD_PROMPTS = Path(
    r"reports\text_rendering_qwen_baseline\v13_guard_weighted_prompts.jsonl"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V13_ROOT = Path(r"runs\cache\text_preservation_blended_v13")
DEFAULT_TEXT_PRESERVATION_BLEND_V13_TARGET_DIR = DEFAULT_TEXT_PRESERVATION_BLEND_V13_ROOT / "targets"
DEFAULT_TEXT_PRESERVATION_BLEND_V13_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V13_ROOT / "bridge" / "text_preservation_blended_v13_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_V14_FOCUS_MANIFEST = Path(
    r"reports\text_rendering_qwen_baseline\v14_focus_fixed_gate_manifest.json"
)
DEFAULT_TEXT_PRESERVATION_V14_FOCUS_PROMPTS = Path(
    r"reports\text_rendering_qwen_baseline\v14_focus_fixed_gate_prompts.jsonl"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V14_ROOT = Path(r"runs\cache\text_preservation_blended_v14")
DEFAULT_TEXT_PRESERVATION_BLEND_V14_TARGET_DIR = DEFAULT_TEXT_PRESERVATION_BLEND_V14_ROOT / "targets"
DEFAULT_TEXT_PRESERVATION_V17_TARGETED_TEACHER_REFRESH_MANIFEST = Path(
    r"reports\text_rendering_qwen_baseline\v17_targeted_teacher_refresh_manifest.json"
)
DEFAULT_TEXT_PRESERVATION_V17_TARGETED_TEACHER_REFRESH_PROMPTS = Path(
    r"reports\text_rendering_qwen_baseline\v17_targeted_teacher_refresh_prompts.jsonl"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V17_ROOT = Path(r"runs\cache\text_preservation_blended_v17")
DEFAULT_TEXT_PRESERVATION_BLEND_V17_TARGET_DIR = DEFAULT_TEXT_PRESERVATION_BLEND_V17_ROOT / "targets"
DEFAULT_TEXT_PRESERVATION_BLEND_V17_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V17_ROOT / "bridge" / "text_preservation_blended_v17_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_V18_TEA_MICRO_REFRESH_MANIFEST = Path(
    r"reports\text_rendering_qwen_baseline\v18_tea_micro_refresh_manifest.json"
)
DEFAULT_TEXT_PRESERVATION_V18_TEA_MICRO_REFRESH_PROMPTS = Path(
    r"reports\text_rendering_qwen_baseline\v18_tea_micro_refresh_prompts.jsonl"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V18_ROOT = Path(r"runs\cache\text_preservation_blended_v18")
DEFAULT_TEXT_PRESERVATION_BLEND_V18_TARGET_DIR = DEFAULT_TEXT_PRESERVATION_BLEND_V18_ROOT / "targets"
DEFAULT_TEXT_PRESERVATION_BLEND_V18_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V18_ROOT / "bridge" / "text_preservation_blended_v18_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_V19_DUAL_GUARD_REFRESH_MANIFEST = Path(
    r"reports\text_rendering_qwen_baseline\v19_dual_guard_refresh_manifest.json"
)
DEFAULT_TEXT_PRESERVATION_V19_DUAL_GUARD_REFRESH_PROMPTS = Path(
    r"reports\text_rendering_qwen_baseline\v19_dual_guard_refresh_prompts.jsonl"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V19_ROOT = Path(r"runs\cache\text_preservation_blended_v19")
DEFAULT_TEXT_PRESERVATION_BLEND_V19_TARGET_DIR = DEFAULT_TEXT_PRESERVATION_BLEND_V19_ROOT / "targets"
DEFAULT_TEXT_PRESERVATION_BLEND_V19_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V19_ROOT / "bridge" / "text_preservation_blended_v19_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_V23_HARD_HELDOUT_REFRESH_MANIFEST = Path(
    r"reports\text_rendering_qwen_baseline\v23_hard_heldout_refresh_manifest.json"
)
DEFAULT_TEXT_PRESERVATION_V23_HARD_HELDOUT_REFRESH_PROMPTS = Path(
    r"reports\text_rendering_qwen_baseline\v23_hard_heldout_refresh_prompts.jsonl"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V22_ALPHA28_BRIDGE = Path(
    r"runs\cache\text_preservation_blended_v22\bridge\text_preservation_blended_v22_alpha28_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V23_ROOT = Path(r"runs\cache\text_preservation_blended_v23")
DEFAULT_TEXT_PRESERVATION_BLEND_V23_TARGET_DIR = DEFAULT_TEXT_PRESERVATION_BLEND_V23_ROOT / "targets"
DEFAULT_TEXT_PRESERVATION_BLEND_V23_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V23_ROOT / "bridge" / "text_preservation_blended_v23_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_V24_FIXED_GATE_PROTECTED_HELDOUT_REFRESH_MANIFEST = Path(
    r"reports\text_rendering_qwen_baseline\v24_fixed_gate_protected_heldout_refresh_manifest.json"
)
DEFAULT_TEXT_PRESERVATION_V24_FIXED_GATE_PROTECTED_HELDOUT_REFRESH_PROMPTS = Path(
    r"reports\text_rendering_qwen_baseline\v24_fixed_gate_protected_heldout_refresh_prompts.jsonl"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V24_ROOT = Path(r"runs\cache\text_preservation_blended_v24")
DEFAULT_TEXT_PRESERVATION_BLEND_V24_TARGET_DIR = DEFAULT_TEXT_PRESERVATION_BLEND_V24_ROOT / "targets"
DEFAULT_TEXT_PRESERVATION_BLEND_V24_BRIDGE = (
    DEFAULT_TEXT_PRESERVATION_BLEND_V24_ROOT / "bridge" / "text_preservation_blended_v24_bridge.pt"
)
DEFAULT_TEXT_PRESERVATION_BLEND_V12_TARGET_DIR = DEFAULT_TEXT_PRESERVATION_BLEND_V12_ROOT / "targets"
DEFAULT_GENERAL_SCENE_REGRESSION_PROMPTS = Path(r"reports\general_scene_regression_v5\prompts.jsonl")
DEFAULT_GENERAL_SCENE_REGRESSION_IMAGE_DIR = Path(r"runs\images\general_scene_regression_v5")
DEFAULT_GENERAL_SCENE_REGRESSION_REPORT_DIR = Path(r"reports\general_scene_regression_v5")
DEFAULT_GENERAL_SCENE_REGRESSION_50_PROMPTS = Path(r"reports\general_scene_regression_v5_50\prompts.jsonl")
DEFAULT_GENERAL_SCENE_REGRESSION_50_IMAGE_DIR = Path(r"runs\images\general_scene_regression_v5_50")
DEFAULT_GENERAL_SCENE_REGRESSION_50_REPORT_DIR = Path(r"reports\general_scene_regression_v5_50")


_TEXT_SURFACES = (
    "neon sign",
    "book cover",
    "poster",
    "glass jar label",
    "paper note",
    "game UI panel",
    "street banner",
    "ticket stub",
    "shop awning",
    "warning label",
    "map title",
    "magic seal",
    "wooden plaque",
    "menu board",
    "shipping label",
    "arcade marquee",
)

_TEXT_SETTINGS = (
    "a rainy storefront window",
    "a blue hardcover book on a maple desk",
    "a clean transit poster on a tiled wall",
    "a small kitchen jar in morning light",
    "a pinned note on cork board",
    "a compact fantasy game status panel",
    "a fabric banner stretched between posts",
    "a cream concert ticket on a table",
    "a striped awning above a closed shop",
    "a simple label on a metal kettle",
    "a folded illustrated map",
    "a glowing circle carved into dark stone",
    "a polished wooden door",
    "a chalk menu in a quiet cafe",
    "a cardboard package label",
    "a retro arcade cabinet header",
)

_TEXT_VALUES = (
    "NOVA CAFE",
    "MOON INDEX",
    "SKY RAIL",
    "MINT",
    "CALL LUNA",
    "MP 88",
    "EAST GATE",
    "ROW 7",
    "SUN BAKERY",
    "HOT",
    "OLD HARBOR",
    "AETHER",
    "BLUE STAR",
    "LEVEL 12",
    "OPEN",
    "CLOUD NINE",
    "TEA",
    "HP 42",
    "LUNA GATE",
    "STAR ATLAS",
    "MEET AT DAWN",
    "NOON BELL",
    "GOLD KEY",
    "ROOM 305",
    "SALT",
    "NORTH EXIT",
    "RED MOON",
    "KITE SHOP",
    "ZONE 4",
    "PIXEL BAR",
    "JAZZ NIGHT",
    "EMBER",
    "MILK",
    "BOOK 17",
    "RIVER WALK",
    "ALPHA",
    "BETA 9",
    "LILAC",
    "CAFE 24",
    "DREAM LOG",
    "FROST",
    "SUNSET",
    "BLOOM",
    "SILVER LINE",
    "ARCADE",
    "TICKET A",
    "HONEY",
    "VIOLET",
    "ORBIT",
    "DELTA",
    "SPARK",
    "GREEN TEA",
    "WISH",
    "MAPLE",
    "CROWN",
    "BRIGHT",
    "QUIET HILL",
    "STATION 8",
    "MORNING",
    "NIGHT BUS",
    "RUBY",
    "PEACH",
    "COSMOS",
    "LIGHT",
)


_TEXT_STYLES = (
    "center the text and make every letter readable",
    "use bold block letters with high contrast",
    "use serif letters with clean spacing",
    "use handwritten letters that remain readable",
    "use simple sans-serif letters with generous spacing",
    "use clear white letters on a dark background",
    "use dark ink on a pale label",
    "use glowing cyan letters without distorting the word",
)

_HARD_NEGATIVE_TEXT_VALUES = (
    "BRIGHT",
    "NOON BELL",
    "GREEN TEA",
    "QUIET HILL",
    "TICKET A",
    "SILVER LINE",
    "VIOLET",
    "ROW 7",
    "SUN BAKERY",
    "LEVEL 12",
    "ROOM 305",
    "MAPLE",
    "GOLD KEY",
    "OLD HARBOR",
    "PIXEL BAR",
    "AETHER",
)

_HARD_NEGATIVE_SURFACES = (
    "dense black-and-white poster panel",
    "small shopfront awning sign",
    "retro arcade cabinet marquee",
    "folded ticket stub with tiny print",
    "book cover label at a slight angle",
    "glass jar label with thick outline art",
    "menu board with boxed text",
    "street banner with thin serif letters",
)

_HARD_NEGATIVE_SETTINGS = (
    "a high-contrast ink illustration on white paper",
    "a compact storefront drawn as clean line art",
    "a dark UI panel with decorative borders",
    "a close-up tabletop product label",
    "a tilted blue book on a wooden desk",
    "a black background with white label art",
    "a narrow sign in a busy frame",
    "a pale poster on tiled wall",
)

_HARD_NEGATIVE_STYLES = (
    "use large isolated letters with generous spacing",
    "avoid extra surrounding words and keep only the target text readable",
    "use simple block letters and preserve the exact spelling",
    "make the target text the only prominent lettering",
    "use high contrast between letters and background",
    "center the target text and keep it away from decorative borders",
)

_HARD_NEGATIVE_LIGHTING = (
    "soft studio lighting",
    "flat catalog lighting",
    "low-key dramatic lighting",
    "clean overcast daylight",
    "cool cyan rim light",
    "warm desk lamp light",
    "neutral white background light",
    "subtle rainy reflections",
    "matte paper texture",
    "glossy product-photo finish",
)

_GENERAL_SCENE_REGRESSION_CASES = (
    (
        "cafe_interior",
        "Draw a cozy indoor cafe anime illustration with one seated character, a cup on the table, window light, shelves in the background, balanced composition.",
        "character_environment_balance",
    ),
    (
        "forest_character",
        "Draw a moonlit forest path with a small anime character silhouette, layered trees, soft green light, and visible depth.",
        "depth_and_mood",
    ),
    (
        "city_night",
        "Draw a rainy city night street scene with reflections, umbrellas, shop lights, and a cinematic anime atmosphere.",
        "urban_lighting",
    ),
    (
        "fantasy_mage",
        "Draw a full-body fantasy mage in a detailed robe standing in ruins, magical glow, clear pose, background included.",
        "full_body_character",
    ),
    (
        "beach_day",
        "Draw a bright beach day scene with towels, parasol, shells, sparkling water, and one relaxed anime character.",
        "object_rich_scene",
    ),
    (
        "market_alley",
        "Draw a lively market alley with several distant shoppers, fruit stalls, fabric awnings, and warm afternoon light.",
        "crowd_and_props",
    ),
    (
        "robot_closeup",
        "Draw a close-up of a small friendly robot with glossy metal, cables, glowing eyes, and workshop clutter behind it.",
        "material_detail",
    ),
    (
        "classroom",
        "Draw a quiet school classroom with desks, bags, chalkboard, soft sunlight, and one character near the window.",
        "interior_layout",
    ),
    (
        "food_still_life",
        "Draw an appetizing anime food still life with a steaming bowl, side dishes, chopsticks, and warm table lighting.",
        "food_rendering",
    ),
    (
        "action_fantasy",
        "Draw a fantasy action scene just before a duel, dynamic pose, weapon glow, dust, and a dramatic background.",
        "motion_and_composition",
    ),
    (
        "spaceship_interior",
        "Draw a science-fiction spaceship interior with consoles, soft blue lighting, one pilot silhouette, and deep perspective.",
        "hard_surface_space",
    ),
    (
        "rainy_street",
        "Draw a rainy street with a single umbrella, puddle reflections, distant lamps, and a calm cinematic mood.",
        "weather_lighting",
    ),
    (
        "desk_workspace",
        "Draw a tidy desk workspace with laptop, notebook, lamp, pencils, sticky notes, and a soft evening atmosphere.",
        "small_object_coherence",
    ),
    (
        "garden_teahouse",
        "Draw a peaceful garden teahouse with stepping stones, flowers, paper lanterns, and one character in the distance.",
        "environment_detail",
    ),
    (
        "library_corner",
        "Draw a warm library corner with stacked books, ladder, plush chair, dust motes, and gentle anime lighting.",
        "dense_background",
    ),
)

_GENERAL_SCENE_REGRESSION_VARIANTS = (
    "wide establishing composition",
    "medium shot with clear foreground props",
    "low-angle cinematic composition",
    "soft morning color palette",
    "dramatic evening color palette",
    "slightly closer crop with detailed background",
)


def build_text_preservation_bridge_plan(
    *,
    prompt_file: str | Path = QWEN_BASELINE_PROMPT_FILE,
    target_dir: str | Path = DEFAULT_TEXT_PRESERVATION_TARGET_DIR,
    gemma_dir: str | Path = DEFAULT_TEXT_PRESERVATION_GEMMA_DIR,
    output: str | Path = DEFAULT_TEXT_PRESERVATION_BRIDGE,
    resume_kv: str | Path = DEFAULT_TEXT_PRESERVATION_RESUME,
    sample_count: int = 6,
    gpu_index: int = 0,
    epochs: int = 40,
    lr: float = 1e-4,
    batch_size: int = 2,
    accum: int = 1,
    val: int = 6,
    prefetch_gb: float = 1.0,
) -> dict[str, Any]:
    if gpu_index != 0:
        raise ValueError("text preservation bridge plan is 4070-only; use CUDA device 0")

    prompt_path = Path(prompt_file)
    target_root = Path(target_dir)
    gemma_root = Path(gemma_dir)
    output_path = Path(output)
    resume_path = Path(resume_kv)
    gemma_plan = GemmaCachePlan(
        name="text_preservation_4070",
        gpu_index=gpu_index,
        target_patterns=("*.pt",),
        embed_on_gpu=True,
        batch_size=8,
    )

    train_command = _bridge_train_command(
        target_dir=target_root,
        gemma_dir=gemma_root,
        output=output_path,
        resume_kv=resume_path,
        gpu_index=gpu_index,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        accum=accum,
        val=val,
        prefetch_gb=prefetch_gb,
    )

    return {
        "stage": "text_preservation_micro_overfit",
        "mode": "executable",
        "executes_gpu_commands": True,
        "sample_count": sample_count,
        "prompt_file": _json_path(prompt_path),
        "target_dir": _json_path(target_root),
        "gemma_dir": _json_path(gemma_root),
        "output": _json_path(output_path),
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "training_strategy": {
            "method": "existing_te_distillation_micro_overfit",
            "target_cache": "Qwen/Anima llm_adapter target from 06_cache_targets.py",
            "source_cache": "Gemma hidden states from 07_cache_gemma_batched.py",
            "objective": "08_train_stream_batched.py MSE bridge objective",
            "resume_from": _json_path(resume_path),
            "intent": "preserve Qwen teacher text-rendering features before broad scene semantics",
        },
        "setup_commands": [
            f"New-Item -ItemType Directory -Force -Path \"{target_root}\"",
            f"New-Item -ItemType Directory -Force -Path \"{gemma_root}\"",
            f"New-Item -ItemType Directory -Force -Path \"{output_path.parent}\"",
        ],
        "target_cache_command": build_cache_targets_command(
            subset_path=prompt_path,
            outdir=target_root,
            shard=1000,
            gpu_index=gpu_index,
        ),
        "gemma_cache_command": gemma_plan.command(
            subset=prompt_path,
            target_dir=target_root,
            outdir=gemma_root,
        ),
        "train_command": train_command,
        "status_command": "python -m gemmanima.cli text-preservation-bridge-status --json",
        "eval_command": f"python -m gemmanima.cli bridge-eval-status --checkpoint \"{output_path}\" --json",
        "post_train_qwen_eval_plan_command": (
            "python -m gemmanima.cli text-rendering-qwen-baseline-plan "
            f"--student-checkpoint \"{output_path}\" "
            "--student-name gemma_text_preservation --json"
        ),
    }


def build_text_preservation_bridge_status(
    *,
    target_dir: str | Path = DEFAULT_TEXT_PRESERVATION_TARGET_DIR,
    gemma_dir: str | Path = DEFAULT_TEXT_PRESERVATION_GEMMA_DIR,
    bridge_checkpoint: str | Path = DEFAULT_TEXT_PRESERVATION_BRIDGE,
) -> dict[str, Any]:
    pairing = audit_cache_pairing(target_dir=target_dir, gemma_dir=gemma_dir)
    bridge = audit_bridge_checkpoint(bridge_checkpoint)
    return {
        "stage": "text_preservation_micro_overfit",
        "target_dir": _json_path(Path(target_dir)),
        "gemma_dir": _json_path(Path(gemma_dir)),
        "bridge_checkpoint": _json_path(Path(bridge_checkpoint)),
        "ready_for_training": pairing["ready_for_bridge_training"],
        "cache_pairing": pairing,
        "bridge": bridge,
    }


def build_text_preservation_prompt_records(
    *,
    count: int = 48,
    start_idx: int = 900000,
    include_eval_cases: bool = False,
    prompt_index_offset: int = 0,
    src_prefix: str = "text_preserve_blend",
    include_sample_marker: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if include_eval_cases:
        records.extend(build_text_rendering_qwen_prompt_records())
    for index in range(count):
        prompt_index = prompt_index_offset + index
        surface = _TEXT_SURFACES[prompt_index % len(_TEXT_SURFACES)]
        target_text = _TEXT_VALUES[(prompt_index * 5) % len(_TEXT_VALUES)]
        setting = _TEXT_SETTINGS[(prompt_index * 7) % len(_TEXT_SETTINGS)]
        style = _TEXT_STYLES[(prompt_index * 11) % len(_TEXT_STYLES)]
        layout = ["large centered crop", "slight angle", "flat front view", "close-up readable crop"][
            (prompt_index * 13) % 4
        ]
        marker = f"; sample {prompt_index:04d}" if include_sample_marker else ""
        text = (
            f"Object-only text preservation blend: draw {setting} with a {surface} "
            f"that clearly reads {target_text}; {style}; {layout}{marker}. "
            "No people, no characters, no faces, no hands."
        )
        records.append(
            {
                "text": text,
                "src": f"{src_prefix}_{index:03d}",
                "id": f"{src_prefix}_{index:03d}",
                "idx": start_idx + index,
                "eval_idx": index,
                "target_text": target_text,
            }
        )
    return records


def build_text_preservation_v6_prompt_records(
    *,
    count: int = 320,
    start_idx: int = 970000,
    prompt_index_offset: int = 40000,
    src_prefix: str = "text_preserve_v6_hard",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    layouts = (
        "large centered crop",
        "flat front view",
        "close-up readable crop",
        "slight angle with text still centered",
    )
    for index in range(count):
        prompt_index = prompt_index_offset + index
        target_text = _HARD_NEGATIVE_TEXT_VALUES[index % len(_HARD_NEGATIVE_TEXT_VALUES)]
        surface = _HARD_NEGATIVE_SURFACES[(prompt_index * 3) % len(_HARD_NEGATIVE_SURFACES)]
        setting = _HARD_NEGATIVE_SETTINGS[(prompt_index * 5) % len(_HARD_NEGATIVE_SETTINGS)]
        style = _HARD_NEGATIVE_STYLES[(prompt_index * 7) % len(_HARD_NEGATIVE_STYLES)]
        layout = layouts[(prompt_index * 11) % len(layouts)]
        lighting = _HARD_NEGATIVE_LIGHTING[(prompt_index * 13) % len(_HARD_NEGATIVE_LIGHTING)]
        text = (
            f"Object-only hard negative text preservation: draw {setting} with a {surface} "
            f"that clearly reads {target_text}; {style}; {layout}; {lighting}. "
            "No people, no characters, no faces, no hands. Do not add any other readable words."
        )
        records.append(
            {
                "text": text,
                "src": f"{src_prefix}_{index:03d}",
                "id": f"{src_prefix}_{index:03d}",
                "idx": start_idx + index,
                "eval_idx": index,
                "target_text": target_text,
            }
        )
    return records


def write_text_preservation_v6_prompt_file(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V6_PROMPTS,
    *,
    count: int = 320,
    include_eval_cases: bool = True,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if include_eval_cases:
        records.extend(build_text_rendering_qwen_prompt_records())
    records.extend(build_text_preservation_v6_prompt_records(count=count))
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def build_general_scene_regression_prompt_records(
    *,
    count: int = 15,
    start_idx: int = 980000,
    src_prefix: str = "general_scene_regression",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(count):
        case_index = index % len(_GENERAL_SCENE_REGRESSION_CASES)
        variant_index = index // len(_GENERAL_SCENE_REGRESSION_CASES)
        category, prompt, focus = _GENERAL_SCENE_REGRESSION_CASES[case_index]
        if variant_index:
            variant = _GENERAL_SCENE_REGRESSION_VARIANTS[
                variant_index % len(_GENERAL_SCENE_REGRESSION_VARIANTS)
            ]
            prompt = f"{prompt} Use {variant}."
        records.append(
            {
                "text": prompt,
                "src": f"{src_prefix}_{index:03d}",
                "id": f"{src_prefix}_{index:03d}",
                "idx": start_idx + index,
                "eval_idx": index,
                "category": category,
                "regression_focus": focus,
            }
        )
    return records


def write_general_scene_regression_prompt_file(
    output: str | Path = DEFAULT_GENERAL_SCENE_REGRESSION_PROMPTS,
    *,
    count: int = 15,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = build_general_scene_regression_prompt_records(count=count)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def write_text_preservation_prompt_file(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_PROMPTS,
    *,
    count: int = 48,
    start_idx: int = 900000,
    include_eval_cases: bool = False,
    prompt_index_offset: int = 0,
    src_prefix: str = "text_preserve_blend",
    include_sample_marker: bool = True,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = build_text_preservation_prompt_records(
        count=count,
        start_idx=start_idx,
        include_eval_cases=include_eval_cases,
        prompt_index_offset=prompt_index_offset,
        src_prefix=src_prefix,
        include_sample_marker=include_sample_marker,
    )
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def build_text_preservation_heldout_prompt_records(
    *,
    count: int = 64,
    start_idx: int = 950000,
    prompt_index_offset: int = 10000,
    src_prefix: str = "text_preserve_heldout",
    include_sample_marker: bool = True,
) -> list[dict[str, Any]]:
    return build_text_preservation_prompt_records(
        count=count,
        start_idx=start_idx,
        prompt_index_offset=prompt_index_offset,
        src_prefix=src_prefix,
        include_sample_marker=include_sample_marker,
    )


def write_text_preservation_heldout_prompt_file(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_HELDOUT_PROMPTS,
    *,
    count: int = 64,
    prompt_index_offset: int = 10000,
    src_prefix: str = "text_preserve_heldout",
    include_sample_marker: bool = True,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = build_text_preservation_heldout_prompt_records(
        count=count,
        prompt_index_offset=prompt_index_offset,
        src_prefix=src_prefix,
        include_sample_marker=include_sample_marker,
    )
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def build_text_preservation_heldout_eval_plan(
    *,
    count: int = 64,
    gpu_index: int = 0,
    prompt_file: str | Path = DEFAULT_TEXT_PRESERVATION_HELDOUT_PROMPTS,
    out_root: str | Path = DEFAULT_TEXT_PRESERVATION_HELDOUT_IMAGE_DIR,
    report_root: str | Path = DEFAULT_TEXT_PRESERVATION_HELDOUT_REPORT_DIR,
    student_checkpoint: str | Path = DEFAULT_TEXT_PRESERVATION_V4_BRIDGE,
    student_name: str = "gemma_text_preservation_blended_v4",
    prompt_index_offset: int = 10000,
    src_prefix: str = "text_preserve_heldout",
    include_sample_marker: bool = True,
    seed_base: int = 550001,
    size: int = 512,
    steps: int = 20,
    cfg: float = 4.5,
    sampler: str = "euler",
    scheduler: str = "normal",
    unet_dtype: str = "default",
) -> dict[str, Any]:
    if gpu_index != 0:
        raise ValueError("text preservation heldout eval is 4070-only; use CUDA device 0")
    prompt_path = Path(prompt_file)
    out_root_path = Path(out_root)
    report_root_path = Path(report_root)
    student_checkpoint_path = Path(student_checkpoint)
    records = build_text_preservation_heldout_prompt_records(
        count=count,
        prompt_index_offset=prompt_index_offset,
        src_prefix=src_prefix,
        include_sample_marker=include_sample_marker,
    )
    qwen_dir = out_root_path / "qwen"
    gemma_dir = out_root_path / student_name
    cases = [
        _heldout_eval_case(
            record,
            index=index,
            qwen_dir=qwen_dir,
            gemma_dir=gemma_dir,
            report_root=report_root_path,
            student_checkpoint=student_checkpoint_path,
            student_name=student_name,
            seed=seed_base + index,
        )
        for index, record in enumerate(records)
    ]
    return {
        "stage": "text_preservation_heldout_eval",
        "mode": "executable",
        "executes_gpu_commands": True,
        "prompt_file": _json_path(prompt_path),
        "out_root": _json_path(out_root_path),
        "report_root": _json_path(report_root_path),
        "student_checkpoint": _json_path(student_checkpoint_path),
        "student_name": student_name,
        "seed_base": seed_base,
        "prompt_index_offset": prompt_index_offset,
        "src_prefix": src_prefix,
        "include_sample_marker": include_sample_marker,
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "prompt_write_command": (
            "python -m gemmanima.cli text-preservation-heldout-prompts "
            f"--output {_quote_pwsh(str(prompt_path))} --count {count} --json"
            + (f" --prompt-index-offset {prompt_index_offset}" if prompt_index_offset != 10000 else "")
            + (f" --src-prefix {src_prefix}" if src_prefix != "text_preserve_heldout" else "")
            + (" --no-sample-marker" if not include_sample_marker else "")
        ),
        "qwen_command": _eval_generate_command(
            gpu_index=gpu_index,
            mode="qwen",
            name="qwen",
            prompt_file=prompt_path,
            out_root=out_root_path,
            seed=seed_base,
            size=size,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            unet_dtype=unet_dtype,
            limit=count,
        ),
        "gemma_command": _eval_generate_command(
            gpu_index=gpu_index,
            mode="gemma",
            name=student_name,
            prompt_file=prompt_path,
            out_root=out_root_path,
            seed=seed_base,
            size=size,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            unet_dtype=unet_dtype,
            limit=count,
            adapter=student_checkpoint_path,
        ),
        "total_cases": len(cases),
        "cases": cases,
    }


def build_general_scene_regression_eval_plan(
    *,
    count: int = 15,
    gpu_index: int = 0,
    prompt_file: str | Path = DEFAULT_GENERAL_SCENE_REGRESSION_PROMPTS,
    out_root: str | Path = DEFAULT_GENERAL_SCENE_REGRESSION_IMAGE_DIR,
    report_root: str | Path = DEFAULT_GENERAL_SCENE_REGRESSION_REPORT_DIR,
    student_checkpoint: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE,
    student_name: str = "gemma_text_preservation_blended_v5",
    seed_base: int = 660001,
    size: int = 512,
    steps: int = 20,
    cfg: float = 4.5,
    sampler: str = "euler",
    scheduler: str = "normal",
    unet_dtype: str = "default",
) -> dict[str, Any]:
    if gpu_index != 0:
        raise ValueError("general scene regression eval is 4070-only; use CUDA device 0")
    prompt_path = Path(prompt_file)
    out_root_path = Path(out_root)
    report_root_path = Path(report_root)
    student_checkpoint_path = Path(student_checkpoint)
    records = build_general_scene_regression_prompt_records(count=count)
    qwen_dir = out_root_path / "qwen"
    gemma_dir = out_root_path / student_name
    cases = [
        _general_scene_eval_case(
            record,
            index=index,
            qwen_dir=qwen_dir,
            gemma_dir=gemma_dir,
            report_root=report_root_path,
            student_checkpoint=student_checkpoint_path,
            student_name=student_name,
            seed=seed_base + index,
        )
        for index, record in enumerate(records)
    ]
    return {
        "stage": "general_scene_regression_eval",
        "mode": "executable",
        "executes_gpu_commands": True,
        "prompt_file": _json_path(prompt_path),
        "out_root": _json_path(out_root_path),
        "report_root": _json_path(report_root_path),
        "student_checkpoint": _json_path(student_checkpoint_path),
        "student_name": student_name,
        "seed_base": seed_base,
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "prompt_write_command": (
            "python -m gemmanima.cli text-preservation-general-scene-prompts "
            f"--output {_quote_pwsh(str(prompt_path))} --count {count} --json"
        ),
        "qwen_command": _eval_generate_command(
            gpu_index=gpu_index,
            mode="qwen",
            name="qwen",
            prompt_file=prompt_path,
            out_root=out_root_path,
            seed=seed_base,
            size=size,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            unet_dtype=unet_dtype,
            limit=count,
        ),
        "gemma_command": _eval_generate_command(
            gpu_index=gpu_index,
            mode="gemma",
            name=student_name,
            prompt_file=prompt_path,
            out_root=out_root_path,
            seed=seed_base,
            size=size,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            unet_dtype=unet_dtype,
            limit=count,
            adapter=student_checkpoint_path,
        ),
        "total_cases": len(cases),
        "cases": cases,
    }


def build_text_preservation_blended_plan(
    *,
    prompt_file: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_PROMPTS,
    root: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_ROOT,
    output: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_BRIDGE,
    resume_kv: str | Path = DEFAULT_TEXT_PRESERVATION_BRIDGE,
    source_general_target_dir: str | Path = r"runs\cache\poc1_10k\targets",
    source_general_gemma_dir: str | Path = r"runs\cache\poc1_10k\gemma",
    sample_count: int = 48,
    include_eval_cases: bool = True,
    prompt_index_offset: int = 0,
    src_prefix: str = "text_preserve_blend",
    include_sample_marker: bool = True,
    text_repeat: int = 16,
    general_shards: int = 1,
    gpu_index: int = 0,
    epochs: int = 2,
    lr: float = 5e-5,
    batch_size: int = 2,
    accum: int = 1,
    val: int = 48,
    prefetch_gb: float = 4.0,
) -> dict[str, Any]:
    if gpu_index != 0:
        raise ValueError("text preservation blended plan is 4070-only; use CUDA device 0")
    root_path = Path(root)
    prompt_path = Path(prompt_file)
    text_target_dir = root_path / "text_targets"
    text_gemma_dir = root_path / "text_gemma"
    blend_target_dir = root_path / "blend_targets"
    blend_gemma_dir = root_path / "blend_gemma"
    output_path = Path(output)
    resume_path = Path(resume_kv)
    total_text_prompts = sample_count + (6 if include_eval_cases else 0)
    text_shards = math.ceil(total_text_prompts / 1000)
    gemma_plan = GemmaCachePlan(
        name="text_preservation_blended_4070",
        gpu_index=gpu_index,
        target_patterns=("*.pt",),
        embed_on_gpu=True,
        batch_size=8,
    )
    return {
        "stage": "text_preservation_blended_candidate",
        "mode": "executable",
        "executes_gpu_commands": True,
        "sample_count": total_text_prompts,
        "prompt_file": _json_path(prompt_path),
        "root": _json_path(root_path),
        "text_target_dir": _json_path(text_target_dir),
        "text_gemma_dir": _json_path(text_gemma_dir),
        "blend_target_dir": _json_path(blend_target_dir),
        "blend_gemma_dir": _json_path(blend_gemma_dir),
        "output": _json_path(output_path),
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "blend": {
            "text_repeat": text_repeat,
            "general_shards": general_shards,
            "text_shards": text_shards,
            "link_strategy": "hardlink",
            "text_source_shard": _json_path(text_target_dir / "shard_0000.pt"),
            "text_source_shards": [
                _json_path(text_target_dir / f"shard_{index:04d}.pt")
                for index in range(text_shards)
            ],
            "general_source_target_dir": _json_path(Path(source_general_target_dir)),
            "general_source_gemma_dir": _json_path(Path(source_general_gemma_dir)),
        },
        "training_strategy": {
            "resume_from": _json_path(resume_path),
            "objective": "08_train_stream_batched.py MSE bridge objective",
            "text_prompt_count": total_text_prompts,
            "text_repeat": text_repeat,
            "general_shards": general_shards,
            "include_sample_marker": include_sample_marker,
        },
        "setup_commands": [
            f"New-Item -ItemType Directory -Force -Path \"{prompt_path.parent}\"",
            f"New-Item -ItemType Directory -Force -Path \"{text_target_dir}\"",
            f"New-Item -ItemType Directory -Force -Path \"{text_gemma_dir}\"",
            f"New-Item -ItemType Directory -Force -Path \"{blend_target_dir}\"",
            f"New-Item -ItemType Directory -Force -Path \"{blend_gemma_dir}\"",
            f"New-Item -ItemType Directory -Force -Path \"{output_path.parent}\"",
        ],
        "prompt_write_command": (
            "python -m gemmanima.cli text-preservation-prompts "
            f"--output \"{prompt_path}\" --count {sample_count} --json"
            + (" --include-eval-cases" if include_eval_cases else "")
            + (f" --prompt-index-offset {prompt_index_offset}" if prompt_index_offset else "")
            + (f" --src-prefix {src_prefix}" if src_prefix != "text_preserve_blend" else "")
            + (" --no-sample-marker" if not include_sample_marker else "")
        ),
        "text_target_cache_command": build_cache_targets_command(
            subset_path=prompt_path,
            outdir=text_target_dir,
            shard=1000,
            gpu_index=gpu_index,
        ),
        "text_gemma_cache_command": gemma_plan.command(
            subset=prompt_path,
            target_dir=text_target_dir,
            outdir=text_gemma_dir,
        ),
        "blend_link_commands": _blend_link_commands(
            text_target_dir=text_target_dir,
            text_gemma_dir=text_gemma_dir,
            blend_target_dir=blend_target_dir,
            blend_gemma_dir=blend_gemma_dir,
            source_general_target_dir=Path(source_general_target_dir),
            source_general_gemma_dir=Path(source_general_gemma_dir),
            text_repeat=text_repeat,
            text_shards=text_shards,
            general_shards=general_shards,
        ),
        "train_command": _bridge_train_command(
            target_dir=blend_target_dir,
            gemma_dir=blend_gemma_dir,
            output=output_path,
            resume_kv=resume_path,
            gpu_index=gpu_index,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            accum=accum,
            val=val,
            prefetch_gb=prefetch_gb,
        ),
        "status_command": (
            "python -m gemmanima.cli text-preservation-blended-status --json"
        ),
        "post_train_qwen_eval_plan_command": (
            "python -m gemmanima.cli text-rendering-qwen-baseline-plan "
            f"--student-checkpoint \"{output_path}\" "
            "--student-name gemma_text_preservation_blended --json"
        ),
    }


def build_text_preservation_v5_plan(
    *,
    sample_count: int = 1024,
    text_repeat: int = 6,
    general_shards: int = 4,
    epochs: int = 2,
    lr: float = 3e-5,
    batch_size: int = 2,
    accum: int = 1,
    val: int = 64,
    prefetch_gb: float = 4.0,
    gpu_index: int = 0,
) -> dict[str, Any]:
    plan = build_text_preservation_blended_plan(
        prompt_file=DEFAULT_TEXT_PRESERVATION_BLEND_V5_PROMPTS,
        root=DEFAULT_TEXT_PRESERVATION_BLEND_V5_ROOT,
        output=DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE,
        resume_kv=DEFAULT_TEXT_PRESERVATION_V4_BRIDGE,
        sample_count=sample_count,
        include_eval_cases=True,
        prompt_index_offset=20000,
        src_prefix="text_preserve_v5",
        include_sample_marker=False,
        text_repeat=text_repeat,
        general_shards=general_shards,
        gpu_index=gpu_index,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        accum=accum,
        val=val,
        prefetch_gb=prefetch_gb,
    )
    plan["stage"] = "text_preservation_blended_v5_candidate"
    plan["training_strategy"]["method"] = "heldout_informed_no_sample_marker_blend"
    plan["training_strategy"]["intent"] = (
        "expand text coverage while preventing non-target sample identifiers from being rendered"
    )
    plan["post_train_heldout_eval_plan_command"] = (
        "python -m gemmanima.cli text-preservation-heldout-eval-plan "
        f"--student-checkpoint \"{DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE}\" "
        "--student-name gemma_text_preservation_blended_v5 --json"
    )
    return plan


def build_text_preservation_v6_hard_negative_plan(
    *,
    sample_count: int = 320,
    text_repeat: int = 10,
    general_shards: int = 4,
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 2,
    accum: int = 1,
    val: int = 64,
    prefetch_gb: float = 4.0,
    gpu_index: int = 0,
) -> dict[str, Any]:
    plan = build_text_preservation_blended_plan(
        prompt_file=DEFAULT_TEXT_PRESERVATION_BLEND_V6_PROMPTS,
        root=DEFAULT_TEXT_PRESERVATION_BLEND_V6_ROOT,
        output=DEFAULT_TEXT_PRESERVATION_BLEND_V6_BRIDGE,
        resume_kv=DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE,
        sample_count=sample_count,
        include_eval_cases=True,
        prompt_index_offset=40000,
        src_prefix="text_preserve_v6_hard",
        include_sample_marker=False,
        text_repeat=text_repeat,
        general_shards=general_shards,
        gpu_index=gpu_index,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        accum=accum,
        val=val,
        prefetch_gb=prefetch_gb,
    )
    plan["stage"] = "text_preservation_blended_v6_hard_negative_candidate"
    plan["training_strategy"]["method"] = "hard_negative_no_sample_marker_blend"
    plan["training_strategy"]["intent"] = (
        "target held-out failure modes while preserving v5 fixed-case text gains"
    )
    plan["prompt_write_command"] = (
        "python -m gemmanima.cli text-preservation-v6-prompts "
        f"--output \"{DEFAULT_TEXT_PRESERVATION_BLEND_V6_PROMPTS}\" "
        f"--count {sample_count} --json"
    )
    plan["post_train_heldout_eval_plan_command"] = (
        "python -m gemmanima.cli text-preservation-heldout-eval-plan "
        f"--student-checkpoint \"{DEFAULT_TEXT_PRESERVATION_BLEND_V6_BRIDGE}\" "
        "--student-name gemma_text_preservation_blended_v6 "
        "--prompt-file \"reports\\text_preservation_heldout_v5_clean\\prompts.jsonl\" "
        "--out-root \"runs\\images\\text_preservation_heldout_v6_clean\" "
        "--report-root \"reports\\text_preservation_heldout_v6_clean\" "
        "--prompt-index-offset 30000 --src-prefix text_preserve_heldout_clean "
        "--no-sample-marker --json"
    )
    return plan


def build_text_preservation_v7_balanced_plan(
    *,
    root: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V7_ROOT,
    output: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V7_BRIDGE,
    v5_text_target_dir: str | Path = r"runs\cache\text_preservation_blended_v5\text_targets",
    v5_text_gemma_dir: str | Path = r"runs\cache\text_preservation_blended_v5\text_gemma",
    hard_target_dir: str | Path = r"runs\cache\text_preservation_blended_v6\text_targets",
    hard_gemma_dir: str | Path = r"runs\cache\text_preservation_blended_v6\text_gemma",
    source_general_target_dir: str | Path = r"runs\cache\poc1_10k\targets",
    source_general_gemma_dir: str | Path = r"runs\cache\poc1_10k\gemma",
    resume_kv: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE,
    v5_text_repeats: int = 4,
    hard_negative_repeats: int = 1,
    general_shards: int = 10,
    gpu_index: int = 0,
    epochs: int = 2,
    lr: float = 1e-5,
    batch_size: int = 2,
    accum: int = 1,
    val: int = 64,
    prefetch_gb: float = 4.0,
) -> dict[str, Any]:
    if gpu_index != 0:
        raise ValueError("text preservation v7 balanced plan is 4070-only; use CUDA device 0")
    root_path = Path(root)
    output_path = Path(output)
    resume_path = Path(resume_kv)
    blend_target_dir = root_path / "blend_targets"
    blend_gemma_dir = root_path / "blend_gemma"
    return {
        "stage": "text_preservation_blended_v7_balanced_candidate",
        "mode": "executable",
        "executes_gpu_commands": True,
        "root": _json_path(root_path),
        "blend_target_dir": _json_path(blend_target_dir),
        "blend_gemma_dir": _json_path(blend_gemma_dir),
        "output": _json_path(output_path),
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "blend": {
            "v5_text_repeats": v5_text_repeats,
            "hard_negative_repeats": hard_negative_repeats,
            "general_shards": general_shards,
            "link_strategy": "hardlink",
            "v5_text_target_dir": _json_path(Path(v5_text_target_dir)),
            "v5_text_gemma_dir": _json_path(Path(v5_text_gemma_dir)),
            "hard_negative_target_dir": _json_path(Path(hard_target_dir)),
            "hard_negative_gemma_dir": _json_path(Path(hard_gemma_dir)),
            "general_source_target_dir": _json_path(Path(source_general_target_dir)),
            "general_source_gemma_dir": _json_path(Path(source_general_gemma_dir)),
        },
        "training_strategy": {
            "method": "balanced_v5_replay_hard_negative_general_replay",
            "resume_from": _json_path(resume_path),
            "objective": "08_train_stream_batched.py MSE bridge objective",
            "intent": "recover hard text cases without damaging v5 fixed text or broad general replay",
        },
        "setup_commands": [
            f"New-Item -ItemType Directory -Force -Path \"{blend_target_dir}\"",
            f"New-Item -ItemType Directory -Force -Path \"{blend_gemma_dir}\"",
            f"New-Item -ItemType Directory -Force -Path \"{output_path.parent}\"",
        ],
        "blend_link_commands": _balanced_v7_link_commands(
            v5_text_target_dir=Path(v5_text_target_dir),
            v5_text_gemma_dir=Path(v5_text_gemma_dir),
            hard_target_dir=Path(hard_target_dir),
            hard_gemma_dir=Path(hard_gemma_dir),
            source_general_target_dir=Path(source_general_target_dir),
            source_general_gemma_dir=Path(source_general_gemma_dir),
            blend_target_dir=blend_target_dir,
            blend_gemma_dir=blend_gemma_dir,
            v5_text_repeats=v5_text_repeats,
            hard_negative_repeats=hard_negative_repeats,
            general_shards=general_shards,
        ),
        "train_command": _bridge_train_command(
            target_dir=blend_target_dir,
            gemma_dir=blend_gemma_dir,
            output=output_path,
            resume_kv=resume_path,
            gpu_index=gpu_index,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            accum=accum,
            val=val,
            prefetch_gb=prefetch_gb,
        ),
        "eval_command": f"python -m gemmanima.cli bridge-eval-status --checkpoint \"{output_path}\" --json",
        "post_train_qwen_eval_plan_command": (
            "python -m gemmanima.cli text-rendering-qwen-baseline-plan "
            f"--student-checkpoint \"{output_path}\" "
            "--student-name gemma_text_preservation_blended_v7 --json"
        ),
    }


def build_text_preservation_v8_fixed_gate_plan(
    *,
    root: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V8_ROOT,
    output: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V8_BRIDGE,
    fixed_target_dir: str | Path = DEFAULT_TEXT_PRESERVATION_TARGET_DIR,
    fixed_gemma_dir: str | Path = DEFAULT_TEXT_PRESERVATION_GEMMA_DIR,
    v5_text_target_dir: str | Path = r"runs\cache\text_preservation_blended_v5\text_targets",
    v5_text_gemma_dir: str | Path = r"runs\cache\text_preservation_blended_v5\text_gemma",
    hard_target_dir: str | Path = r"runs\cache\text_preservation_blended_v6\text_targets",
    hard_gemma_dir: str | Path = r"runs\cache\text_preservation_blended_v6\text_gemma",
    source_general_target_dir: str | Path = r"runs\cache\poc1_10k\targets",
    source_general_gemma_dir: str | Path = r"runs\cache\poc1_10k\gemma",
    resume_kv: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE,
    fixed_gate_repeats: int = 8,
    v5_text_repeats: int = 2,
    hard_negative_repeats: int = 1,
    general_shards: int = 4,
    gpu_index: int = 0,
    epochs: int = 1,
    lr: float = 5e-6,
    batch_size: int = 2,
    accum: int = 1,
    val: int = 64,
    prefetch_gb: float = 4.0,
) -> dict[str, Any]:
    if gpu_index != 0:
        raise ValueError("text preservation v8 fixed-gate plan is 4070-only; use CUDA device 0")
    root_path = Path(root)
    output_path = Path(output)
    resume_path = Path(resume_kv)
    blend_target_dir = root_path / "blend_targets"
    blend_gemma_dir = root_path / "blend_gemma"
    return {
        "stage": "text_preservation_blended_v8_fixed_gate_candidate",
        "mode": "executable",
        "executes_gpu_commands": True,
        "root": _json_path(root_path),
        "blend_target_dir": _json_path(blend_target_dir),
        "blend_gemma_dir": _json_path(blend_gemma_dir),
        "output": _json_path(output_path),
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "blend": {
            "fixed_gate_repeats": fixed_gate_repeats,
            "v5_text_repeats": v5_text_repeats,
            "hard_negative_repeats": hard_negative_repeats,
            "general_shards": general_shards,
            "link_strategy": "hardlink",
            "fixed_gate_target_dir": _json_path(Path(fixed_target_dir)),
            "fixed_gate_gemma_dir": _json_path(Path(fixed_gemma_dir)),
            "v5_text_target_dir": _json_path(Path(v5_text_target_dir)),
            "v5_text_gemma_dir": _json_path(Path(v5_text_gemma_dir)),
            "hard_negative_target_dir": _json_path(Path(hard_target_dir)),
            "hard_negative_gemma_dir": _json_path(Path(hard_gemma_dir)),
            "general_source_target_dir": _json_path(Path(source_general_target_dir)),
            "general_source_gemma_dir": _json_path(Path(source_general_gemma_dir)),
        },
        "training_strategy": {
            "method": "fixed_gate_preserving_conservative_replay",
            "resume_from": _json_path(resume_path),
            "objective": "08_train_stream_batched.py MSE bridge objective",
            "intent": (
                "protect the fixed six text-rendering gate while cautiously replaying "
                "v5 broad text, low-weight hard negatives, and general scene shards"
            ),
        },
        "setup_commands": [
            f"New-Item -ItemType Directory -Force -Path \"{blend_target_dir}\"",
            f"New-Item -ItemType Directory -Force -Path \"{blend_gemma_dir}\"",
            f"New-Item -ItemType Directory -Force -Path \"{output_path.parent}\"",
        ],
        "blend_link_commands": _fixed_gate_v8_link_commands(
            fixed_target_dir=Path(fixed_target_dir),
            fixed_gemma_dir=Path(fixed_gemma_dir),
            v5_text_target_dir=Path(v5_text_target_dir),
            v5_text_gemma_dir=Path(v5_text_gemma_dir),
            hard_target_dir=Path(hard_target_dir),
            hard_gemma_dir=Path(hard_gemma_dir),
            source_general_target_dir=Path(source_general_target_dir),
            source_general_gemma_dir=Path(source_general_gemma_dir),
            blend_target_dir=blend_target_dir,
            blend_gemma_dir=blend_gemma_dir,
            fixed_gate_repeats=fixed_gate_repeats,
            v5_text_repeats=v5_text_repeats,
            hard_negative_repeats=hard_negative_repeats,
            general_shards=general_shards,
        ),
        "train_command": _bridge_train_command(
            target_dir=blend_target_dir,
            gemma_dir=blend_gemma_dir,
            output=output_path,
            resume_kv=resume_path,
            gpu_index=gpu_index,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            accum=accum,
            val=val,
            prefetch_gb=prefetch_gb,
        ),
        "eval_command": f"python -m gemmanima.cli bridge-eval-status --checkpoint \"{output_path}\" --json",
        "post_train_qwen_eval_plan_command": (
            "python -m gemmanima.cli text-rendering-qwen-baseline-plan "
            f"--student-checkpoint \"{output_path}\" "
            "--student-name gemma_text_preservation_blended_v8 --json"
        ),
    }


def build_text_preservation_promotion_status(
    *,
    fixed_report_root: str | Path = r"reports\text_rendering_qwen_baseline",
    candidates: dict[str, dict[str, Any]] | None = None,
    baseline: str = "v5",
) -> dict[str, Any]:
    candidate_specs = candidates or {
        "v5": {
            "student_name": "gemma_text_preservation_blended_v5",
            "heldout_metrics": r"reports\text_preservation_heldout_v5_clean\metrics_summary.json",
            "heldout_review": r"reports\text_preservation_heldout_v5_clean\visual_review.json",
            "general_metrics": r"reports\general_scene_regression_v5_50\metrics_summary.json",
        },
        "v6": {
            "student_name": "gemma_text_preservation_blended_v6",
            "heldout_metrics": r"reports\text_preservation_heldout_v6_clean\metrics_summary.json",
        },
        "v7": {
            "student_name": "gemma_text_preservation_blended_v7",
        },
        "v8": {
            "student_name": "gemma_text_preservation_blended_v8",
            "heldout_metrics": r"reports\text_preservation_heldout_v8_clean\metrics_summary.json",
        },
        "v9": {
            "student_name": "gemma_text_preservation_blended_v9",
        },
        "v10": {
            "student_name": "gemma_text_preservation_blended_v10",
        },
        "v11": {
            "student_name": "gemma_text_preservation_blended_v11",
        },
        "v12": {
            "student_name": "gemma_text_preservation_blended_v12",
        },
        "v13": {
            "student_name": "gemma_text_preservation_blended_v13",
        },
        "v14": {
            "student_name": "gemma_text_preservation_blended_v14",
        },
        "v15": {
            "student_name": "gemma_text_preservation_blended_v15",
        },
        "v16": {
            "student_name": "gemma_text_preservation_blended_v16",
        },
        "v17": {
            "student_name": "gemma_text_preservation_blended_v17",
        },
        "v18": {
            "student_name": "gemma_text_preservation_blended_v18",
        },
        "v19": {
            "student_name": "gemma_text_preservation_blended_v19",
        },
        "v22_alpha28": {
            "student_name": "gemma_text_preservation_blended_v22_alpha28",
            "heldout_metrics": r"reports\text_preservation_heldout_v22_alpha28_clean\metrics_summary.json",
            "heldout_review": r"reports\text_preservation_heldout_v22_alpha28_clean\visual_review.json",
            "general_metrics": r"reports\general_scene_regression_v22_alpha28_50\metrics_summary.json",
        },
        "v23": {
            "student_name": "gemma_text_preservation_blended_v23",
            "heldout_metrics": r"reports\text_preservation_heldout_v23_clean\metrics_summary.json",
            "heldout_review": r"reports\text_preservation_heldout_v23_clean\visual_review.json",
            "general_metrics": r"reports\general_scene_regression_v23_50\metrics_summary.json",
            "general_review": r"reports\general_scene_regression_v23_50\visual_review.json",
        },
    }
    fixed_root = Path(fixed_report_root)
    payload: dict[str, Any] = {
        "stage": "text_preservation_promotion_status",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "baseline": baseline,
        "fixed_report_root": _json_path(fixed_root),
        "candidates": {},
    }
    for version, spec in candidate_specs.items():
        student_name = str(spec["student_name"])
        fixed_reports = sorted(fixed_root.glob(f"*_qwen_vs_{student_name}_compare.json"))
        fixed_gate = _summarize_compare_reports(fixed_reports)
        candidate: dict[str, Any] = {
            "student_name": student_name,
            "fixed_gate": fixed_gate,
            "heldout": _read_optional_json(spec.get("heldout_metrics")),
            "heldout_review": _read_optional_json(spec.get("heldout_review")),
            "general_scene": _read_optional_json(spec.get("general_metrics")),
            "required_artifacts": _promotion_required_artifacts(
                fixed_report_root=fixed_root,
                student_name=student_name,
                fixed_report_count=len(fixed_reports),
                expected_fixed_count=None,
                spec=spec,
            ),
        }
        payload["candidates"][version] = candidate

    baseline_gate = payload["candidates"].get(baseline, {}).get("fixed_gate", {})
    baseline_mean = baseline_gate.get("mean_mse")
    baseline_count = baseline_gate.get("count", 0)
    baseline_case_mse = _case_mse_map(baseline_gate)
    for candidate in payload["candidates"].values():
        candidate["required_artifacts"][0]["expected_count"] = baseline_count
    for version, candidate in payload["candidates"].items():
        fixed_gate = candidate["fixed_gate"]
        mean_mse = fixed_gate.get("mean_mse")
        failure_reasons = _fixed_gate_failure_reasons(
            fixed_gate=fixed_gate,
            baseline_gate=baseline_gate,
            baseline_case_mse=baseline_case_mse,
        )
        if version == baseline:
            decision = {"status": "current_baseline", "reason": "baseline candidate for fixed-gate comparison"}
        elif mean_mse is None:
            decision = {"status": "pending", "reason": "fixed gate reports are missing"}
        elif baseline_mean is None:
            decision = {"status": "pending", "reason": "baseline fixed gate reports are missing"}
        elif failure_reasons:
            decision = {
                "status": "reject",
                "reason": _fixed_gate_reject_reason(mean_mse=mean_mse, baseline_mean=baseline_mean),
            }
        else:
            decision = {
                "status": "eligible_for_next_gate",
                "reason": f"fixed gate did not regress: mean MSE {mean_mse:.12f} <= baseline {baseline_mean:.12f}",
            }
        candidate["failure_reasons"] = [] if version == baseline else failure_reasons
        candidate["decision"] = decision
    payload["recommendation"] = _build_promotion_recommendation(
        candidates=payload["candidates"],
        baseline=baseline,
    )
    return payload


def write_text_preservation_promotion_status(
    output: str | Path,
    *,
    fixed_report_root: str | Path = r"reports\text_rendering_qwen_baseline",
    candidates: dict[str, dict[str, Any]] | None = None,
    baseline: str = "v5",
) -> dict[str, Any]:
    payload = build_text_preservation_promotion_status(
        fixed_report_root=fixed_report_root,
        candidates=candidates,
        baseline=baseline,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = str(output_path)
    return payload


def build_text_preservation_compact_promotion_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "text_preservation_promotion_status_compact",
        "recommendation": status["recommendation"],
        "candidates": [
            {
                "version": version,
                "student_name": candidate["student_name"],
                "decision": candidate["decision"]["status"],
                "fixed_gate_mean_mse": candidate["fixed_gate"]["mean_mse"],
                "fixed_gate_max_mse": candidate["fixed_gate"].get("max_mse"),
                "failure_reasons": candidate.get("failure_reasons", []),
            }
            for version, candidate in status["candidates"].items()
        ],
    }


def build_text_preservation_release_gate_status(
    *,
    fixed_report_root: str | Path = r"reports\text_rendering_qwen_baseline",
    candidate_specs: dict[str, dict[str, Any]] | None = None,
    baseline: str = "v5",
    min_heldout_readable: int = 47,
    max_heldout_failed: int = 4,
    min_general_cases: int = 50,
) -> dict[str, Any]:
    specs = candidate_specs or {
        "v5": {
            "student_name": "gemma_text_preservation_blended_v5",
            "heldout_metrics": r"reports\text_preservation_heldout_v5_clean\metrics_summary.json",
            "heldout_review": r"reports\text_preservation_heldout_v5_clean\visual_review.json",
            "general_metrics": r"reports\general_scene_regression_v5_50\metrics_summary.json",
            "general_review": r"reports\general_scene_regression_v5_50\visual_review.json",
        },
        "v6": {
            "student_name": "gemma_text_preservation_blended_v6",
            "heldout_metrics": r"reports\text_preservation_heldout_v6_clean\metrics_summary.json",
        },
        "v7": {"student_name": "gemma_text_preservation_blended_v7"},
        "v8": {"student_name": "gemma_text_preservation_blended_v8"},
        "v9": {"student_name": "gemma_text_preservation_blended_v9"},
        "v10": {"student_name": "gemma_text_preservation_blended_v10"},
        "v11": {"student_name": "gemma_text_preservation_blended_v11"},
        "v12": {"student_name": "gemma_text_preservation_blended_v12"},
        "v13": {"student_name": "gemma_text_preservation_blended_v13"},
        "v14": {"student_name": "gemma_text_preservation_blended_v14"},
        "v15": {"student_name": "gemma_text_preservation_blended_v15"},
        "v16": {"student_name": "gemma_text_preservation_blended_v16"},
        "v17": {"student_name": "gemma_text_preservation_blended_v17"},
        "v18": {"student_name": "gemma_text_preservation_blended_v18"},
        "v19": {"student_name": "gemma_text_preservation_blended_v19"},
        "v22_alpha28": {
            "student_name": "gemma_text_preservation_blended_v22_alpha28",
            "heldout_metrics": r"reports\text_preservation_heldout_v22_alpha28_clean\metrics_summary.json",
            "heldout_review": r"reports\text_preservation_heldout_v22_alpha28_clean\visual_review.json",
            "general_metrics": r"reports\general_scene_regression_v22_alpha28_50\metrics_summary.json",
            "general_review": r"reports\general_scene_regression_v22_alpha28_50\visual_review.json",
        },
        "v23": {
            "student_name": "gemma_text_preservation_blended_v23",
            "heldout_metrics": r"reports\text_preservation_heldout_v23_clean\metrics_summary.json",
            "heldout_review": r"reports\text_preservation_heldout_v23_clean\visual_review.json",
            "general_metrics": r"reports\general_scene_regression_v23_50\metrics_summary.json",
            "general_review": r"reports\general_scene_regression_v23_50\visual_review.json",
        },
    }
    promotion = build_text_preservation_promotion_status(
        fixed_report_root=fixed_report_root,
        candidates=specs,
        baseline=baseline,
    )
    baseline_candidate = promotion["candidates"].get(baseline, {})
    failure_reasons = _release_gate_failure_reasons(
        promotion=promotion,
        baseline_candidate=baseline_candidate,
        min_heldout_readable=min_heldout_readable,
        max_heldout_failed=max_heldout_failed,
        min_general_cases=min_general_cases,
    )
    release_status = "pass" if not failure_reasons else "fail"
    return {
        "stage": "text_preservation_release_gate",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "protected_baseline": baseline,
        "promotion": promotion,
        "release_gate": {
            "status": release_status,
            "failure_reasons": failure_reasons,
            "thresholds": {
                "min_heldout_readable": min_heldout_readable,
                "max_heldout_failed": max_heldout_failed,
                "min_general_cases": min_general_cases,
            },
        },
        "v9_training_gate": _v9_training_gate_decision(
            release_status=release_status,
            promotion=promotion,
        ),
    }


def write_text_preservation_release_gate_status(
    output: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_release_gate_status(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = str(output_path)
    return payload


def build_text_preservation_v9_objective_plan(
    *,
    release_gate: dict[str, Any] | None = None,
    baseline: str = "v5",
) -> dict[str, Any]:
    gate = release_gate if release_gate is not None else build_text_preservation_release_gate_status(baseline=baseline)
    v9_gate = gate.get("v9_training_gate", {})
    release_status = gate.get("release_gate", {}).get("status")
    protected_baseline = gate.get("protected_baseline", baseline)
    return {
        "stage": "text_preservation_v9_objective_plan",
        "mode": "design_contract",
        "executes_gpu_commands": False,
        "protected_baseline": protected_baseline,
        "release_gate_status": release_status,
        "v9_training_gate": v9_gate,
        "training_plan": {
            "status": "blocked",
            "train_command": None,
            "reason": "objective redesign required before GPU training",
        },
        "objective_redesign": {
            "status": "blocked_until_artifact_gate_first_objective_redesign",
            "recommended_approach": "artifact_gate_first",
            "hard_preconditions": [
                "fixed6_per_case_protection",
                "heldout_readability_floor",
                "general_scene_no_collapse_review",
                "promotion_decision_separate_from_bridge_mse",
            ],
            "required_changes": [
                "add image/text-level gate feedback before expanding replay training",
                "keep fixed6 per-case protection as a hard precondition",
                "separate bridge validation MSE from promotion decisions",
                "require held-out and general-scene evaluation before any candidate promotion",
            ],
            "candidate_approaches": [
                {
                    "id": "artifact_gate_first",
                    "status": "recommended",
                    "description": (
                        "mine fixed, held-out, and general-scene artifacts into explicit pass/fail gates "
                        "before another bridge train run"
                    ),
                },
                {
                    "id": "offline_text_error_mining",
                    "status": "candidate",
                    "description": (
                        "classify existing Qwen/Gemma image pairs by visible text quality and build a "
                        "hard-negative curriculum from real failures"
                    ),
                },
                {
                    "id": "renderer_feedback_loop",
                    "status": "candidate",
                    "description": (
                        "add post-render feedback labels so training targets reflect readable text outcomes "
                        "instead of only hidden-state distance"
                    ),
                },
            ],
        },
        "next_safe_actions": [
            {
                "action": "implement_artifact_gate_first_objective",
                "type": "code",
                "gpu_required": False,
            },
            {
                "action": "build_v9_candidate_plan_only_after_objective_tests_pass",
                "type": "planning",
                "gpu_required": False,
            },
            {
                "action": "run_v9_training_only_after_objective_plan_emits_explicit_train_permission",
                "type": "training",
                "gpu_required": True,
                "allowed_device": "RTX 4070 Ti SUPER only",
            },
        ],
    }


def write_text_preservation_v9_objective_plan(
    output: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v9_objective_plan(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = str(output_path)
    return payload


def build_text_preservation_v9_artifact_gate_objective(
    *,
    objective_plan: dict[str, Any] | None = None,
    baseline: str = "v5",
) -> dict[str, Any]:
    plan = objective_plan if objective_plan is not None else build_text_preservation_v9_objective_plan(baseline=baseline)
    protected_baseline = plan.get("protected_baseline", baseline)
    return {
        "stage": "text_preservation_v9_artifact_gate_objective",
        "mode": "objective_contract",
        "executes_gpu_commands": False,
        "protected_baseline": protected_baseline,
        "source_objective_plan_status": plan.get("objective_redesign", {}).get("status"),
        "objective": {
            "method": "artifact_gate_first",
            "teacher": "Qwen text-rendering baseline",
            "student": "Gemma bridge candidate",
            "baseline": protected_baseline,
            "training_signal": (
                "combine bridge hidden-state regression with post-render artifact feedback before any "
                "large replay-weighted expansion"
            ),
            "artifact_gates": [
                {
                    "id": "fixed6_per_case_image_mse",
                    "required": True,
                    "source": "reports/text_rendering_qwen_baseline/*_compare.json",
                    "rule": "candidate_mse_must_not_exceed_v5_per_case",
                },
                {
                    "id": "fixed6_mean_and_max_image_mse",
                    "required": True,
                    "source": "reports/text_rendering_qwen_baseline/metrics_summary_*.json",
                    "rule": "candidate_mean_and_max_mse_must_not_exceed_v5",
                },
                {
                    "id": "heldout_text_readability_review",
                    "required": True,
                    "source": "reports/text_preservation_heldout_*_clean/visual_review.json",
                    "rule": "readable_count_at_or_above_47_and_failed_count_at_or_below_4",
                },
                {
                    "id": "general_scene_no_collapse_review",
                    "required": True,
                    "source": "reports/general_scene_regression_*_50/visual_review.json",
                    "rule": "no_blank_or_text_only_collapse_across_50_cases",
                },
            ],
        },
        "candidate_plan_contract": {
            "required_before_train_command": [
                "trainer_artifact_feedback_support",
                "artifact_gate_loss_config",
                "fixed6_baseline_case_map",
                "heldout_and_general_eval_plan",
            ],
            "forbidden_until_ready": [
                "replay_weight_only_training",
                "bridge_val_mse_only_promotion",
                "gpu_train_command_without_artifact_feedback",
            ],
        },
        "candidate_planning_permission": {
            "status": "allowed",
            "reason": "objective gates are explicit enough to design the next candidate plan",
        },
        "gpu_training_permission": {
            "status": "blocked_until_trainer_supports_artifact_feedback",
            "train_command": None,
        },
    }


def write_text_preservation_v9_artifact_gate_objective(
    output: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v9_artifact_gate_objective(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = str(output_path)
    return payload


def build_text_preservation_v9_trainer_support_audit(
    *,
    train_script: str | Path = DEFAULT_TRAIN_SCRIPT,
) -> dict[str, Any]:
    script_path = Path(train_script)
    source = script_path.read_text(encoding="utf-8", errors="ignore") if script_path.exists() else ""
    feature_tokens = {
        "artifact_feedback_dataset": ["--artifact-feedback", "artifact_feedback_dataset"],
        "artifact_gate_loss_config": ["--artifact-gate-loss", "artifact_gate_loss_config"],
        "post_render_metric_ingest": ["post_render_metric", "image_mse_feedback", "readability_feedback"],
        "kv_anchor_regularization": ["--kv-anchor", "kv_anchor_regularization"],
        "source_bucket_feedback_filter": ["source_buckets", "source_bucket_from_shard"],
    }
    present_features = [
        feature_id
        for feature_id, tokens in feature_tokens.items()
        if any(token in source for token in tokens)
    ]
    missing_features = [
        feature_id
        for feature_id in feature_tokens
        if feature_id not in present_features
    ]
    status = "supported" if not missing_features else "missing_artifact_feedback_support"
    training_permission = (
        {
            "status": "allowed_to_build_candidate_train_command",
            "train_command": "defer_to_v9_candidate_plan",
        }
        if status == "supported"
        else {
            "status": "blocked_until_trainer_supports_artifact_feedback",
            "train_command": None,
        }
    )
    return {
        "stage": "text_preservation_v9_trainer_support_audit",
        "mode": "artifact_feedback_support_audit",
        "executes_gpu_commands": False,
        "train_script": _json_path(script_path),
        "trainer_support": {
            "status": status,
            "script_exists": script_path.exists(),
            "present_features": present_features,
            "missing_features": missing_features,
            "required_features": list(feature_tokens),
        },
        "gpu_training_permission": training_permission,
    }


def write_text_preservation_v9_trainer_support_audit(
    output: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v9_trainer_support_audit(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = str(output_path)
    return payload


def build_text_preservation_v21_text_roi_gate_report(
    *,
    baseline_reports: list[str | Path] | None = None,
    candidate_specs: dict[str, dict[str, Any]] | None = None,
    roi_map: dict[str, dict[str, Any]] | None = None,
    protected_baseline: str = "v5",
) -> dict[str, Any]:
    baseline_paths = [Path(path) for path in (baseline_reports or [])]
    roi_by_case = roi_map or _default_fixed6_text_roi_map()
    baseline_cases = {
        _compare_case_id(path): _text_roi_case_record(path, roi_by_case.get(_compare_case_id(path)))
        for path in baseline_paths
    }
    candidate_payloads: dict[str, Any] = {}
    for name, spec in (candidate_specs or {}).items():
        case_payloads: dict[str, Any] = {}
        failure_reasons: list[str] = []
        for report_path in [Path(path) for path in spec.get("reports", [])]:
            case_id = _compare_case_id(report_path)
            case_payload = _text_roi_case_record(report_path, roi_by_case.get(case_id))
            case_payloads[case_id] = case_payload
            baseline_roi = baseline_cases.get(case_id, {}).get("text_roi_metrics") or {}
            candidate_roi = case_payload.get("text_roi_metrics") or {}
            baseline_mse = baseline_roi.get("mse")
            candidate_mse = candidate_roi.get("mse")
            if candidate_mse is not None and baseline_mse is not None and candidate_mse > baseline_mse:
                failure_reasons.append(f"text_roi_case_regression:{case_id}")
        candidate_payloads[name] = {
            "student_name": spec.get("student_name", name),
            "cases": case_payloads,
            "failure_reasons": failure_reasons,
            "decision": "reject_observer_only" if failure_reasons else "roi_non_regressing_observer_only",
        }
    return {
        "stage": "text_preservation_v21_text_roi_gate_report",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "promotion_effect": "none_observer_only",
        "protected_baseline": protected_baseline,
        "workflow_position": {
            "previous_step": "v20_v5_v17_interpolation_sweep_evaluated",
            "current_step": "build_v21_text_roi_gate_report",
            "next_step": "use_roi_report_to_design_v21_multi_objective_search_without_relaxing_v5_gate",
        },
        "roi_policy": {
            "purpose": "diagnose whether whole-image MSE is being pulled by background/layout drift",
            "promotion_rule": "does_not_replace_fixed6_whole_image_gate",
            "baseline_policy": "missing_roi_or_missing_images_are_reported_without promoting candidates",
        },
        "baseline": baseline_cases,
        "candidates": candidate_payloads,
        "recommendation": {
            "status": "protect_baseline",
            "protected_baseline": protected_baseline,
            "promote_candidate": None,
            "reason": "text ROI metrics are observer-only diagnostics and do not relax fixed6 per-case gates",
        },
    }


def write_text_preservation_v21_text_roi_gate_report(
    output: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v21_text_roi_gate_report(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_v9_artifact_feedback_dataset(
    *,
    fixed_report_root: str | Path = r"reports\text_rendering_qwen_baseline",
    student_name: str = "gemma_text_preservation_blended_v5",
) -> dict[str, Any]:
    report_root = Path(fixed_report_root)
    paths = sorted(report_root.glob(f"*_qwen_vs_{student_name}_compare.json"))
    fixed_gate = _summarize_compare_reports(paths)
    mean_mse = fixed_gate.get("mean_mse") or 1.0
    records = []
    for idx, row in enumerate(fixed_gate.get("reports", [])):
        normalized = row["mse"] / mean_mse
        weight = min(4.0, max(1.0, 1.0 + normalized))
        readability_feedback = "partial" if normalized >= 1.25 else "readable"
        records.append(
            {
                "idx": idx,
                "case_id": row["case_id"],
                "weight": weight,
                "image_mse_feedback": row["mse"],
                "readability_feedback": readability_feedback,
                "source_report": row["report"],
            }
        )
    return {
        "stage": "text_preservation_v9_artifact_feedback_dataset",
        "mode": "artifact_feedback_dataset",
        "executes_gpu_commands": False,
        "student_name": student_name,
        "fixed_report_root": _json_path(report_root),
        "records": records,
        "record_count": len(records),
    }


def write_text_preservation_v9_artifact_feedback_dataset(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V9_ARTIFACT_FEEDBACK,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v9_artifact_feedback_dataset(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in payload["records"]) + "\n",
        encoding="utf-8",
    )
    payload["output"] = str(output_path)
    return payload


def build_text_preservation_artifact_feedback_alignment_audit(
    *,
    blend_target_dir: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V10_ROOT / "blend_targets",
    artifact_feedback: str | Path = DEFAULT_TEXT_PRESERVATION_V10_ARTIFACT_FEEDBACK,
) -> dict[str, Any]:
    import torch

    target_dir = Path(blend_target_dir)
    feedback_path = Path(artifact_feedback)
    feedback_records = _read_feedback_jsonl(feedback_path)
    feedback_by_idx = {int(record["idx"]): record for record in feedback_records}
    feedback_ids = set(feedback_by_idx)
    source_buckets: dict[str, dict[str, Any]] = {}
    occurrences_by_idx = {idx: 0 for idx in feedback_ids}
    weighted_occurrence_count = 0

    for shard_path in sorted(target_dir.glob("*.pt")):
        bucket = _feedback_source_bucket(shard_path)
        bucket_row = source_buckets.setdefault(
            bucket,
            {
                "shard_count": 0,
                "example_count": 0,
                "weighted_occurrences": 0,
            },
        )
        bucket_row["shard_count"] += 1
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        bucket_row["example_count"] += len(shard)
        for row in shard:
            idx = int(row["idx"])
            if idx in feedback_ids:
                occurrences_by_idx[idx] += 1
                bucket_row["weighted_occurrences"] += 1
                weighted_occurrence_count += 1

    missing_feedback_ids = [idx for idx, count in occurrences_by_idx.items() if count == 0]
    spillover_buckets = [
        bucket
        for bucket, row in source_buckets.items()
        if bucket != "00_fixed_gate" and row["weighted_occurrences"] > 0
    ]
    if missing_feedback_ids:
        decision = {
            "status": "fail_feedback_ids_missing_from_blend",
            "reason": "some artifact feedback ids do not occur in the blended target shards",
        }
    elif spillover_buckets:
        decision = {
            "status": "warn_feedback_spills_into_replay_sources",
            "reason": "artifact feedback ids also occur outside fixed-gate shards",
        }
    else:
        decision = {
            "status": "pass_fixed_gate_only_alignment",
            "reason": "artifact feedback ids occur only in fixed-gate shards",
        }

    return {
        "stage": "text_preservation_artifact_feedback_alignment_audit",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "blend_target_dir": _json_path(target_dir),
        "artifact_feedback": _json_path(feedback_path),
        "feedback_record_count": len(feedback_records),
        "weighted_occurrence_count": weighted_occurrence_count,
        "occurrences_by_idx": {str(idx): count for idx, count in sorted(occurrences_by_idx.items())},
        "source_buckets": source_buckets,
        "spillover_buckets": spillover_buckets,
        "missing_feedback_ids": missing_feedback_ids,
        "decision": decision,
    }


def write_text_preservation_artifact_feedback_alignment_audit(
    output: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_artifact_feedback_alignment_audit(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = str(output_path)
    return payload


def build_text_preservation_v9_artifact_gate_loss_config() -> dict[str, Any]:
    return {
        "stage": "text_preservation_v9_artifact_gate_loss_config",
        "executes_gpu_commands": False,
        "enabled": True,
        "min_weight": 0.25,
        "max_weight": 4.0,
        "weight_source": "post_render_artifact_feedback",
    }


def build_text_preservation_v11_artifact_gate_loss_config() -> dict[str, Any]:
    payload = build_text_preservation_v9_artifact_gate_loss_config()
    payload["stage"] = "text_preservation_v11_artifact_gate_loss_config"
    payload["source_buckets"] = ["00_fixed_gate"]
    payload["weight_scope"] = "fixed_gate_only"
    return payload


def write_text_preservation_v9_artifact_gate_loss_config(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V9_LOSS_CONFIG,
) -> dict[str, Any]:
    payload = build_text_preservation_v9_artifact_gate_loss_config()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = str(output_path)
    return payload


def write_text_preservation_v11_artifact_gate_loss_config(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V11_LOSS_CONFIG,
) -> dict[str, Any]:
    payload = build_text_preservation_v11_artifact_gate_loss_config()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = str(output_path)
    return payload


def build_text_preservation_v9_candidate_plan(
    *,
    root: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V9_ROOT,
    output: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V9_BRIDGE,
    artifact_feedback: str | Path = DEFAULT_TEXT_PRESERVATION_V9_ARTIFACT_FEEDBACK,
    artifact_gate_loss_config: str | Path = DEFAULT_TEXT_PRESERVATION_V9_LOSS_CONFIG,
    trainer_audit: dict[str, Any] | None = None,
    gpu_index: int = 0,
) -> dict[str, Any]:
    audit = trainer_audit if trainer_audit is not None else build_text_preservation_v9_trainer_support_audit()
    if audit.get("trainer_support", {}).get("status") != "supported":
        return {
            "stage": "text_preservation_blended_v9_artifact_gate_candidate",
            "mode": "blocked",
            "executes_gpu_commands": False,
            "reason": "trainer artifact feedback support is required before building a v9 train command",
            "trainer_audit": audit,
            "train_command": None,
        }
    plan = build_text_preservation_v8_fixed_gate_plan(root=root, output=output, gpu_index=gpu_index)
    plan["stage"] = "text_preservation_blended_v9_artifact_gate_candidate"
    plan["artifact_feedback"] = _json_path(Path(artifact_feedback))
    plan["artifact_gate_loss_config"] = _json_path(Path(artifact_gate_loss_config))
    plan["artifact_feedback_write_command"] = (
        "python -m gemmanima.cli text-preservation-v9-artifact-feedback "
        f"--output \"{Path(artifact_feedback)}\" --json"
    )
    plan["artifact_gate_loss_config_write_command"] = (
        "python -m gemmanima.cli text-preservation-v9-artifact-gate-loss-config "
        f"--output \"{Path(artifact_gate_loss_config)}\" --json"
    )
    plan["training_strategy"] = {
        "method": "artifact_gate_first_feedback_weighted_replay",
        "resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
        "objective": "08_train_stream_batched.py MSE bridge objective plus artifact feedback sample weights",
        "intent": "preserve fixed text-rendering gates by weighting replay cases from post-render artifact evidence",
    }
    plan["train_command"] = (
        f"{plan['train_command']} "
        f"--artifact-feedback \"{Path(artifact_feedback)}\" "
        f"--artifact-gate-loss-config \"{Path(artifact_gate_loss_config)}\""
    )
    plan["post_train_qwen_eval_plan_command"] = (
        "python -m gemmanima.cli text-rendering-qwen-baseline-plan "
        f"--student-checkpoint \"{Path(output)}\" "
        "--student-name gemma_text_preservation_blended_v9 --json"
    )
    return plan


def build_text_preservation_v10_candidate_plan(
    *,
    root: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V10_ROOT,
    output: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V10_BRIDGE,
    artifact_feedback: str | Path = DEFAULT_TEXT_PRESERVATION_V10_ARTIFACT_FEEDBACK,
    artifact_gate_loss_config: str | Path = DEFAULT_TEXT_PRESERVATION_V10_LOSS_CONFIG,
    anchor_checkpoint: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE,
    anchor_lambda: float = 0.1,
    trainer_audit: dict[str, Any] | None = None,
    gpu_index: int = 0,
) -> dict[str, Any]:
    audit = trainer_audit if trainer_audit is not None else build_text_preservation_v9_trainer_support_audit()
    present = set(audit.get("trainer_support", {}).get("present_features", []))
    if "kv_anchor_regularization" not in present:
        return {
            "stage": "text_preservation_blended_v10_protected_anchor_candidate",
            "mode": "blocked",
            "executes_gpu_commands": False,
            "reason": "trainer kv anchor regularization support is required before v10 training",
            "trainer_audit": audit,
            "train_command": None,
        }
    plan = build_text_preservation_v9_candidate_plan(
        root=root,
        output=output,
        artifact_feedback=artifact_feedback,
        artifact_gate_loss_config=artifact_gate_loss_config,
        trainer_audit=audit,
        gpu_index=gpu_index,
    )
    plan["stage"] = "text_preservation_blended_v10_protected_anchor_candidate"
    plan["artifact_feedback_write_command"] = (
        "python -m gemmanima.cli text-preservation-v9-artifact-feedback "
        f"--output \"{Path(artifact_feedback)}\" --json"
    )
    plan["artifact_gate_loss_config_write_command"] = (
        "python -m gemmanima.cli text-preservation-v9-artifact-gate-loss-config "
        f"--output \"{Path(artifact_gate_loss_config)}\" --json"
    )
    plan["anchor"] = {
        "checkpoint": _json_path(Path(anchor_checkpoint)),
        "lambda": anchor_lambda,
        "purpose": "keep the v10 bridge close to protected v5 while artifact feedback adjusts fixed text cases",
    }
    plan["training_strategy"] = {
        "method": "artifact_feedback_with_v5_kv_anchor",
        "resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
        "objective": "MSE bridge objective plus artifact feedback sample weights and v5 KV-anchor regularization",
        "intent": "reduce v9-style fixed6 image drift while preserving readable text",
    }
    plan["train_command"] = (
        f"{plan['train_command']} "
        f"--kv-anchor \"{Path(anchor_checkpoint)}\" "
        f"--kv-anchor-lambda {anchor_lambda}"
    )
    plan["post_train_qwen_eval_plan_command"] = (
        "python -m gemmanima.cli text-rendering-qwen-baseline-plan "
        f"--student-checkpoint \"{Path(output)}\" "
        "--student-name gemma_text_preservation_blended_v10 --json"
    )
    return plan


def build_text_preservation_v11_candidate_plan(
    *,
    root: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V11_ROOT,
    output: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V11_BRIDGE,
    artifact_feedback: str | Path = DEFAULT_TEXT_PRESERVATION_V11_ARTIFACT_FEEDBACK,
    artifact_gate_loss_config: str | Path = DEFAULT_TEXT_PRESERVATION_V11_LOSS_CONFIG,
    anchor_checkpoint: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE,
    anchor_lambda: float = 0.1,
    trainer_audit: dict[str, Any] | None = None,
    gpu_index: int = 0,
) -> dict[str, Any]:
    audit = trainer_audit if trainer_audit is not None else build_text_preservation_v9_trainer_support_audit()
    present = set(audit.get("trainer_support", {}).get("present_features", []))
    if "source_bucket_feedback_filter" not in present:
        return {
            "stage": "text_preservation_blended_v11_source_filtered_candidate",
            "mode": "blocked",
            "executes_gpu_commands": False,
            "reason": "trainer source-bucket feedback filter support is required before v11 training",
            "trainer_audit": audit,
            "train_command": None,
        }
    plan = build_text_preservation_v10_candidate_plan(
        root=root,
        output=output,
        artifact_feedback=artifact_feedback,
        artifact_gate_loss_config=artifact_gate_loss_config,
        anchor_checkpoint=anchor_checkpoint,
        anchor_lambda=anchor_lambda,
        trainer_audit=audit,
        gpu_index=gpu_index,
    )
    plan["stage"] = "text_preservation_blended_v11_source_filtered_candidate"
    plan["artifact_feedback_source_buckets"] = ["00_fixed_gate"]
    plan["artifact_gate_loss_config_write_command"] = (
        "python -m gemmanima.cli text-preservation-v11-artifact-gate-loss-config "
        f"--output \"{Path(artifact_gate_loss_config)}\" --json"
    )
    plan["training_strategy"] = {
        "method": "fixed_gate_source_filtered_artifact_feedback",
        "resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
        "objective": (
            "MSE bridge objective plus artifact feedback sample weights limited to fixed-gate shards "
            "and v5 KV-anchor regularization"
        ),
        "intent": "avoid v9/v10 feedback spillover into v5 text replay and hard-negative replay shards",
    }
    plan["post_train_qwen_eval_plan_command"] = (
        "python -m gemmanima.cli text-rendering-qwen-baseline-plan "
        f"--student-checkpoint \"{Path(output)}\" "
        "--student-name gemma_text_preservation_blended_v11 --json"
    )
    return plan


def build_text_preservation_kv_delta_audit(
    *,
    baseline_checkpoint: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE,
    candidate_checkpoints: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    candidates = candidate_checkpoints or {
        "v9": DEFAULT_TEXT_PRESERVATION_BLEND_V9_BRIDGE,
        "v10": DEFAULT_TEXT_PRESERVATION_BLEND_V10_BRIDGE,
        "v11": DEFAULT_TEXT_PRESERVATION_BLEND_V11_BRIDGE,
    }
    baseline_path = Path(baseline_checkpoint)
    baseline_kv, baseline_summary = _load_checkpoint_kv_for_delta(baseline_path)
    payload: dict[str, Any] = {
        "stage": "text_preservation_kv_delta_audit",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "baseline": baseline_summary,
        "candidates": {},
    }
    for version, checkpoint in candidates.items():
        candidate_path = Path(checkpoint)
        candidate_kv, candidate_summary = _load_checkpoint_kv_for_delta(candidate_path)
        if baseline_kv is None:
            candidate_summary["status"] = "baseline_unavailable"
        elif candidate_kv is None:
            candidate_summary["status"] = "candidate_unavailable"
        else:
            candidate_summary.update(_compare_checkpoint_kv_delta(baseline_kv, candidate_kv))
        payload["candidates"][version] = candidate_summary
    payload["summary"] = _summarize_kv_delta_candidates(payload["candidates"])
    return payload


def write_text_preservation_kv_delta_audit(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_KV_DELTA_REPORT,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_kv_delta_audit(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_v12_surface_plan(
    *,
    release_gate: dict[str, Any] | None = None,
    kv_delta_audit: dict[str, Any] | None = None,
    baseline: str = "v5",
) -> dict[str, Any]:
    gate = release_gate if release_gate is not None else build_text_preservation_release_gate_status(baseline=baseline)
    kv_audit = kv_delta_audit if kv_delta_audit is not None else build_text_preservation_kv_delta_audit()
    release_status = gate.get("release_gate", {}).get("status")
    protected_baseline = gate.get("protected_baseline", baseline)
    return {
        "stage": "text_preservation_v12_training_surface_plan",
        "mode": "design_contract",
        "executes_gpu_commands": False,
        "protected_baseline": protected_baseline,
        "workflow_position": {
            "current_step": "v12_surface_redesign",
            "previous_step": "v9_v10_v11_rejected_and_kv_delta_audited",
            "next_step": "build_render_readability_label_manifest",
        },
        "evidence": {
            "release_gate_status": release_status,
            "promotion_recommendation": gate.get("promotion", {}).get("recommendation", {}),
            "kv_delta_summary": kv_audit.get("summary", {}),
        },
        "diagnosis": {
            "status": "training_surface_must_change",
            "reason": (
                "v9-v11 failed the fixed image-level promotion gate and KV delta audit shows "
                "the v5 anchor/source filter did not materially change drift"
            ),
        },
        "recommended_surface": {
            "id": "render_readability_conditioned_target_refresh",
            "status": "recommended",
            "description": (
                "build a new target/cache surface from rendered Qwen-vs-Gemma evidence, "
                "readability labels, and per-case promotion gates before issuing another GPU train command"
            ),
            "changes_from_v9_v11": [
                "new manifest joins prompt ids, rendered images, compare metrics, and manual readability labels",
                "curriculum is selected from failed/partial render outcomes instead of replay weight tweaks",
                "candidate training remains blocked until the new surface artifacts and trainer contract exist",
            ],
        },
        "required_artifacts_before_training": [
            "render_readability_label_manifest",
            "surface_curriculum_manifest",
            "qwen_target_refresh_manifest",
            "fixed6_per_case_baseline_map",
            "heldout_partial_failed_case_pack",
            "trainer_surface_contract_audit",
        ],
        "trainer_contract_required": [
            "readability_label_ingest",
            "surface_curriculum_manifest",
            "per_case_gate_loss_budget",
            "pre_train_promotion_gate_assertions",
        ],
        "forbidden_strategies": [
            "replay_weight_only_training",
            "source_bucket_filter_only_training",
            "same_kv_anchor_only_training",
            "bridge_val_mse_only_promotion",
        ],
        "gpu_training_permission": {
            "status": "blocked_until_surface_redesign_artifacts_exist",
            "train_command": None,
            "allowed_device_after_unblock": "RTX 4070 Ti SUPER only via CUDA_VISIBLE_DEVICES='0'",
            "reserved_device": "RTX 5060 / CUDA device 1",
        },
        "next_safe_actions": [
            {
                "action": "build_render_readability_label_manifest",
                "type": "code",
                "gpu_required": False,
            },
            {
                "action": "audit_trainer_surface_contract",
                "type": "code",
                "gpu_required": False,
            },
            {
                "action": "plan_v12_candidate_only_after_surface_contract_passes",
                "type": "planning",
                "gpu_required": False,
            },
        ],
    }


def write_text_preservation_v12_surface_plan(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V12_SURFACE_PLAN_REPORT,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v12_surface_plan(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_render_readability_label_manifest(
    *,
    fixed_review: str | Path = r"reports\text_rendering_qwen_baseline\visual_review_text_preservation_blended_v5.json",
    heldout_review: str | Path = r"reports\text_preservation_heldout_v5_clean\visual_review.json",
    heldout_report_root: str | Path = r"reports\text_preservation_heldout_v5_clean",
    general_review: str | Path = r"reports\general_scene_regression_v5_50\visual_review.json",
    general_metrics: str | Path = r"reports\general_scene_regression_v5_50\metrics_summary.json",
    protected_baseline: str = "v5",
) -> dict[str, Any]:
    fixed_review_path = Path(fixed_review)
    heldout_review_path = Path(heldout_review)
    heldout_root = Path(heldout_report_root)
    general_review_path = Path(general_review)
    general_metrics_path = Path(general_metrics)
    fixed_data = _read_optional_json(fixed_review_path) or {}
    heldout_data = _read_optional_json(heldout_review_path) or {}
    general_data = _read_optional_json(general_review_path) or {}
    general_metrics_data = _read_optional_json(general_metrics_path) or {}

    records: list[dict[str, Any]] = []
    for row in fixed_data.get("fixed_gate", {}).get("reports", []):
        report_path = Path(row.get("report", ""))
        compare = _read_optional_json(report_path) or {}
        records.append(
            _render_readability_record(
                suite="fixed6",
                case_id=report_path.name.split("_qwen_vs_")[0] if report_path.name else None,
                compare_report=report_path,
                compare=compare,
                readability_label="accepted_baseline",
                curriculum_role="fixed_gate_protection",
                source_review=fixed_review_path,
                baseline_metrics=row,
            )
        )

    heldout_labels: list[tuple[str, list[int]]] = [
        ("readable", [int(index) for index in heldout_data.get("readable_indices", [])]),
        ("partial", [int(index) for index in heldout_data.get("partial_indices", [])]),
        ("failed", [int(index) for index in heldout_data.get("failed_indices", [])]),
    ]
    for label, indices in heldout_labels:
        for index in indices:
            compare_report = _find_heldout_compare_report(heldout_root, index)
            compare = _read_optional_json(compare_report) or {}
            records.append(
                _render_readability_record(
                    suite="heldout_v5_clean",
                    case_id=f"text_preserve_heldout_clean_{index:03d}",
                    compare_report=compare_report,
                    compare=compare,
                    readability_label=label,
                    curriculum_role=_readability_curriculum_role(label),
                    source_review=heldout_review_path,
                    baseline_metrics={},
                )
            )

    label_counts: dict[str, int] = {}
    for record in records:
        label = record["readability_label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    priority_records = [
        record
        for record in records
        if record.get("curriculum_role") == "v12_priority_refresh"
    ]
    return {
        "stage": "text_preservation_render_readability_label_manifest",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "protected_baseline": protected_baseline,
        "workflow_position": {
            "current_step": "build_render_readability_label_manifest",
            "previous_step": "v12_surface_redesign",
            "next_step": "build_surface_curriculum_manifest",
        },
        "sources": {
            "fixed_review": _json_path(fixed_review_path),
            "heldout_review": _json_path(heldout_review_path),
            "heldout_report_root": _json_path(heldout_root),
            "general_review": _json_path(general_review_path),
            "general_metrics": _json_path(general_metrics_path),
        },
        "record_count": len(records),
        "label_counts": label_counts,
        "priority_refresh_count": len(priority_records),
        "records": records,
        "general_scene_guard": {
            "decision": general_data.get("decision"),
            "case_count": general_metrics_data.get("case_count"),
            "mse": general_metrics_data.get("mse"),
            "review": general_data,
        },
        "gpu_training_permission": {
            "status": "blocked_until_surface_curriculum_manifest_exists",
            "train_command": None,
            "allowed_device_after_unblock": "RTX 4070 Ti SUPER only via CUDA_VISIBLE_DEVICES='0'",
            "reserved_device": "RTX 5060 / CUDA device 1",
        },
    }


def write_text_preservation_render_readability_label_manifest(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_RENDER_READABILITY_MANIFEST,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_render_readability_label_manifest(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_surface_curriculum_manifest(
    *,
    label_manifest: dict[str, Any] | str | Path = DEFAULT_TEXT_PRESERVATION_RENDER_READABILITY_MANIFEST,
    protected_baseline: str = "v5",
    max_readable_guards: int = 12,
) -> dict[str, Any]:
    source_path: Path | None = None
    if isinstance(label_manifest, dict):
        labels = label_manifest
    else:
        source_path = Path(label_manifest)
        labels = _read_optional_json(source_path) or {"records": []}
    records = list(labels.get("records", []))
    priority_records = [
        record
        for record in records
        if record.get("curriculum_role") == "v12_priority_refresh"
    ]
    fixed_guards = [
        record
        for record in records
        if record.get("curriculum_role") == "fixed_gate_protection"
    ]
    readable_guards = sorted(
        [
            record
            for record in records
            if record.get("curriculum_role") == "readable_replay_guard"
        ],
        key=lambda record: float(record.get("image_mse") or 0.0),
        reverse=True,
    )[:max_readable_guards]
    curriculum = [
        _surface_curriculum_entry(record)
        for record in sorted(
            priority_records,
            key=lambda record: (
                0 if record.get("readability_label") == "failed" else 1,
                -float(record.get("image_mse") or 0.0),
            ),
        )
    ]
    curriculum.extend(_surface_curriculum_entry(record) for record in fixed_guards)
    curriculum.extend(_surface_curriculum_entry(record) for record in readable_guards)
    counts: dict[str, int] = {}
    for entry in curriculum:
        bucket = entry["curriculum_bucket"]
        counts[bucket] = counts.get(bucket, 0) + 1
    return {
        "stage": "text_preservation_surface_curriculum_manifest",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "protected_baseline": protected_baseline,
        "workflow_position": {
            "current_step": "build_surface_curriculum_manifest",
            "previous_step": "build_render_readability_label_manifest",
            "next_step": "build_qwen_target_refresh_manifest",
        },
        "source_label_manifest": _json_path(source_path) if source_path is not None else None,
        "record_count": len(records),
        "priority_refresh_count": len(priority_records),
        "fixed_gate_guard_count": len(fixed_guards),
        "readable_replay_guard_count": len(readable_guards),
        "curriculum_count": len(curriculum),
        "curriculum_counts": counts,
        "curriculum": curriculum,
        "next_required_artifacts": [
            "qwen_target_refresh_manifest",
            "heldout_partial_failed_case_pack",
            "trainer_surface_contract_audit",
        ],
        "gpu_training_permission": {
            "status": "blocked_until_qwen_target_refresh_and_trainer_contract_exist",
            "train_command": None,
            "allowed_device_after_unblock": "RTX 4070 Ti SUPER only via CUDA_VISIBLE_DEVICES='0'",
            "reserved_device": "RTX 5060 / CUDA device 1",
        },
    }


def write_text_preservation_surface_curriculum_manifest(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_SURFACE_CURRICULUM_MANIFEST,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_surface_curriculum_manifest(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_qwen_target_refresh_manifest(
    *,
    curriculum_manifest: dict[str, Any] | str | Path = DEFAULT_TEXT_PRESERVATION_SURFACE_CURRICULUM_MANIFEST,
    prompt_file: str | Path = DEFAULT_TEXT_PRESERVATION_QWEN_TARGET_REFRESH_PROMPTS,
    target_dir: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V12_TARGET_DIR,
    start_idx: int = 990000,
    gpu_index: int = 0,
) -> dict[str, Any]:
    source_path: Path | None = None
    if isinstance(curriculum_manifest, dict):
        curriculum = curriculum_manifest
    else:
        source_path = Path(curriculum_manifest)
        curriculum = _read_optional_json(source_path) or {"curriculum": []}
    prompt_path = Path(prompt_file)
    target_path = Path(target_dir)
    prompt_records = [
        _qwen_target_refresh_prompt_record(entry, index=index, start_idx=start_idx)
        for index, entry in enumerate(curriculum.get("curriculum", []))
    ]
    return {
        "stage": "text_preservation_qwen_target_refresh_manifest",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "workflow_position": {
            "current_step": "build_qwen_target_refresh_manifest",
            "previous_step": "build_surface_curriculum_manifest",
            "next_step": "audit_trainer_surface_contract",
        },
        "source_curriculum_manifest": _json_path(source_path) if source_path is not None else None,
        "prompt_file": _json_path(prompt_path),
        "target_dir": _json_path(target_path),
        "prompt_record_count": len(prompt_records),
        "prompt_records": prompt_records,
        "target_cache_command": build_cache_targets_command(
            subset_path=prompt_path,
            outdir=target_path,
            shard=1000,
            gpu_index=gpu_index,
        ),
        "setup_commands": [
            f"New-Item -ItemType Directory -Force -Path \"{prompt_path.parent}\"",
            f"New-Item -ItemType Directory -Force -Path \"{target_path}\"",
        ],
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "gpu_training_permission": {
            "status": "blocked_until_qwen_targets_cached_and_trainer_contract_exists",
            "train_command": None,
        },
    }


def write_text_preservation_qwen_target_refresh_manifest(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_QWEN_TARGET_REFRESH_MANIFEST,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_qwen_target_refresh_manifest(**kwargs)
    prompt_path = Path(payload["prompt_file"])
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in payload["prompt_records"]) + "\n",
        encoding="utf-8",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_v12_trainer_surface_contract_audit(
    *,
    train_script: str | Path = DEFAULT_TRAIN_SCRIPT,
) -> dict[str, Any]:
    script_path = Path(train_script)
    source = script_path.read_text(encoding="utf-8", errors="ignore") if script_path.exists() else ""
    feature_tokens = {
        "readability_label_ingest": ["readability_label", "readability_feedback"],
        "surface_curriculum_manifest": ["--surface-curriculum", "surface_curriculum_manifest"],
        "per_case_gate_loss_budget": ["per_case_gate_loss_budget", "--per-case-gate-loss"],
        "pre_train_promotion_gate_assertions": ["pre_train_promotion_gate", "--pre-train-promotion-gate"],
    }
    present_features = [
        feature_id
        for feature_id, tokens in feature_tokens.items()
        if any(token in source for token in tokens)
    ]
    missing_features = [
        feature_id
        for feature_id in feature_tokens
        if feature_id not in present_features
    ]
    status = "supported" if not missing_features else "missing_surface_contract_support"
    return {
        "stage": "text_preservation_v12_trainer_surface_contract_audit",
        "mode": "trainer_surface_contract_audit",
        "executes_gpu_commands": False,
        "workflow_position": {
            "current_step": "audit_trainer_surface_contract",
            "previous_step": "build_qwen_target_refresh_manifest",
            "next_step": "cache_qwen_target_refresh_after_contract_support",
        },
        "train_script": _json_path(script_path),
        "trainer_surface_contract": {
            "status": status,
            "script_exists": script_path.exists(),
            "required_features": list(feature_tokens),
            "present_features": present_features,
            "missing_features": missing_features,
        },
        "gpu_training_permission": (
            {
                "status": "blocked_until_qwen_targets_are_cached",
                "train_command": None,
                "reason": "trainer surface contract exists but Qwen target refresh cache is not confirmed",
            }
            if status == "supported"
            else {
                "status": "blocked_until_trainer_surface_contract_support",
                "train_command": None,
                "reason": "trainer must support v12 surface curriculum, per-case gate budget, and pre-train promotion assertions",
            }
        ),
    }


def write_text_preservation_v12_trainer_surface_contract_audit(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V12_TRAINER_CONTRACT_AUDIT,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v12_trainer_surface_contract_audit(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_v13_recovery_plan(
    *,
    promotion_status: str | Path = r"reports\text_rendering_qwen_baseline\promotion_status.json",
    baseline: str = "v5",
    rejected_version: str = "v12",
    gpu_index: int = 0,
) -> dict[str, Any]:
    status = _read_optional_json(promotion_status) or build_text_preservation_promotion_status(baseline=baseline)
    candidates = status.get("candidates", {})
    baseline_candidate = candidates.get(baseline, {})
    rejected_candidate = candidates.get(rejected_version, {})
    baseline_gate = baseline_candidate.get("fixed_gate", {})
    rejected_gate = rejected_candidate.get("fixed_gate", {})
    baseline_case_mse = _case_mse_map(baseline_gate)
    worst_regressions = sorted(
        [
            {
                "case_id": row["case_id"],
                "candidate_mse": row["mse"],
                "baseline_mse": baseline_case_mse.get(row["case_id"]),
                "delta_mse": (
                    row["mse"] - baseline_case_mse[row["case_id"]]
                    if row["case_id"] in baseline_case_mse
                    else None
                ),
                "report": row.get("report"),
            }
            for row in rejected_gate.get("reports", [])
        ],
        key=lambda row: row["delta_mse"] if row["delta_mse"] is not None else -1.0,
        reverse=True,
    )
    decision = rejected_candidate.get("decision", {})
    rejected = decision.get("status") == "reject"
    return {
        "stage": "text_preservation_v13_recovery_plan",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "protected_baseline": baseline,
        "rejected_version": rejected_version,
        "source_promotion_status": _json_path(Path(promotion_status)),
        "workflow_position": {
            "previous_step": "v12_fixed6_qwen_vs_gemma_evaluation",
            "current_step": "v13_recovery_planning",
            "next_safe_step": "build_guard_weighted_v13_manifest",
        },
        "gate_evidence": {
            "baseline_mean_mse": baseline_gate.get("mean_mse"),
            "candidate_mean_mse": rejected_gate.get("mean_mse"),
            "candidate_decision": decision,
            "failure_reasons": rejected_candidate.get("failure_reasons", []),
            "worst_regressions": worst_regressions[:6],
        },
        "diagnosis": {
            "status": "v12_rejected" if rejected else "needs_fresh_promotion_status",
            "hypotheses": [
                {
                    "name": "surface_refresh_overwrote_fixed_gate_alignment",
                    "evidence": "v12 used the new surface curriculum and passed bridge MSE, but fixed6 image MSE regressed on every protected case.",
                },
                {
                    "name": "bridge_mse_gate_is_not_a_render_gate",
                    "evidence": "v12 bridge validation MSE stayed below 0.004 while rendered image mean MSE exceeded the protected baseline.",
                },
                {
                    "name": "learning_rate_or_refresh_weight_too_aggressive",
                    "evidence": "v12 was trained from v5 on only the refreshed surface set; the largest regressions are protected text surfaces.",
                },
            ],
        },
        "v13_strategy": {
            "method": "protected_baseline_low_lr_guarded_recovery",
            "resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
            "do_not_resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V12_BRIDGE),
            "dataset_contract": [
                "fixed6 guard records must be present and higher weighted than refreshed partial/failed records",
                "readable v5 replay records stay in the mix to prevent collapse toward refreshed targets",
                "partial/failed refresh records are capped until fixed6 per-case drift is non-regressing",
            ],
            "training_contract": {
                "cuda_visible_devices": str(gpu_index),
                "max_lr": 1e-5,
                "epochs": 1,
                "per_case_gate_loss_budget": 0.001,
                "pre_train_promotion_gate": r"reports\text_rendering_qwen_baseline\release_gate_status.json",
            },
            "promotion_contract": [
                "fixed6 Qwen-vs-Gemma reports must be regenerated before any heldout expansion",
                "candidate must have mean MSE <= v5 and no per-case fixed6 regression",
                "v5 remains the protected baseline unless the promotion observer returns eligible_for_next_gate",
            ],
        },
        "planned_artifacts": {
            "v13_guard_manifest": r"reports\text_rendering_qwen_baseline\v13_guard_weighted_manifest.json",
            "v13_prompt_file": r"reports\text_rendering_qwen_baseline\v13_guard_weighted_prompts.jsonl",
            "v13_cache_root": r"runs\cache\text_preservation_blended_v13",
            "v13_checkpoint": r"runs\cache\text_preservation_blended_v13\bridge\text_preservation_blended_v13_bridge.pt",
        },
        "gpu_permission": {
            "status": "blocked_until_guard_manifest_exists" if rejected else "blocked_until_v12_rejection_confirmed",
            "allowed_gpu": f"RTX 4070 Ti SUPER only via CUDA_VISIBLE_DEVICES='{gpu_index}'",
            "forbidden_gpu": "RTX 5060",
        },
    }


def write_text_preservation_v13_recovery_plan(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V13_RECOVERY_PLAN_REPORT,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v13_recovery_plan(**kwargs)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_v13_guard_weighted_manifest(
    *,
    source_manifest: dict[str, Any] | str | Path = DEFAULT_TEXT_PRESERVATION_QWEN_TARGET_REFRESH_MANIFEST,
    prompt_file: str | Path = DEFAULT_TEXT_PRESERVATION_V13_GUARD_PROMPTS,
    target_dir: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V13_TARGET_DIR,
    max_refresh_records: int = 2,
    max_readable_guards: int = 4,
    start_idx: int = 991000,
    gpu_index: int = 0,
) -> dict[str, Any]:
    source_path: Path | None = None
    if isinstance(source_manifest, dict):
        source = source_manifest
    else:
        source_path = Path(source_manifest)
        source = _read_optional_json(source_path) or {"prompt_records": []}
    records = list(source.get("prompt_records", []))
    fixed_guards = [
        record
        for record in records
        if record.get("curriculum_bucket") == "fixed_gate_guard"
    ]
    refresh = sorted(
        [
            record
            for record in records
            if record.get("curriculum_bucket") in {"failed_refresh", "partial_refresh"}
        ],
        key=lambda record: (
            0 if record.get("curriculum_bucket") == "failed_refresh" else 1,
            int(record.get("eval_idx", 999999)),
        ),
    )[:max_refresh_records]
    readable_guards = [
        record
        for record in records
        if record.get("curriculum_bucket") == "readable_replay_guard"
    ][:max_readable_guards]
    selected = []
    selected.extend(
        _v13_guard_prompt_record(record, index=len(selected), start_idx=start_idx, bucket="fixed_gate_guard", weight=8.0)
        for record in fixed_guards
    )
    selected.extend(
        _v13_guard_prompt_record(record, index=len(selected), start_idx=start_idx, bucket="readable_replay_guard", weight=1.0)
        for record in readable_guards
    )
    selected.extend(
        _v13_guard_prompt_record(record, index=len(selected), start_idx=start_idx, bucket="capped_refresh_ablation", weight=0.75)
        for record in refresh
    )
    counts: dict[str, int] = {}
    weights: dict[str, float] = {}
    for record in selected:
        bucket = record["curriculum_bucket"]
        counts[bucket] = counts.get(bucket, 0) + 1
        weights[bucket] = weights.get(bucket, 0.0) + float(record.get("sample_weight") or 0.0)
    prompt_path = Path(prompt_file)
    target_path = Path(target_dir)
    return {
        "stage": "text_preservation_v13_guard_weighted_manifest",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "workflow_position": {
            "previous_step": "v13_recovery_planning",
            "current_step": "build_guard_weighted_v13_manifest",
            "next_step": "cache_v13_guard_weighted_qwen_targets",
        },
        "source_manifest": _json_path(source_path) if source_path is not None else None,
        "prompt_file": _json_path(prompt_path),
        "target_dir": _json_path(target_path),
        "prompt_record_count": len(selected),
        "source_counts": {
            "fixed_gate_guard": len(fixed_guards),
            "refresh_available": len(
                [
                    record
                    for record in records
                    if record.get("curriculum_bucket") in {"failed_refresh", "partial_refresh"}
                ]
            ),
            "readable_replay_guard_available": len(
                [
                    record
                    for record in records
                    if record.get("curriculum_bucket") == "readable_replay_guard"
                ]
            ),
        },
        "selected_counts": counts,
        "selected_weight_totals": weights,
        "prompt_records": selected,
        "target_cache_command": build_cache_targets_command(
            subset_path=prompt_path,
            outdir=target_path,
            shard=1000,
            gpu_index=gpu_index,
        ),
        "training_contract": {
            "resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
            "do_not_resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V12_BRIDGE),
            "max_lr": 1e-5,
            "per_case_gate_loss_budget": 0.001,
            "first_run_scope": "tiny_ablation_fixed6_guard_plus_two_refresh_cases",
        },
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "gpu_training_permission": {
            "status": "allowed_to_cache_targets_only",
            "train_command": None,
            "reason": "cache targets first, then audit pairing before any v13 train command is emitted",
        },
    }


def write_text_preservation_v13_guard_weighted_manifest(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V13_GUARD_MANIFEST,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v13_guard_weighted_manifest(**kwargs)
    prompt_path = Path(payload["prompt_file"])
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in payload["prompt_records"]) + "\n",
        encoding="utf-8",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_v14_focus_fixed_gate_manifest(
    *,
    source_manifest: dict[str, Any] | str | Path = DEFAULT_TEXT_PRESERVATION_QWEN_TARGET_REFRESH_MANIFEST,
    prompt_file: str | Path = DEFAULT_TEXT_PRESERVATION_V14_FOCUS_PROMPTS,
    target_dir: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V14_TARGET_DIR,
    focus_case_ids: tuple[str, ...] = ("text_eval_001_sign_luna_gate", "text_eval_006_label_tea"),
    start_idx: int = 992000,
    gpu_index: int = 0,
) -> dict[str, Any]:
    source_path: Path | None = None
    if isinstance(source_manifest, dict):
        source = source_manifest
    else:
        source_path = Path(source_manifest)
        source = _read_optional_json(source_path) or {"prompt_records": []}
    records = [
        record
        for record in source.get("prompt_records", [])
        if record.get("curriculum_bucket") == "fixed_gate_guard"
    ]
    selected = [
        _v13_guard_prompt_record(
            record,
            index=index,
            start_idx=start_idx,
            bucket="v14_fixed_gate_focus" if record.get("source_case_id") in focus_case_ids else "v14_fixed_gate_guard",
            weight=16.0 if record.get("source_case_id") in focus_case_ids else 6.0,
        )
        for index, record in enumerate(records)
    ]
    counts: dict[str, int] = {}
    weights: dict[str, float] = {}
    for record in selected:
        bucket = record["curriculum_bucket"]
        counts[bucket] = counts.get(bucket, 0) + 1
        weights[bucket] = weights.get(bucket, 0.0) + float(record.get("sample_weight") or 0.0)
    prompt_path = Path(prompt_file)
    target_path = Path(target_dir)
    return {
        "stage": "text_preservation_v14_focus_fixed_gate_manifest",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "workflow_position": {
            "previous_step": "v13_fixed6_rejected",
            "current_step": "build_v14_focus_fixed_gate_manifest",
            "next_step": "cache_v14_focus_qwen_targets",
        },
        "source_manifest": _json_path(source_path) if source_path is not None else None,
        "prompt_file": _json_path(prompt_path),
        "target_dir": _json_path(target_path),
        "focus_case_ids": list(focus_case_ids),
        "prompt_record_count": len(selected),
        "selected_counts": counts,
        "selected_weight_totals": weights,
        "prompt_records": selected,
        "target_cache_command": build_cache_targets_command(
            subset_path=prompt_path,
            outdir=target_path,
            shard=1000,
            gpu_index=gpu_index,
        ),
        "training_contract": {
            "resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
            "do_not_resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V13_BRIDGE),
            "max_lr": 5e-6,
            "per_case_gate_loss_budget": 0.0005,
            "first_run_scope": "focus_only_luna_gate_and_tea_fixed_gate_ablation",
            "refresh_records": 0,
        },
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "gpu_training_permission": {
            "status": "allowed_to_cache_targets_only",
            "train_command": None,
            "reason": "cache targets first, then audit pairing before the focus-only train command is emitted",
        },
    }


def write_text_preservation_v14_focus_fixed_gate_manifest(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V14_FOCUS_MANIFEST,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v14_focus_fixed_gate_manifest(**kwargs)
    prompt_path = Path(payload["prompt_file"])
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in payload["prompt_records"]) + "\n",
        encoding="utf-8",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_v17_targeted_teacher_refresh_manifest(
    *,
    source_manifest: dict[str, Any] | str | Path = DEFAULT_TEXT_PRESERVATION_QWEN_TARGET_REFRESH_MANIFEST,
    prompt_file: str | Path = DEFAULT_TEXT_PRESERVATION_V17_TARGETED_TEACHER_REFRESH_PROMPTS,
    target_dir: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V17_TARGET_DIR,
    focus_case_ids: tuple[str, ...] = (
        "text_eval_005_handwritten_note_meet_at_dawn",
        "text_eval_006_label_tea",
    ),
    focus_variant_count: int = 4,
    start_idx: int = 993000,
    gpu_index: int = 0,
) -> dict[str, Any]:
    source_path: Path | None = None
    if isinstance(source_manifest, dict):
        source = source_manifest
    else:
        source_path = Path(source_manifest)
        source = _read_optional_json(source_path) or {"prompt_records": []}
    fixed_gate_records = [
        record
        for record in source.get("prompt_records", [])
        if record.get("curriculum_bucket") == "fixed_gate_guard"
    ]
    guard_records = [
        _v17_prompt_record(
            record,
            index=index,
            start_idx=start_idx,
            bucket="v17_fixed_gate_guard",
            weight=6.0,
            src_prefix="v17_fixed_gate_guard",
        )
        for index, record in enumerate(fixed_gate_records)
    ]
    variant_entries: list[dict[str, Any]] = []
    focus_set = set(focus_case_ids)
    for record in fixed_gate_records:
        if record.get("source_case_id") in focus_set:
            variant_entries.extend(_v17_teacher_refresh_variants(record, limit=focus_variant_count))
    refresh_start = start_idx + len(guard_records)
    refresh_records = [
        _v17_prompt_record(
            record,
            index=index,
            start_idx=refresh_start,
            bucket="v17_targeted_teacher_refresh",
            weight=18.0,
            src_prefix="v17_teacher_refresh",
        )
        for index, record in enumerate(variant_entries)
    ]
    selected = [*guard_records, *refresh_records]
    counts: dict[str, int] = {}
    weights: dict[str, float] = {}
    for record in selected:
        bucket = record["curriculum_bucket"]
        counts[bucket] = counts.get(bucket, 0) + 1
        weights[bucket] = weights.get(bucket, 0.0) + float(record.get("sample_weight") or 0.0)
    prompt_path = Path(prompt_file)
    target_path = Path(target_dir)
    return {
        "stage": "text_preservation_v17_targeted_teacher_refresh_manifest",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "workflow_position": {
            "previous_step": "v16_true_focus_fixed6_rejected",
            "current_step": "build_v17_targeted_teacher_refresh_manifest",
            "next_step": "cache_v17_targeted_teacher_refresh_qwen_targets",
        },
        "source_manifest": _json_path(source_path) if source_path is not None else None,
        "prompt_file": _json_path(prompt_path),
        "target_dir": _json_path(target_path),
        "focus_case_ids": list(focus_case_ids),
        "prompt_record_count": len(selected),
        "selected_counts": counts,
        "selected_weight_totals": weights,
        "prompt_records": selected,
        "target_cache_command": build_cache_targets_command(
            subset_path=prompt_path,
            outdir=target_path,
            shard=1000,
            gpu_index=gpu_index,
        ),
        "training_contract": {
            "resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
            "do_not_resume_from": [
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V13_BRIDGE),
                r"runs/cache/text_preservation_blended_v16/bridge/text_preservation_blended_v16_bridge.pt",
            ],
            "output": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V17_BRIDGE),
            "max_lr": 3e-6,
            "per_case_gate_loss_budget": 0.0005,
            "artifact_gate_loss_config": {
                "min_weight": 1.0,
                "max_weight": 18.0,
            },
            "first_run_scope": "targeted_teacher_refresh_meet_at_dawn_and_tea_fixed6_regressions",
            "guard_records": len(guard_records),
            "refresh_records": len(refresh_records),
        },
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "gpu_training_permission": {
            "status": "allowed_to_cache_targets_only",
            "train_command": None,
            "reason": "cache refreshed Qwen targets first, then audit pairing before the v17 train command is emitted",
        },
    }


def write_text_preservation_v17_targeted_teacher_refresh_manifest(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V17_TARGETED_TEACHER_REFRESH_MANIFEST,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v17_targeted_teacher_refresh_manifest(**kwargs)
    prompt_path = Path(payload["prompt_file"])
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in payload["prompt_records"]) + "\n",
        encoding="utf-8",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_v18_tea_micro_refresh_manifest(
    *,
    source_manifest: dict[str, Any] | str | Path = DEFAULT_TEXT_PRESERVATION_QWEN_TARGET_REFRESH_MANIFEST,
    prompt_file: str | Path = DEFAULT_TEXT_PRESERVATION_V18_TEA_MICRO_REFRESH_PROMPTS,
    target_dir: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V18_TARGET_DIR,
    focus_case_ids: tuple[str, ...] = ("text_eval_006_label_tea",),
    v17_gain_guard_case_ids: tuple[str, ...] = ("text_eval_005_handwritten_note_meet_at_dawn",),
    focus_variant_count: int = 6,
    start_idx: int = 994000,
    gpu_index: int = 0,
) -> dict[str, Any]:
    source_path: Path | None = None
    if isinstance(source_manifest, dict):
        source = source_manifest
    else:
        source_path = Path(source_manifest)
        source = _read_optional_json(source_path) or {"prompt_records": []}
    fixed_gate_records = [
        record
        for record in source.get("prompt_records", [])
        if record.get("curriculum_bucket") == "fixed_gate_guard"
    ]
    fixed_guard_records = [
        _v17_prompt_record(
            record,
            index=index,
            start_idx=start_idx,
            bucket="v18_fixed_gate_guard",
            weight=6.0,
            src_prefix="v18_fixed_gate_guard",
        )
        for index, record in enumerate(fixed_gate_records)
    ]
    fixed_count = len(fixed_guard_records)
    gain_guard_set = set(v17_gain_guard_case_ids)
    gain_guard_source_records = [
        record for record in fixed_gate_records if record.get("source_case_id") in gain_guard_set
    ]
    gain_guard_records = [
        _v17_prompt_record(
            record,
            index=index,
            start_idx=start_idx + fixed_count,
            bucket="v18_v17_gain_guard",
            weight=12.0,
            src_prefix="v18_v17_gain_guard",
        )
        for index, record in enumerate(gain_guard_source_records)
    ]
    focus_set = set(focus_case_ids)
    variant_entries: list[dict[str, Any]] = []
    for record in fixed_gate_records:
        if record.get("source_case_id") in focus_set:
            variant_entries.extend(_v18_tea_micro_refresh_variants(record, limit=focus_variant_count))
    refresh_start = start_idx + fixed_count + len(gain_guard_records)
    refresh_records = [
        _v17_prompt_record(
            record,
            index=index,
            start_idx=refresh_start,
            bucket="v18_tea_micro_refresh",
            weight=20.0,
            src_prefix="v18_tea_micro_refresh",
        )
        for index, record in enumerate(variant_entries)
    ]
    selected = [*fixed_guard_records, *gain_guard_records, *refresh_records]
    counts: dict[str, int] = {}
    weights: dict[str, float] = {}
    for record in selected:
        bucket = record["curriculum_bucket"]
        counts[bucket] = counts.get(bucket, 0) + 1
        weights[bucket] = weights.get(bucket, 0.0) + float(record.get("sample_weight") or 0.0)
    prompt_path = Path(prompt_file)
    target_path = Path(target_dir)
    return {
        "stage": "text_preservation_v18_tea_micro_refresh_manifest",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "workflow_position": {
            "previous_step": "v17_targeted_teacher_refresh_fixed6_rejected",
            "current_step": "build_v18_tea_micro_refresh_manifest",
            "next_step": "cache_v18_tea_micro_refresh_qwen_targets",
        },
        "source_manifest": _json_path(source_path) if source_path is not None else None,
        "prompt_file": _json_path(prompt_path),
        "target_dir": _json_path(target_path),
        "focus_case_ids": list(focus_case_ids),
        "v17_gain_guard_case_ids": list(v17_gain_guard_case_ids),
        "prompt_record_count": len(selected),
        "selected_counts": counts,
        "selected_weight_totals": weights,
        "prompt_records": selected,
        "target_cache_command": build_cache_targets_command(
            subset_path=prompt_path,
            outdir=target_path,
            shard=1000,
            gpu_index=gpu_index,
        ),
        "training_contract": {
            "protected_baseline": "v5",
            "resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V17_BRIDGE),
            "baseline_resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
            "do_not_overwrite": [
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V17_BRIDGE),
            ],
            "output": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V18_BRIDGE),
            "max_lr": 1e-6,
            "per_case_gate_loss_budget": 0.00025,
            "artifact_gate_loss_config": {
                "min_weight": 1.0,
                "max_weight": 20.0,
            },
            "first_run_scope": "tea_only_micro_refresh_after_v17_single_case_regression",
            "guard_records": len(fixed_guard_records),
            "v17_gain_guard_records": len(gain_guard_records),
            "refresh_records": len(refresh_records),
        },
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "gpu_training_permission": {
            "status": "allowed_to_cache_targets_only",
            "train_command": None,
            "reason": "cache refreshed TEA micro-refresh targets first, then audit pairing before v18 train command is emitted",
        },
    }


def write_text_preservation_v18_tea_micro_refresh_manifest(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V18_TEA_MICRO_REFRESH_MANIFEST,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v18_tea_micro_refresh_manifest(**kwargs)
    prompt_path = Path(payload["prompt_file"])
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in payload["prompt_records"]) + "\n",
        encoding="utf-8",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_v19_dual_guard_refresh_manifest(
    *,
    source_manifest: dict[str, Any] | str | Path = DEFAULT_TEXT_PRESERVATION_QWEN_TARGET_REFRESH_MANIFEST,
    prompt_file: str | Path = DEFAULT_TEXT_PRESERVATION_V19_DUAL_GUARD_REFRESH_PROMPTS,
    target_dir: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V19_TARGET_DIR,
    focus_case_ids: tuple[str, ...] = (
        "text_eval_005_handwritten_note_meet_at_dawn",
        "text_eval_006_label_tea",
    ),
    stability_guard_case_ids: tuple[str, ...] = (
        "text_eval_002_book_cover_star_atlas",
        "text_eval_005_handwritten_note_meet_at_dawn",
        "text_eval_006_label_tea",
    ),
    focus_variant_count: int = 4,
    start_idx: int = 995000,
    gpu_index: int = 0,
) -> dict[str, Any]:
    source_path: Path | None = None
    if isinstance(source_manifest, dict):
        source = source_manifest
    else:
        source_path = Path(source_manifest)
        source = _read_optional_json(source_path) or {"prompt_records": []}
    fixed_gate_records = [
        record
        for record in source.get("prompt_records", [])
        if record.get("curriculum_bucket") == "fixed_gate_guard"
    ]
    fixed_guard_records = [
        _v17_prompt_record(
            record,
            index=index,
            start_idx=start_idx,
            bucket="v19_fixed_gate_guard",
            weight=8.0,
            src_prefix="v19_fixed_gate_guard",
        )
        for index, record in enumerate(fixed_gate_records)
    ]
    stability_guard_set = set(stability_guard_case_ids)
    stability_source_records = [
        record for record in fixed_gate_records if record.get("source_case_id") in stability_guard_set
    ]
    stability_start = start_idx + len(fixed_guard_records)
    stability_records = [
        _v17_prompt_record(
            record,
            index=index,
            start_idx=stability_start,
            bucket="v19_stability_guard",
            weight=10.0,
            src_prefix="v19_stability_guard",
        )
        for index, record in enumerate(stability_source_records)
    ]
    focus_set = set(focus_case_ids)
    variant_entries: list[dict[str, Any]] = []
    for record in fixed_gate_records:
        if record.get("source_case_id") in focus_set:
            variant_entries.extend(_v19_dual_guard_refresh_variants(record, limit=focus_variant_count))
    refresh_start = stability_start + len(stability_records)
    refresh_records = [
        _v17_prompt_record(
            record,
            index=index,
            start_idx=refresh_start,
            bucket="v19_meet_tea_refresh",
            weight=14.0,
            src_prefix="v19_meet_tea_refresh",
        )
        for index, record in enumerate(variant_entries)
    ]
    selected = [*fixed_guard_records, *stability_records, *refresh_records]
    counts: dict[str, int] = {}
    weights: dict[str, float] = {}
    for record in selected:
        bucket = record["curriculum_bucket"]
        counts[bucket] = counts.get(bucket, 0) + 1
        weights[bucket] = weights.get(bucket, 0.0) + float(record.get("sample_weight") or 0.0)
    prompt_path = Path(prompt_file)
    target_path = Path(target_dir)
    return {
        "stage": "text_preservation_v19_dual_guard_refresh_manifest",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "workflow_position": {
            "previous_step": "v18_tea_micro_refresh_fixed6_rejected",
            "current_step": "build_v19_dual_guard_refresh_manifest",
            "next_step": "cache_v19_dual_guard_refresh_qwen_targets",
        },
        "source_manifest": _json_path(source_path) if source_path is not None else None,
        "prompt_file": _json_path(prompt_path),
        "target_dir": _json_path(target_path),
        "focus_case_ids": list(focus_case_ids),
        "stability_guard_case_ids": list(stability_guard_case_ids),
        "prompt_record_count": len(selected),
        "selected_counts": counts,
        "selected_weight_totals": weights,
        "prompt_records": selected,
        "target_cache_command": build_cache_targets_command(
            subset_path=prompt_path,
            outdir=target_path,
            shard=1000,
            gpu_index=gpu_index,
        ),
        "training_contract": {
            "protected_baseline": "v5",
            "resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
            "do_not_resume_from": [
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V17_BRIDGE),
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V18_BRIDGE),
            ],
            "do_not_overwrite": [
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V17_BRIDGE),
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V18_BRIDGE),
            ],
            "output": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V19_BRIDGE),
            "max_lr": 1.5e-6,
            "per_case_gate_loss_budget": 0.0002,
            "artifact_gate_loss_config": {
                "min_weight": 1.0,
                "max_weight": 14.0,
            },
            "first_run_scope": "meet_at_dawn_and_tea_dual_refresh_from_v5_with_star_atlas_stability_guard",
            "guard_records": len(fixed_guard_records),
            "stability_guard_records": len(stability_records),
            "refresh_records": len(refresh_records),
        },
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "gpu_training_permission": {
            "status": "allowed_to_cache_targets_only",
            "train_command": None,
            "reason": "cache refreshed v19 dual-guard targets first, then audit pairing before training command is emitted",
        },
    }


def write_text_preservation_v19_dual_guard_refresh_manifest(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V19_DUAL_GUARD_REFRESH_MANIFEST,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v19_dual_guard_refresh_manifest(**kwargs)
    prompt_path = Path(payload["prompt_file"])
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in payload["prompt_records"]) + "\n",
        encoding="utf-8",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_v23_hard_heldout_refresh_manifest(
    *,
    source_manifest: dict[str, Any] | str | Path = DEFAULT_TEXT_PRESERVATION_QWEN_TARGET_REFRESH_MANIFEST,
    heldout_review: str | Path = r"reports\text_preservation_heldout_v22_alpha28_clean\visual_review.json",
    heldout_prompts: str | Path = r"reports\text_preservation_heldout_v5_clean\prompts.jsonl",
    prompt_file: str | Path = DEFAULT_TEXT_PRESERVATION_V23_HARD_HELDOUT_REFRESH_PROMPTS,
    target_dir: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V23_TARGET_DIR,
    max_partial_records: int = 12,
    start_idx: int = 995000,
    gpu_index: int = 0,
) -> dict[str, Any]:
    source_path: Path | None = None
    if isinstance(source_manifest, dict):
        source = source_manifest
    else:
        source_path = Path(source_manifest)
        source = _read_optional_json(source_path) or {"prompt_records": []}

    fixed_gate_records = [
        record
        for record in source.get("prompt_records", [])
        if record.get("curriculum_bucket") == "fixed_gate_guard"
    ]
    fixed_guard_records = [
        _v17_prompt_record(
            record,
            index=index,
            start_idx=start_idx,
            bucket="v23_fixed_gate_guard",
            weight=8.0,
            src_prefix="v23_fixed_gate_guard",
        )
        for index, record in enumerate(fixed_gate_records)
    ]

    review_path = Path(heldout_review)
    review = _read_optional_json(review_path) or {}
    failed_indices = [int(index) for index in review.get("failed_indices", [])]
    partial_indices = [int(index) for index in review.get("partial_indices", [])[: max(0, max_partial_records)]]
    prompt_rows = _read_feedback_jsonl(Path(heldout_prompts))
    prompts_by_index: dict[int, dict[str, Any]] = {}
    for row_index, row in enumerate(prompt_rows):
        eval_index = int(row.get("eval_idx", row_index))
        prompts_by_index[eval_index] = row

    failed_entries = [
        _v23_heldout_refresh_entry(prompts_by_index.get(index), index=index, label="failed")
        for index in failed_indices
        if prompts_by_index.get(index) is not None
    ]
    failed_start = start_idx + len(fixed_guard_records)
    failed_records = [
        _v17_prompt_record(
            entry,
            index=index,
            start_idx=failed_start,
            bucket="v23_failed_heldout_refresh",
            weight=18.0,
            src_prefix="v23_failed_heldout_refresh",
        )
        for index, entry in enumerate(failed_entries)
    ]

    partial_entries = [
        _v23_heldout_refresh_entry(prompts_by_index.get(index), index=index, label="partial")
        for index in partial_indices
        if prompts_by_index.get(index) is not None
    ]
    partial_start = failed_start + len(failed_records)
    partial_records = [
        _v17_prompt_record(
            entry,
            index=index,
            start_idx=partial_start,
            bucket="v23_partial_heldout_refresh",
            weight=10.0,
            src_prefix="v23_partial_heldout_refresh",
        )
        for index, entry in enumerate(partial_entries)
    ]

    selected = [*fixed_guard_records, *failed_records, *partial_records]
    counts: dict[str, int] = {}
    weights: dict[str, float] = {}
    for record in selected:
        bucket = record["curriculum_bucket"]
        counts[bucket] = counts.get(bucket, 0) + 1
        weights[bucket] = weights.get(bucket, 0.0) + float(record.get("sample_weight") or 0.0)

    prompt_path = Path(prompt_file)
    target_path = Path(target_dir)
    return {
        "stage": "text_preservation_v23_hard_heldout_refresh_manifest",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "workflow_position": {
            "previous_step": "v22_alpha28_failed_heldout_promotion_gate",
            "current_step": "build_v23_hard_heldout_refresh_manifest",
            "next_step": "cache_v23_hard_heldout_refresh_qwen_targets",
        },
        "source_manifest": _json_path(source_path) if source_path is not None else None,
        "heldout_review": _json_path(review_path),
        "heldout_prompts": _json_path(Path(heldout_prompts)),
        "prompt_file": _json_path(prompt_path),
        "target_dir": _json_path(target_path),
        "failed_indices": failed_indices,
        "partial_indices": partial_indices,
        "prompt_record_count": len(selected),
        "selected_counts": counts,
        "selected_weight_totals": weights,
        "prompt_records": selected,
        "target_cache_command": build_cache_targets_command(
            subset_path=prompt_path,
            outdir=target_path,
            shard=1000,
            gpu_index=gpu_index,
        ),
        "training_contract": {
            "protected_baseline": "v5",
            "resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
            "do_not_resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V22_ALPHA28_BRIDGE),
            "do_not_overwrite": [
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V22_ALPHA28_BRIDGE),
            ],
            "output": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V23_BRIDGE),
            "max_lr": 1e-6,
            "per_case_gate_loss_budget": 0.00015,
            "artifact_gate_loss_config": {
                "min_weight": 1.0,
                "max_weight": 18.0,
            },
            "first_run_scope": "hard_heldout_refresh_after_v22_alpha28_from_v5",
            "fixed_gate_guard_records": len(fixed_guard_records),
            "failed_refresh_records": len(failed_records),
            "partial_refresh_records": len(partial_records),
        },
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "gpu_training_permission": {
            "status": "allowed_to_cache_targets_only",
            "train_command": None,
            "reason": "cache refreshed v23 hard-heldout Qwen targets first, then audit pairing before training command is emitted",
        },
    }


def write_text_preservation_v23_hard_heldout_refresh_manifest(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V23_HARD_HELDOUT_REFRESH_MANIFEST,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v23_hard_heldout_refresh_manifest(**kwargs)
    prompt_path = Path(payload["prompt_file"])
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in payload["prompt_records"]) + "\n",
        encoding="utf-8",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def build_text_preservation_v24_fixed_gate_protected_heldout_refresh_manifest(
    *,
    source_manifest: dict[str, Any] | str | Path = DEFAULT_TEXT_PRESERVATION_QWEN_TARGET_REFRESH_MANIFEST,
    heldout_review: str | Path = r"reports\text_preservation_heldout_v23_clean\visual_review.json",
    heldout_prompts: str | Path = r"reports\text_preservation_heldout_v5_clean\prompts.jsonl",
    prompt_file: str | Path = DEFAULT_TEXT_PRESERVATION_V24_FIXED_GATE_PROTECTED_HELDOUT_REFRESH_PROMPTS,
    target_dir: str | Path = DEFAULT_TEXT_PRESERVATION_BLEND_V24_TARGET_DIR,
    regression_case_ids: tuple[str, ...] = (
        "text_eval_005_handwritten_note_meet_at_dawn",
        "text_eval_006_label_tea",
    ),
    max_partial_records: int = 8,
    focus_variant_count: int = 4,
    start_idx: int = 996000,
    gpu_index: int = 0,
) -> dict[str, Any]:
    source_path: Path | None = None
    if isinstance(source_manifest, dict):
        source = source_manifest
    else:
        source_path = Path(source_manifest)
        source = _read_optional_json(source_path) or {"prompt_records": []}

    fixed_gate_records = [
        record
        for record in source.get("prompt_records", [])
        if record.get("curriculum_bucket") == "fixed_gate_guard"
    ]
    fixed_guard_records = [
        _v17_prompt_record(
            record,
            index=index,
            start_idx=start_idx,
            bucket="v24_fixed_gate_guard",
            weight=12.0,
            src_prefix="v24_fixed_gate_guard",
        )
        for index, record in enumerate(fixed_gate_records)
    ]

    regression_set = set(regression_case_ids)
    regression_source_records = [
        record for record in fixed_gate_records if record.get("source_case_id") in regression_set
    ]
    regression_focus_start = start_idx + len(fixed_guard_records)
    regression_focus_records = [
        _v17_prompt_record(
            _v24_fixed_gate_regression_entry(record),
            index=index,
            start_idx=regression_focus_start,
            bucket="v24_fixed_gate_regression_focus",
            weight=28.0,
            src_prefix="v24_fixed_gate_regression_focus",
        )
        for index, record in enumerate(regression_source_records)
    ]
    variant_entries: list[dict[str, Any]] = []
    for record in regression_source_records:
        variant_entries.extend(_v24_fixed_gate_protection_variants(record, limit=focus_variant_count))
    regression_variant_start = regression_focus_start + len(regression_focus_records)
    regression_variant_records = [
        _v17_prompt_record(
            entry,
            index=index,
            start_idx=regression_variant_start,
            bucket="v24_fixed_gate_regression_variant",
            weight=18.0,
            src_prefix="v24_fixed_gate_regression_variant",
        )
        for index, entry in enumerate(variant_entries)
    ]

    review_path = Path(heldout_review)
    review = _read_optional_json(review_path) or {}
    failed_indices = [int(index) for index in review.get("failed_indices", [])]
    partial_indices = [int(index) for index in review.get("partial_indices", [])[: max(0, max_partial_records)]]
    prompt_rows = _read_feedback_jsonl(Path(heldout_prompts))
    prompts_by_index: dict[int, dict[str, Any]] = {}
    for row_index, row in enumerate(prompt_rows):
        eval_index = int(row.get("eval_idx", row_index))
        prompts_by_index[eval_index] = row

    failed_entries = [
        _v23_heldout_refresh_entry(prompts_by_index.get(index), index=index, label="failed")
        for index in failed_indices
        if prompts_by_index.get(index) is not None
    ]
    failed_start = regression_variant_start + len(regression_variant_records)
    failed_records = [
        _v17_prompt_record(
            entry,
            index=index,
            start_idx=failed_start,
            bucket="v24_failed_heldout_refresh",
            weight=12.0,
            src_prefix="v24_failed_heldout_refresh",
        )
        for index, entry in enumerate(failed_entries)
    ]

    partial_entries = [
        _v23_heldout_refresh_entry(prompts_by_index.get(index), index=index, label="partial")
        for index in partial_indices
        if prompts_by_index.get(index) is not None
    ]
    partial_start = failed_start + len(failed_records)
    partial_records = [
        _v17_prompt_record(
            entry,
            index=index,
            start_idx=partial_start,
            bucket="v24_partial_heldout_refresh",
            weight=7.0,
            src_prefix="v24_partial_heldout_refresh",
        )
        for index, entry in enumerate(partial_entries)
    ]

    selected = [
        *fixed_guard_records,
        *regression_focus_records,
        *regression_variant_records,
        *failed_records,
        *partial_records,
    ]
    counts: dict[str, int] = {}
    weights: dict[str, float] = {}
    for record in selected:
        bucket = record["curriculum_bucket"]
        counts[bucket] = counts.get(bucket, 0) + 1
        weights[bucket] = weights.get(bucket, 0.0) + float(record.get("sample_weight") or 0.0)

    prompt_path = Path(prompt_file)
    target_path = Path(target_dir)
    return {
        "stage": "text_preservation_v24_fixed_gate_protected_heldout_refresh_manifest",
        "mode": "artifact_observer",
        "executes_gpu_commands": False,
        "workflow_position": {
            "previous_step": "v23_rejected_fixed_gate_regression",
            "current_step": "build_v24_fixed_gate_protected_heldout_refresh_manifest",
            "next_step": "cache_v24_fixed_gate_protected_heldout_qwen_targets",
        },
        "source_manifest": _json_path(source_path) if source_path is not None else None,
        "heldout_review": _json_path(review_path),
        "heldout_prompts": _json_path(Path(heldout_prompts)),
        "prompt_file": _json_path(prompt_path),
        "target_dir": _json_path(target_path),
        "regression_case_ids": list(regression_case_ids),
        "failed_indices": failed_indices,
        "partial_indices": partial_indices,
        "prompt_record_count": len(selected),
        "selected_counts": counts,
        "selected_weight_totals": weights,
        "prompt_records": selected,
        "target_cache_command": build_cache_targets_command(
            subset_path=prompt_path,
            outdir=target_path,
            shard=1000,
            gpu_index=gpu_index,
        ),
        "training_contract": {
            "protected_baseline": "v5",
            "resume_from": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
            "do_not_resume_from": [
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V22_ALPHA28_BRIDGE),
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V23_BRIDGE),
            ],
            "do_not_overwrite": [
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V5_BRIDGE),
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V22_ALPHA28_BRIDGE),
                _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V23_BRIDGE),
            ],
            "output": _json_path(DEFAULT_TEXT_PRESERVATION_BLEND_V24_BRIDGE),
            "max_lr": 8e-7,
            "per_case_gate_loss_budget": 0.0001,
            "artifact_gate_loss_config": {
                "min_weight": 1.0,
                "max_weight": 28.0,
            },
            "first_run_scope": "fixed_gate_protected_heldout_refresh_after_v23_regressed_meet_tea",
            "fixed_gate_guard_records": len(fixed_guard_records),
            "fixed_gate_regression_focus_records": len(regression_focus_records),
            "fixed_gate_regression_variant_records": len(regression_variant_records),
            "failed_refresh_records": len(failed_records),
            "partial_refresh_records": len(partial_records),
        },
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "gpu_training_permission": {
            "status": "allowed_to_cache_targets_only",
            "train_command": None,
            "reason": "cache v24 fixed-gate-protected targets first, then audit pairing before training command is emitted",
        },
    }


def write_text_preservation_v24_fixed_gate_protected_heldout_refresh_manifest(
    output: str | Path = DEFAULT_TEXT_PRESERVATION_V24_FIXED_GATE_PROTECTED_HELDOUT_REFRESH_MANIFEST,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_text_preservation_v24_fixed_gate_protected_heldout_refresh_manifest(**kwargs)
    prompt_path = Path(payload["prompt_file"])
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in payload["prompt_records"]) + "\n",
        encoding="utf-8",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output"] = _json_path(output_path)
    return payload


def _bridge_train_command(
    *,
    target_dir: Path,
    gemma_dir: Path,
    output: Path,
    resume_kv: Path,
    gpu_index: int,
    epochs: int,
    lr: float,
    batch_size: int,
    accum: int,
    val: int,
    prefetch_gb: float,
    python_exe: str | Path = DEFAULT_PYTHON,
    script: str | Path = DEFAULT_TRAIN_SCRIPT,
) -> str:
    return (
        f"$env:CUDA_VISIBLE_DEVICES='{gpu_index}'; "
        f"& \"{Path(python_exe)}\" \"{Path(script)}\" "
        f"--targets \"{target_dir}\" "
        f"--gemma \"{gemma_dir}\" "
        f"--out \"{output}\" "
        f"--epochs {epochs} "
        f"--lr {lr} "
        f"--batch-size {batch_size} "
        f"--accum {accum} "
        f"--val {val} "
        f"--prefetch-gb {prefetch_gb} "
        "--save-every-shards 1 "
        f"--resume-kv \"{resume_kv}\""
    )


def _heldout_eval_case(
    record: dict[str, Any],
    *,
    index: int,
    qwen_dir: Path,
    gemma_dir: Path,
    report_root: Path,
    student_checkpoint: Path,
    student_name: str,
    seed: int,
) -> dict[str, Any]:
    qwen_raw = qwen_dir / f"{index:03d}.png"
    gemma_raw = gemma_dir / f"{index:03d}.png"
    qwen_named = qwen_dir / f"{record['id']}.png"
    gemma_named = gemma_dir / f"{record['id']}.png"
    compare_report = report_root / f"{record['id']}_qwen_vs_{student_name}_compare.json"
    return {
        "id": record["id"],
        "target_text": record["target_text"],
        "seed": seed,
        "prompt": record["text"],
        "qwen_image": _json_path(qwen_named),
        "gemma_image": _json_path(gemma_named),
        "qwen_raw_image": _json_path(qwen_raw),
        "gemma_raw_image": _json_path(gemma_raw),
        "rename_commands": [
            f"Copy-Item -Force -LiteralPath {_quote_pwsh(str(qwen_raw))} -Destination {_quote_pwsh(str(qwen_named))}",
            f"Copy-Item -Force -LiteralPath {_quote_pwsh(str(gemma_raw))} -Destination {_quote_pwsh(str(gemma_named))}",
        ],
        "compare_report": _json_path(compare_report),
        "compare_command": _compare_command(
            prompt=record["text"],
            seed=seed,
            teacher_image=str(qwen_named),
            student_image=str(gemma_named),
            student_checkpoint=student_checkpoint,
            output=str(compare_report),
        ),
    }


def _general_scene_eval_case(
    record: dict[str, Any],
    *,
    index: int,
    qwen_dir: Path,
    gemma_dir: Path,
    report_root: Path,
    student_checkpoint: Path,
    student_name: str,
    seed: int,
) -> dict[str, Any]:
    qwen_raw = qwen_dir / f"{index:03d}.png"
    gemma_raw = gemma_dir / f"{index:03d}.png"
    qwen_named = qwen_dir / f"{record['id']}.png"
    gemma_named = gemma_dir / f"{record['id']}.png"
    compare_report = report_root / f"{record['id']}_qwen_vs_{student_name}_compare.json"
    return {
        "id": record["id"],
        "category": record["category"],
        "regression_focus": record["regression_focus"],
        "seed": seed,
        "prompt": record["text"],
        "qwen_image": _json_path(qwen_named),
        "gemma_image": _json_path(gemma_named),
        "qwen_raw_image": _json_path(qwen_raw),
        "gemma_raw_image": _json_path(gemma_raw),
        "rename_commands": [
            f"Copy-Item -Force -LiteralPath {_quote_pwsh(str(qwen_raw))} -Destination {_quote_pwsh(str(qwen_named))}",
            f"Copy-Item -Force -LiteralPath {_quote_pwsh(str(gemma_raw))} -Destination {_quote_pwsh(str(gemma_named))}",
        ],
        "compare_report": _json_path(compare_report),
        "compare_command": _compare_command(
            prompt=record["text"],
            seed=seed,
            teacher_image=str(qwen_named),
            student_image=str(gemma_named),
            student_checkpoint=student_checkpoint,
            output=str(compare_report),
        ),
    }


def _blend_link_commands(
    *,
    text_target_dir: Path,
    text_gemma_dir: Path,
    blend_target_dir: Path,
    blend_gemma_dir: Path,
    source_general_target_dir: Path,
    source_general_gemma_dir: Path,
    text_repeat: int,
    text_shards: int,
    general_shards: int,
) -> list[str]:
    commands: list[str] = []
    for index in range(text_repeat):
        for shard_index in range(text_shards):
            if text_shards == 1:
                name = f"00_text_{index:04d}.pt"
            else:
                name = f"00_text_r{index:02d}_s{shard_index:04d}.pt"
            source_name = f"shard_{shard_index:04d}.pt"
            commands.append(_hardlink_command(text_target_dir / source_name, blend_target_dir / name))
            commands.append(_hardlink_command(text_gemma_dir / source_name, blend_gemma_dir / name))
    for index in range(general_shards):
        src_name = f"shard_{index:04d}.pt"
        dst_name = f"10_general_{index:04d}.pt"
        commands.append(_hardlink_command(source_general_target_dir / src_name, blend_target_dir / dst_name))
        commands.append(_hardlink_command(source_general_gemma_dir / src_name, blend_gemma_dir / dst_name))
    return commands


def _balanced_v7_link_commands(
    *,
    v5_text_target_dir: Path,
    v5_text_gemma_dir: Path,
    hard_target_dir: Path,
    hard_gemma_dir: Path,
    source_general_target_dir: Path,
    source_general_gemma_dir: Path,
    blend_target_dir: Path,
    blend_gemma_dir: Path,
    v5_text_repeats: int,
    hard_negative_repeats: int,
    general_shards: int,
) -> list[str]:
    commands: list[str] = []
    for repeat_index in range(v5_text_repeats):
        for shard_index in range(2):
            source_name = f"shard_{shard_index:04d}.pt"
            dst_name = f"00_v5_text_r{repeat_index:02d}_s{shard_index:04d}.pt"
            commands.append(_hardlink_command(v5_text_target_dir / source_name, blend_target_dir / dst_name))
            commands.append(_hardlink_command(v5_text_gemma_dir / source_name, blend_gemma_dir / dst_name))
    for repeat_index in range(hard_negative_repeats):
        dst_name = f"05_hard_negative_r{repeat_index:02d}_s0000.pt"
        commands.append(_hardlink_command(hard_target_dir / "shard_0000.pt", blend_target_dir / dst_name))
        commands.append(_hardlink_command(hard_gemma_dir / "shard_0000.pt", blend_gemma_dir / dst_name))
    for shard_index in range(general_shards):
        source_name = f"shard_{shard_index:04d}.pt"
        dst_name = f"10_general_{shard_index:04d}.pt"
        commands.append(_hardlink_command(source_general_target_dir / source_name, blend_target_dir / dst_name))
        commands.append(_hardlink_command(source_general_gemma_dir / source_name, blend_gemma_dir / dst_name))
    return commands


def _fixed_gate_v8_link_commands(
    *,
    fixed_target_dir: Path,
    fixed_gemma_dir: Path,
    v5_text_target_dir: Path,
    v5_text_gemma_dir: Path,
    hard_target_dir: Path,
    hard_gemma_dir: Path,
    source_general_target_dir: Path,
    source_general_gemma_dir: Path,
    blend_target_dir: Path,
    blend_gemma_dir: Path,
    fixed_gate_repeats: int,
    v5_text_repeats: int,
    hard_negative_repeats: int,
    general_shards: int,
) -> list[str]:
    commands: list[str] = []
    for repeat_index in range(fixed_gate_repeats):
        dst_name = f"00_fixed_gate_r{repeat_index:02d}_s0000.pt"
        commands.append(_hardlink_command(fixed_target_dir / "shard_0000.pt", blend_target_dir / dst_name))
        commands.append(_hardlink_command(fixed_gemma_dir / "shard_0000.pt", blend_gemma_dir / dst_name))
    for repeat_index in range(v5_text_repeats):
        for shard_index in range(2):
            source_name = f"shard_{shard_index:04d}.pt"
            dst_name = f"10_v5_text_r{repeat_index:02d}_s{shard_index:04d}.pt"
            commands.append(_hardlink_command(v5_text_target_dir / source_name, blend_target_dir / dst_name))
            commands.append(_hardlink_command(v5_text_gemma_dir / source_name, blend_gemma_dir / dst_name))
    for repeat_index in range(hard_negative_repeats):
        dst_name = f"20_hard_negative_r{repeat_index:02d}_s0000.pt"
        commands.append(_hardlink_command(hard_target_dir / "shard_0000.pt", blend_target_dir / dst_name))
        commands.append(_hardlink_command(hard_gemma_dir / "shard_0000.pt", blend_gemma_dir / dst_name))
    for shard_index in range(general_shards):
        source_name = f"shard_{shard_index:04d}.pt"
        dst_name = f"30_general_{shard_index:04d}.pt"
        commands.append(_hardlink_command(source_general_target_dir / source_name, blend_target_dir / dst_name))
        commands.append(_hardlink_command(source_general_gemma_dir / source_name, blend_gemma_dir / dst_name))
    return commands


def _summarize_compare_reports(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("image_metrics", {})
        if metrics.get("mse") is None:
            continue
        rows.append(
            {
                "case_id": path.name.split("_qwen_vs_")[0],
                "report": _json_path(path),
                "mse": float(metrics["mse"]),
                "psnr_db": float(metrics["psnr_db"]) if metrics.get("psnr_db") is not None else None,
            }
        )
    if not rows:
        return {"count": 0, "mean_mse": None, "mean_psnr_db": None, "reports": []}
    mse_values = [row["mse"] for row in rows]
    psnr_values = [row["psnr_db"] for row in rows if row["psnr_db"] is not None]
    return {
        "count": len(rows),
        "mean_mse": math.fsum(mse_values) / len(mse_values),
        "min_mse": min(mse_values),
        "max_mse": max(mse_values),
        "mean_psnr_db": (math.fsum(psnr_values) / len(psnr_values)) if psnr_values else None,
        "reports": rows,
    }


def _build_promotion_recommendation(
    *,
    candidates: dict[str, dict[str, Any]],
    baseline: str,
) -> dict[str, Any]:
    eligible = [
        version
        for version, candidate in candidates.items()
        if version != baseline and candidate.get("decision", {}).get("status") == "eligible_for_next_gate"
    ]
    rejected = [
        version
        for version, candidate in candidates.items()
        if version != baseline and candidate.get("decision", {}).get("status") == "reject"
    ]
    pending = [
        version
        for version, candidate in candidates.items()
        if version != baseline and candidate.get("decision", {}).get("status") == "pending"
    ]
    if eligible:
        best = min(
            eligible,
            key=lambda version: candidates[version]["fixed_gate"]["mean_mse"],
        )
        return {
            "status": "candidate_needs_next_gate",
            "protected_baseline": baseline,
            "promote_candidate": best,
            "rejected_candidates": rejected,
            "pending_candidates": pending,
            "reason": "candidate passed fixed gate but still needs heldout/general promotion checks",
        }
    return {
        "status": "protect_baseline",
        "protected_baseline": baseline,
        "promote_candidate": None,
        "rejected_candidates": rejected,
        "pending_candidates": pending,
        "reason": "no candidate beat the protected baseline fixed gate",
    }


def _promotion_required_artifacts(
    *,
    fixed_report_root: Path,
    student_name: str,
    fixed_report_count: int,
    expected_fixed_count: int | None,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    artifacts = [
        {
            "name": "fixed_gate_reports",
            "path": str(fixed_report_root / f"*_qwen_vs_{student_name}_compare.json"),
            "expected_count": expected_fixed_count,
            "actual_count": fixed_report_count,
            "exists": fixed_report_count > 0,
        }
    ]
    for key, artifact_name in (
        ("heldout_metrics", "heldout_metrics"),
        ("heldout_review", "heldout_review"),
        ("general_metrics", "general_scene_metrics"),
        ("general_review", "general_scene_review"),
    ):
        if key in spec:
            path = Path(spec[key])
            artifacts.append(
                {
                    "name": artifact_name,
                    "path": _json_path(path),
                    "exists": path.exists(),
                }
            )
    return artifacts


def _fixed_gate_failure_reasons(
    *,
    fixed_gate: dict[str, Any],
    baseline_gate: dict[str, Any],
    baseline_case_mse: dict[str, float],
) -> list[str]:
    reasons: list[str] = []
    mean_mse = fixed_gate.get("mean_mse")
    baseline_mean = baseline_gate.get("mean_mse")
    if mean_mse is None or baseline_mean is None:
        return reasons
    if fixed_gate.get("count", 0) < baseline_gate.get("count", 0):
        reasons.append("fixed_gate_count_below_required")
    if mean_mse > baseline_mean:
        reasons.append("fixed_gate_mean_regression")
    if fixed_gate.get("max_mse") is not None and baseline_gate.get("max_mse") is not None:
        if fixed_gate["max_mse"] > baseline_gate["max_mse"]:
            reasons.append("fixed_gate_max_regression")
    for row in fixed_gate.get("reports", []):
        case_id = row.get("case_id")
        if case_id in baseline_case_mse and row["mse"] > baseline_case_mse[case_id]:
            reasons.append(f"fixed_gate_case_regression:{case_id}")
    return reasons


def _fixed_gate_reject_reason(*, mean_mse: float, baseline_mean: float) -> str:
    if mean_mse > baseline_mean:
        return f"fixed gate regressed: mean MSE {mean_mse:.12f} > baseline {baseline_mean:.12f}"
    return f"fixed gate per-case regression despite mean MSE {mean_mse:.12f} <= baseline {baseline_mean:.12f}"


def _case_mse_map(fixed_gate: dict[str, Any]) -> dict[str, float]:
    return {
        row["case_id"]: row["mse"]
        for row in fixed_gate.get("reports", [])
        if row.get("case_id") is not None
    }


def _read_optional_json(path: Any) -> Any:
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _read_feedback_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_checkpoint_kv_for_delta(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    summary: dict[str, Any] = {
        "checkpoint": _json_path(path),
        "exists": path.exists(),
        "readable": False,
        "kv_key_count": 0,
        "kv_numel": 0,
    }
    if not path.exists():
        summary["status"] = "missing_checkpoint"
        return None, summary
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - defensive report path
        summary["status"] = "unreadable_checkpoint"
        summary["error"] = str(exc)
        return None, summary
    kv = checkpoint.get("kv") if isinstance(checkpoint, dict) else None
    if not isinstance(kv, dict):
        summary["status"] = "missing_kv"
        summary["readable"] = True
        return None, summary
    tensor_kv = {
        key: value.detach().cpu()
        for key, value in kv.items()
        if hasattr(value, "detach") and hasattr(value, "numel")
    }
    summary.update(
        {
            "status": "loaded",
            "readable": True,
            "kv_key_count": len(tensor_kv),
            "kv_numel": int(math.fsum(float(tensor.numel()) for tensor in tensor_kv.values())),
        }
    )
    return tensor_kv, summary


def _render_readability_record(
    *,
    suite: str,
    case_id: str | None,
    compare_report: Path,
    compare: dict[str, Any],
    readability_label: str,
    curriculum_role: str,
    source_review: Path,
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    metrics = compare.get("image_metrics", {})
    prompt = compare.get("prompt")
    images = compare.get("images", {})
    return {
        "suite": suite,
        "case_id": case_id or compare_report.stem,
        "prompt": prompt,
        "target_text": _extract_target_text(prompt),
        "teacher_image": _json_path(Path(images.get("teacher", metrics.get("reference", "")))),
        "student_image": _json_path(Path(images.get("student", metrics.get("candidate", "")))),
        "compare_report": _json_path(compare_report),
        "source_review": _json_path(source_review),
        "readability_label": readability_label,
        "curriculum_role": curriculum_role,
        "image_mse": metrics.get("mse", baseline_metrics.get("mse")),
        "psnr_db": metrics.get("psnr_db", baseline_metrics.get("psnr_db")),
    }


def _surface_curriculum_entry(record: dict[str, Any]) -> dict[str, Any]:
    label = str(record.get("readability_label", "unknown"))
    if label == "failed":
        bucket = "failed_refresh"
        weight = 4.0
    elif label == "partial":
        bucket = "partial_refresh"
        weight = 2.5
    elif label == "accepted_baseline":
        bucket = "fixed_gate_guard"
        weight = 1.0
    elif label == "readable":
        bucket = "readable_replay_guard"
        weight = 0.5
    else:
        bucket = "other"
        weight = 1.0
    return {
        "case_id": record.get("case_id"),
        "target_text": record.get("target_text"),
        "readability_label": label,
        "curriculum_bucket": bucket,
        "sample_weight": weight,
        "prompt": record.get("prompt"),
        "teacher_image": record.get("teacher_image"),
        "student_image": record.get("student_image"),
        "compare_report": record.get("compare_report"),
        "image_mse": record.get("image_mse"),
        "psnr_db": record.get("psnr_db"),
    }


def _qwen_target_refresh_prompt_record(entry: dict[str, Any], *, index: int, start_idx: int) -> dict[str, Any]:
    case_id = str(entry.get("case_id") or f"v12_refresh_{index:03d}")
    prompt = entry.get("prompt") or ""
    return {
        "text": prompt,
        "src": f"v12_qwen_refresh_{index:03d}",
        "id": f"v12_qwen_refresh_{index:03d}_{case_id}",
        "idx": start_idx + index,
        "eval_idx": index,
        "target_text": entry.get("target_text"),
        "source_case_id": case_id,
        "curriculum_bucket": entry.get("curriculum_bucket"),
        "sample_weight": entry.get("sample_weight"),
        "readability_label": entry.get("readability_label"),
        "source_compare_report": entry.get("compare_report"),
    }


def _v13_guard_prompt_record(
    entry: dict[str, Any],
    *,
    index: int,
    start_idx: int,
    bucket: str,
    weight: float,
) -> dict[str, Any]:
    case_id = str(entry.get("source_case_id") or entry.get("case_id") or f"v13_guard_{index:03d}")
    prompt = entry.get("text") or entry.get("prompt") or ""
    return {
        "text": prompt,
        "src": f"v13_guard_{index:03d}",
        "id": f"v13_guard_{index:03d}_{case_id}",
        "idx": start_idx + index,
        "eval_idx": index,
        "target_text": entry.get("target_text"),
        "source_case_id": case_id,
        "source_curriculum_bucket": entry.get("curriculum_bucket"),
        "curriculum_bucket": bucket,
        "sample_weight": weight,
        "readability_label": entry.get("readability_label"),
        "source_compare_report": entry.get("source_compare_report") or entry.get("compare_report"),
    }


def _v17_prompt_record(
    entry: dict[str, Any],
    *,
    index: int,
    start_idx: int,
    bucket: str,
    weight: float,
    src_prefix: str,
) -> dict[str, Any]:
    case_id = str(entry.get("source_case_id") or entry.get("case_id") or f"{src_prefix}_{index:03d}")
    prompt = entry.get("text") or entry.get("prompt") or ""
    return {
        "text": prompt,
        "src": f"{src_prefix}_{index:03d}",
        "id": f"{src_prefix}_{index:03d}_{case_id}",
        "idx": start_idx + index,
        "eval_idx": index,
        "target_text": entry.get("target_text"),
        "source_case_id": case_id,
        "source_curriculum_bucket": entry.get("curriculum_bucket"),
        "curriculum_bucket": bucket,
        "sample_weight": weight,
        "readability_label": entry.get("readability_label"),
        "source_compare_report": entry.get("source_compare_report") or entry.get("compare_report"),
        "variant_id": entry.get("variant_id"),
        "teacher_refresh_strategy": entry.get("teacher_refresh_strategy"),
    }


def _v17_teacher_refresh_variants(entry: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    case_id = str(entry.get("source_case_id") or entry.get("case_id") or "")
    target_text = str(entry.get("target_text") or "")
    variants_by_case = {
        "text_eval_005_handwritten_note_meet_at_dawn": [
            (
                "large_front_note",
                "Object-only text rendering test: draw a large front-facing handwritten note pinned to a plain wooden door. "
                "The note clearly reads MEET AT DAWN in dark ink, with generous letter spacing and no extra words. "
                "No people, no characters, no hands, no faces; keep the note centered and readable.",
            ),
            (
                "clean_paper_note",
                "Object-only text rendering test: draw a clean cream paper note on a flat wooden door that clearly reads MEET AT DAWN. "
                "Use bold readable handwriting, high contrast ink, and a close crop around the note. "
                "No people, no characters, no hands, no faces.",
            ),
            (
                "block_handwriting_note",
                "Object-only text rendering test: draw a rectangular handwritten note with the exact words MEET AT DAWN. "
                "Use simple block-like handwriting, centered text, and a quiet wooden door background. "
                "No people, no characters, no hands, no faces.",
            ),
            (
                "pinned_door_note",
                "Object-only text rendering test: draw a pinned paper note on a wooden door, close enough to read MEET AT DAWN clearly. "
                "Keep the message as the only text, readable natural handwriting, centered composition. "
                "No people, no characters, no hands, no faces.",
            ),
        ],
        "text_eval_006_label_tea": [
            (
                "large_front_label",
                "Object-only text rendering test: draw a small glass jar facing front with a large plain paper label that clearly reads TEA. "
                "Use dark block letters on a light label, cozy kitchen lighting, no other text. "
                "No people, no characters, no hands, no faces.",
            ),
            (
                "high_contrast_label",
                "Object-only text rendering test: draw a centered glass tea jar with a high-contrast rectangular label reading TEA. "
                "The three letters should be large, simple, and readable; keep the jar and label centered. "
                "No people, no characters, no hands, no faces.",
            ),
            (
                "minimal_label",
                "Object-only text rendering test: draw a minimal kitchen jar with one clean paper label that says TEA. "
                "Use clear sans-serif letters, front view, close crop, and no other writing. "
                "No people, no characters, no hands, no faces.",
            ),
            (
                "warm_kitchen_label",
                "Object-only text rendering test: draw a warm-lit glass jar with a simple centered label that clearly reads TEA. "
                "Make the label large enough to inspect the letters and keep the background uncluttered. "
                "No people, no characters, no hands, no faces.",
            ),
        ],
    }
    templates = variants_by_case.get(
        case_id,
        [
            (
                "targeted_refresh",
                f"Object-only text rendering test: draw the target text {target_text} large, centered, high contrast, and clearly readable. "
                "No people, no characters, no hands, no faces.",
            )
        ],
    )
    variants: list[dict[str, Any]] = []
    for variant_id, prompt in templates[: max(0, limit)]:
        variant = dict(entry)
        variant["text"] = prompt
        variant["prompt"] = prompt
        variant["variant_id"] = variant_id
        variant["teacher_refresh_strategy"] = "targeted_fixed6_surface_prompt_refresh"
        variants.append(variant)
    return variants


def _v18_tea_micro_refresh_variants(entry: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    variants = _v17_teacher_refresh_variants(entry, limit=limit)
    for variant in variants:
        variant["teacher_refresh_strategy"] = "tea_only_micro_refresh_after_v17"
    return variants


def _v19_dual_guard_refresh_variants(entry: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    variants = _v17_teacher_refresh_variants(entry, limit=limit)
    for variant in variants:
        variant["teacher_refresh_strategy"] = "dual_guard_refresh_from_v5_after_v18"
    return variants


def _v23_heldout_refresh_entry(row: dict[str, Any] | None, *, index: int, label: str) -> dict[str, Any]:
    if row is None:
        row = {}
    case_id = str(row.get("src") or row.get("id") or f"text_preserve_heldout_clean_{index:03d}")
    prompt = str(row.get("text") or row.get("prompt") or "")
    return {
        "text": prompt,
        "prompt": prompt,
        "source_case_id": case_id,
        "target_text": row.get("target_text") or _extract_target_text(prompt),
        "curriculum_bucket": f"{label}_heldout_source",
        "readability_label": label,
        "teacher_refresh_strategy": "hard_heldout_refresh_after_v22_alpha28",
    }


def _v24_fixed_gate_regression_entry(entry: dict[str, Any]) -> dict[str, Any]:
    record = dict(entry)
    record["teacher_refresh_strategy"] = "fixed_gate_protection_after_v23"
    return record


def _v24_fixed_gate_protection_variants(entry: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    variants = _v17_teacher_refresh_variants(entry, limit=limit)
    for variant in variants:
        variant["teacher_refresh_strategy"] = "fixed_gate_protection_after_v23"
    return variants


def _readability_curriculum_role(label: str) -> str:
    if label in {"partial", "failed"}:
        return "v12_priority_refresh"
    if label == "readable":
        return "readable_replay_guard"
    return "fixed_gate_protection"


def _find_heldout_compare_report(root: Path, index: int) -> Path:
    default_path = root / (
        f"text_preserve_heldout_clean_{index:03d}_qwen_vs_gemma_text_preservation_blended_v5_compare.json"
    )
    if default_path.exists():
        return default_path
    matches = sorted(root.glob(f"text_preserve_heldout_clean_{index:03d}_qwen_vs_*_compare.json"))
    return matches[0] if matches else default_path


def _extract_target_text(prompt: str | None) -> str | None:
    if not prompt:
        return None
    patterns = (
        r"clearly reads ([^.;]+)",
        r"title ([A-Z0-9 ]+) printed",
        r"word ([A-Z0-9 ]+) repeated",
        r"label ([A-Z0-9 ]+) clearly",
        r"note .*? reads ([^,.;]+)",
        r"paper label that reads ([^,.;]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _default_fixed6_text_roi_map() -> dict[str, dict[str, Any]]:
    return {
        "text_eval_001_sign_luna_gate": {"name": "center_sign_text", "box": [96, 150, 416, 310]},
        "text_eval_002_book_cover_star_atlas": {"name": "book_title_center", "box": [88, 130, 424, 330]},
        "text_eval_003_magic_circle_aether": {"name": "circle_ring_text", "box": [70, 70, 442, 442]},
        "text_eval_004_ui_panel_hp_42": {"name": "ui_label_center", "box": [120, 150, 392, 330]},
        "text_eval_005_handwritten_note_meet_at_dawn": {"name": "door_note_center", "box": [115, 105, 397, 350]},
        "text_eval_006_label_tea": {"name": "jar_label_center", "box": [150, 170, 362, 330]},
    }


def _compare_case_id(path: Path) -> str:
    name = path.name
    if "_qwen_vs_" in name:
        return name.split("_qwen_vs_", 1)[0]
    return path.stem


def _text_roi_case_record(report_path: Path, roi: dict[str, Any] | None) -> dict[str, Any]:
    report = _read_optional_json(report_path) or {}
    images = report.get("images") or {}
    metrics = None
    if roi is not None and images.get("teacher") and images.get("student"):
        metrics = compare_text_roi_if_available(images["teacher"], images["student"], roi=roi)
    return {
        "compare_report": _json_path(report_path),
        "teacher": images.get("teacher"),
        "student": images.get("student"),
        "roi": roi,
        "text_roi_metrics": metrics,
    }


def _compare_checkpoint_kv_delta(
    baseline_kv: dict[str, Any],
    candidate_kv: dict[str, Any],
) -> dict[str, Any]:
    shared_keys = sorted(set(baseline_kv) & set(candidate_kv))
    tensor_rows: dict[str, Any] = {}
    total_sse = 0.0
    total_numel = 0
    tensor_mse_values: list[float] = []
    skipped_shape_mismatch: list[dict[str, Any]] = []
    for key in shared_keys:
        baseline_tensor = baseline_kv[key]
        candidate_tensor = candidate_kv[key]
        if tuple(baseline_tensor.shape) != tuple(candidate_tensor.shape):
            skipped_shape_mismatch.append(
                {
                    "key": key,
                    "baseline_shape": list(baseline_tensor.shape),
                    "candidate_shape": list(candidate_tensor.shape),
                }
            )
            continue
        diff = candidate_tensor.to(dtype=candidate_tensor.float().dtype) - baseline_tensor.to(
            dtype=baseline_tensor.float().dtype
        )
        diff = diff.double()
        sse = float((diff * diff).sum().item())
        numel = int(diff.numel())
        mse = sse / numel if numel else 0.0
        tensor_rows[key] = {
            "shape": list(diff.shape),
            "numel": numel,
            "mse": mse,
            "l2_norm": math.sqrt(sse),
            "max_abs": float(diff.abs().max().item()) if numel else 0.0,
        }
        total_sse += sse
        total_numel += numel
        tensor_mse_values.append(mse)
    compared_count = len(tensor_mse_values)
    if compared_count == 0:
        return {
            "status": "no_comparable_shared_kv_tensors",
            "shared_kv_key_count": len(shared_keys),
            "compared_kv_key_count": 0,
            "missing_from_candidate": sorted(set(baseline_kv) - set(candidate_kv)),
            "extra_in_candidate": sorted(set(candidate_kv) - set(baseline_kv)),
            "shape_mismatches": skipped_shape_mismatch,
            "tensors": {},
        }
    return {
        "status": "compared",
        "shared_kv_key_count": len(shared_keys),
        "compared_kv_key_count": compared_count,
        "missing_from_candidate": sorted(set(baseline_kv) - set(candidate_kv)),
        "extra_in_candidate": sorted(set(candidate_kv) - set(baseline_kv)),
        "shape_mismatches": skipped_shape_mismatch,
        "mean_tensor_mse": math.fsum(tensor_mse_values) / compared_count,
        "max_tensor_mse": max(tensor_mse_values),
        "element_weighted_mse": total_sse / total_numel,
        "l2_norm": math.sqrt(total_sse),
        "tensors": tensor_rows,
    }


def _summarize_kv_delta_candidates(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    compared = {
        version: candidate
        for version, candidate in candidates.items()
        if candidate.get("status") == "compared" and candidate.get("element_weighted_mse") is not None
    }
    if not compared:
        return {
            "status": "no_comparable_candidates",
            "best_candidate_by_element_weighted_mse": None,
            "anchor_effect": {},
        }
    best = min(compared, key=lambda version: compared[version]["element_weighted_mse"])
    anchor_effect: dict[str, Any] = {}
    reference = compared.get("v9")
    if reference is not None and reference.get("element_weighted_mse"):
        reference_mse = float(reference["element_weighted_mse"])
        for version in ("v10", "v11"):
            candidate = compared.get(version)
            if candidate is None:
                continue
            candidate_mse = float(candidate["element_weighted_mse"])
            ratio = candidate_mse / reference_mse
            if ratio <= 0.9:
                status = "reduced_vs_v9"
            elif ratio <= 1.1:
                status = "not_materially_changed_vs_v9"
            else:
                status = "increased_vs_v9"
            anchor_effect[version] = {
                "status": status,
                "ratio_to_v9": ratio,
                "reference_v9_element_weighted_mse": reference_mse,
            }
    return {
        "status": "compared",
        "best_candidate_by_element_weighted_mse": best,
        "ranked_candidates": sorted(compared, key=lambda version: compared[version]["element_weighted_mse"]),
        "anchor_effect": anchor_effect,
    }


def _feedback_source_bucket(path: Path) -> str:
    stem = path.stem
    for prefix in ("00_fixed_gate", "10_v5_text", "20_hard_negative", "30_general"):
        if stem.startswith(prefix):
            return prefix
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def _release_gate_failure_reasons(
    *,
    promotion: dict[str, Any],
    baseline_candidate: dict[str, Any],
    min_heldout_readable: int,
    max_heldout_failed: int,
    min_general_cases: int,
) -> list[str]:
    reasons: list[str] = []
    recommendation = promotion.get("recommendation", {})
    promote_candidate = recommendation.get("promote_candidate")
    if recommendation.get("status") != "protect_baseline" and promote_candidate is None:
        reasons.append("promotion_recommendation_not_protect_baseline")

    fixed_gate = baseline_candidate.get("fixed_gate", {})
    if fixed_gate.get("count") != 6:
        reasons.append("fixed_gate_count_not_six")
    if fixed_gate.get("mean_mse") is None:
        reasons.append("fixed_gate_metrics_missing")

    heldout = baseline_candidate.get("heldout")
    heldout_review = baseline_candidate.get("heldout_review")
    if not heldout:
        reasons.append("heldout_metrics_missing")
    elif heldout.get("case_count", 0) < 64:
        reasons.append("heldout_case_count_below_required")
    if not heldout_review:
        reasons.append("heldout_review_missing")
    else:
        counts = heldout_review.get("counts", {})
        if counts.get("readable", 0) < min_heldout_readable:
            reasons.append("heldout_readable_below_minimum")
        if counts.get("failed", 0) > max_heldout_failed:
            reasons.append("heldout_failed_above_maximum")

    general_scene = baseline_candidate.get("general_scene")
    if not general_scene:
        reasons.append("general_scene_metrics_missing")
    elif general_scene.get("case_count", 0) < min_general_cases:
        reasons.append("general_scene_case_count_below_required")

    general_review = None
    for artifact in baseline_candidate.get("required_artifacts", []):
        if artifact.get("name") == "general_scene_review":
            general_review = _read_optional_json(artifact.get("path"))
            break
    if general_review is None:
        reasons.append("general_scene_review_missing")
    elif general_review.get("decision") not in {"pass_general_scene_smoke_expanded_50", "pass"}:
        reasons.append("general_scene_review_not_pass")
    if promote_candidate is not None:
        candidate = promotion.get("candidates", {}).get(str(promote_candidate), {})
        reasons.extend(
            _promotion_candidate_next_gate_failure_reasons(
                version=str(promote_candidate),
                candidate=candidate,
                min_heldout_readable=min_heldout_readable,
                max_heldout_failed=max_heldout_failed,
                min_general_cases=min_general_cases,
            )
        )
    return reasons


def _promotion_candidate_next_gate_failure_reasons(
    *,
    version: str,
    candidate: dict[str, Any],
    min_heldout_readable: int,
    max_heldout_failed: int,
    min_general_cases: int,
) -> list[str]:
    reasons: list[str] = []
    heldout = candidate.get("heldout")
    heldout_review = candidate.get("heldout_review")
    if not heldout:
        reasons.append(f"candidate_heldout_metrics_missing:{version}")
    elif heldout.get("case_count", 0) < 64:
        reasons.append(f"candidate_heldout_case_count_below_required:{version}")
    if not heldout_review:
        reasons.append(f"candidate_heldout_review_missing:{version}")
    else:
        counts = heldout_review.get("counts", {})
        if counts.get("readable", 0) < min_heldout_readable:
            reasons.append(f"candidate_heldout_readable_below_minimum:{version}")
        if counts.get("failed", 0) > max_heldout_failed:
            reasons.append(f"candidate_heldout_failed_above_maximum:{version}")

    general_scene = candidate.get("general_scene")
    if not general_scene:
        reasons.append(f"candidate_general_scene_metrics_missing:{version}")
    elif general_scene.get("case_count", 0) < min_general_cases:
        reasons.append(f"candidate_general_scene_case_count_below_required:{version}")

    general_review = None
    for artifact in candidate.get("required_artifacts", []):
        if artifact.get("name") == "general_scene_review":
            general_review = _read_optional_json(artifact.get("path"))
            break
    if general_review is None:
        reasons.append(f"candidate_general_scene_review_missing:{version}")
    elif general_review.get("decision") not in {"pass_general_scene_smoke_expanded_50", "pass"}:
        reasons.append(f"candidate_general_scene_review_not_pass:{version}")
    return reasons


def _v9_training_gate_decision(*, release_status: str, promotion: dict[str, Any]) -> dict[str, Any]:
    recommendation = promotion.get("recommendation", {})
    if release_status != "pass":
        return {
            "status": "blocked_until_release_gate_passes",
            "reason": "baseline release gate evidence is incomplete or failing",
        }
    if recommendation.get("promote_candidate") is None:
        return {
            "status": "blocked_until_objective_redesign",
            "reason": "release gate protects v5 and no candidate beat fixed image/text gates",
        }
    return {
        "status": "candidate_requires_heldout_and_general_eval",
        "reason": "a candidate beat fixed gate and needs next promotion gates before training expansion",
    }


def _hardlink_command(source: Path, target: Path) -> str:
    return (
        f"if (Test-Path -LiteralPath \"{target}\") {{ Remove-Item -LiteralPath \"{target}\" }}; "
        f"New-Item -ItemType HardLink -Path \"{target}\" -Target \"{source}\" | Out-Null"
    )


def _json_path(path: Path) -> str:
    return str(path).replace("\\", "/")
