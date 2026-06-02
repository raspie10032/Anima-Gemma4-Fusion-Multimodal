# GEMMANIMA Image State Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a first image-state translator that maps existing image embeddings into the same Anima conditioning space already reached by the text translator.

**Architecture:** Treat `kv_proj_text_delta_300k_from_epoch1_a0p35.pt` as the working text-conditioning anchor. Build image-state samples from `hiddenstage_multimodal_planner_anima_v2.jsonl`, cache Anima/Qwen conditioning targets from the visible prompt, and train a small cross-attention image-state translator from `image_embed_pre` tensors to `[t5_tokens, 1024]` target conditioning.

**Tech Stack:** Python, PyTorch, existing Anima target cache script, JSONL manifests, 4070-only GPU commands.

---

### Task 1: Image-State Training Manifest

**Files:**
- Create: `gemmanima/training/image_state_conditioning.py`
- Modify: `gemmanima/cli.py`
- Test: `tests/test_image_state_conditioning.py`

- [x] **Step 1: Write subset builder**

Normalize multimodal rows into records with `idx`, `text`, `visible_prompt`, `image`, and `image_embed_pre`.

- [x] **Step 2: Write training plan builder**

Emit 4070-only commands for subset writing, Anima target caching, and image-state translator training.

### Task 2: Image-State Translator Trainer

**Files:**
- Create: `scripts/train_image_state_translator.py`

- [x] **Step 1: Implement first trainer**

Use `image_embed_pre` as memory tokens and `t5_ids` as query tokens, then train against Anima conditioning targets.

- [x] **Step 2: Save checkpoint contract**

Checkpoint includes model state, config, text translator anchor path, train MSE, val MSE, and example counts.

### Task 3: First Artifact Generation

**Files:**
- Create: `reports/image_state_conditioning_v1/image_state_conditioning_v1_plan.json`
- Create: `reports/image_state_conditioning_v1/subset_10k.jsonl`

- [x] **Step 1: Generate the plan and subset**

Use existing multimodal manifest and keep all GPU commands 4070-only.
