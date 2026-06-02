# GEMMANIMA Full Structure Fast Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn ad-hoc candidate experiments into a repeatable candidate workflow status surface.

**Architecture:** Keep model defaults protected and represent each candidate as a non-mutating workflow record. Reuse existing fixed6 metrics, general-quality contact sheets, and 4070-only render plan commands instead of inventing a second evaluation stack.

**Tech Stack:** Python CLI, JSON reports, pytest, existing GEMMANIMA training/eval modules.

---

### Task 1: Candidate Workflow Status Surface

**Files:**
- Create: `gemmanima/training/candidate_workflow.py`
- Modify: `gemmanima/cli.py`
- Test: `tests/test_candidate_workflow.py`

- [x] **Step 1: Add a pure-Python status builder**

Create a module that reads optional fixed6, smoke, and general-quality artifacts, checks checkpoint existence, preserves v5 by default, and returns deterministic JSON.

- [x] **Step 2: Add a CLI command**

Expose `candidate-workflow-status` with `--candidate-name`, `--checkpoint`, optional artifact paths, `--output`, and `--json`.

- [x] **Step 3: Add focused tests**

Cover rejected fixed6 state, missing artifacts, CLI JSON output, and non-mutating safety fields.

- [x] **Step 4: Verify**

Run focused pytest plus compileall before using it on the current PoC adapter.

### Task 2: Regression-To-Objective Manifest

**Files:**
- Modify: `gemmanima/training/candidate_workflow.py`
- Modify: `gemmanima/cli.py`
- Test: `tests/test_candidate_workflow.py`

- [x] **Step 1: Convert fixed6 regressions into replay records**

Read `regressions[]` or `cases[]`, preserve the worst deltas, and assign deterministic replay weights.

- [x] **Step 2: Emit the next training objective**

Write a JSON manifest that treats the PoC checkpoint as a style donor while keeping v5 as the protected baseline.

- [x] **Step 3: Add CLI and tests**

Expose `candidate-objective-manifest` and verify it writes deterministic JSON without touching GPU.

### Task 3: Promotion Bundle Gate

**Files:**
- Modify: `gemmanima/training/candidate_workflow.py`
- Modify: `gemmanima/cli.py`
- Test: `tests/test_candidate_workflow.py`

- [x] **Step 1: Summarize promotion readiness**

Read candidate workflow status and required artifact paths, then decide whether default update is allowed.

- [x] **Step 2: Preserve rollback information**

Always include protected baseline checkpoint and default-change instructions.

- [x] **Step 3: Add CLI and tests**

Expose `candidate-promotion-bundle` and verify rejected fixed6 candidates are blocked.
