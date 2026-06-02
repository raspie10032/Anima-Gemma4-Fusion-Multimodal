# Chat Mode Contract Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Gemma text chat obey an explicit mode and output contract selected by call arguments.

**Architecture:** Keep routing unchanged: `gemmanima.api.handle_chat_payload` remains the HTTP/API boundary and `gemmanima.modules.tipo_runtime.run_tipo_text_chat` remains the llama-cli execution boundary. Add a prompt harness that is composed after the language harness and before history, then echo normalized contract metadata in API responses.

**Tech Stack:** Python 3.13, pytest, existing `tipo_runtime.py` subprocess runner tests.

---

### Task 1: Contract Harness Tests

**Files:**
- Modify: `tests/test_tipo_runtime.py`
- Modify: `tests/test_api.py`

- [ ] Add tests proving `build_chat_contract_harness("tag_request")` requires canonical English Danbooru tags, no prose, no markdown, and no translation.
- [ ] Add tests proving `run_tipo_text_chat(..., chat_mode="image_generation_request")` injects the image-generation JSON contract into the final llama prompt and returns normalized contract metadata.
- [ ] Add tests proving `handle_chat_payload({"task": "chat", "chat_mode": "status_question"})` passes the selected mode into `run_tipo_text_chat` and exposes `chat_mode` / `output_contract` in the response.
- [ ] Run the new targeted tests and verify they fail because the functions/fields are not implemented yet.

### Task 2: Contract Harness Implementation

**Files:**
- Modify: `gemmanima/modules/tipo_runtime.py`
- Modify: `gemmanima/api.py`

- [ ] Add `normalize_chat_mode(chat_mode)` with aliases for `chat`, `general`, `tags`, `tag`, `image`, `generate`, `status`, `logs`, `file`, and `checkpoint`.
- [ ] Add `build_chat_contract_harness(chat_mode)` returning strict, mode-specific output rules.
- [ ] Update `_chat_prompt` and `run_tipo_text_chat` to accept `chat_mode`, include the contract harness, and return `chat_mode` plus `output_contract`.
- [ ] Update `handle_chat_payload` to read `chat_mode` / `intent` from payload, pass it through, and echo the normalized fields in success and failure responses.

### Task 3: Verification

**Files:**
- Test only.

- [ ] Run the targeted contract tests.
- [ ] Run the existing relevant suite: `python -m pytest tests/test_tipo_runtime.py tests/test_api.py tests/test_cli.py tests/test_server_gui.py -q`.
- [ ] Run a quick code review pass for prompt contradictions with the language harness.
