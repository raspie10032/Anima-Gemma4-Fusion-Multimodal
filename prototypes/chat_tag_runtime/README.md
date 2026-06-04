# Gemma4 Hidden-State + Anima/GEMMANIMA Runtime Prototype

This prototype exists for one purpose:

**Build one local Gemma4 + Anima/GEMMANIMA model.**

It must preserve Gemma4 chat and multimodal image vision, then feed Gemma4
hidden states directly into Anima synthesis so chat, image understanding,
image-to-tags, and image generation live in one runtime.

## Hard Boundary

- Do not use Codex image generation.
- Do not treat ComfyUI, NovelAI, or another server as a later external backend.
- Do not collapse normal chat and image tag output into one contaminated mode.
- Do not hide planner model defects with runtime tag blacklists or bias filters.
- Do not make prompt-only or tag-only handoff the final image generation path.

The first test file, `test_00_project_purpose.py`, enforces this boundary.

## Architecture

- Target architecture: Gemma4 hidden-state to Anima/GEMMANIMA synthesis
- Current chat stand-in: `llama-cli.exe` + `gemma-4-E2B-it-heretic-ara-Q4_K_M.gguf`
- Current runtime branch: quantized `llama.cpp`
- Future branch point: unquantized Transformers/native runtime, if the deployment environment can afford it
- Current image planner target: `llama-completion.exe` + quantized Gemma4 base GGUF + llama.cpp-compatible TIPO f16 LoRA GGUF
- Current image planner default: merged `gemma4-tipo-ko-v2-Q4_K_M.gguf`
- Required vision module 1: image understanding, for natural-language visual understanding and hidden-state conditioning
- Required vision module 2: image-to-tags, for Danbooru/TIPO-style tag extraction
- Current vision stand-ins: `llama-mtmd-cli.exe` + `gemma4-tipo-vision-Q4_K_M.gguf` + matching mmproj, split by module contract even if early smoke assets share the same runtime
- Image generation target: Gemma4 hidden states -> Anima/GEMMANIMA synthesizer
- GPU pinning: each subprocess gets `CUDA_VISIBLE_DEVICES=0`, so llama.cpp sees the RTX 4070 Ti SUPER as `CUDA0`.

The role-specific GGUF files above are temporary smoke-test stand-ins. The
design target is not a permanent prompt router; it is one Gemma4 model whose
chat and multimodal hidden states condition Anima/GEMMANIMA synthesis directly.
For the current quantized branch, planner specialization should be an attached
LoRA module over the shared quantized Gemma4 base. The existing PEFT
`adapter_model.safetensors` is source material; llama.cpp `--lora` expects a
GGUF LoRA adapter. Keep the planner adapter at f16 by default; the q8_0 adapter
saved little space and was slower in local smoke tests, while q6/q4 LoRA
quantization is not supported by the current standard converter path.

Planner quality problems, such as repeated default hair or eye tags, are model
training issues. The runtime should pass planner output through a simple parser
and merger, not patch it with blacklist rules.

Image understanding and image-to-tags are separate modules. The conductor may
call both for a complex request, but it must not treat tag extraction as a
replacement for visual understanding.

With `--use-planner`, the runtime only tests the temporary TIPO stand-in. This
is not the final generation path.

## Commands

```powershell
python prototypes\chat_tag_runtime\prototype.py chat "Say hello in one short sentence."
python prototypes\chat_tag_runtime\prototype.py tag "D:\path\to\image.png"
python prototypes\chat_tag_runtime\prototype.py image "blue-eyed cat-ear wizard at a glowing desk"
python prototypes\chat_tag_runtime\prototype.py route --task auto --message "draw a blue-eyed cat-ear wizard"
python prototypes\chat_tag_runtime\prototype.py models --json
python prototypes\chat_tag_runtime\prototype.py health --json
python prototypes\chat_tag_runtime\prototype.py serve --port 8787
```

Check the Anima/GEMMANIMA hidden-state synthesizer slot:

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
python prototypes\chat_tag_runtime\prototype.py image "1girl, blue eyes, cat ears, wizard, glowing desk" --generate --json
python prototypes\chat_tag_runtime\prototype.py image "green-eyed mage in a red dress" --use-planner --generate --json
```

HTTP server endpoints:

- `GET /health`
- `POST /route`

Example `/route` payload:

```json
{
  "task": "auto",
  "message": "draw a blue-eyed cat-ear wizard"
}
```

Expected image route result:

```json
{
  "mode": "image",
  "status": "generator_required",
  "generator": "anima-gemmanima-image-generator",
  "external_backend": false,
  "planner_contract": "TIPO Partial tags continuation"
}
```

## Current State

The image path now creates a deterministic builtin generation job, output slot,
and TIPO `Partial tags:` planner prompt for smoke tests. The actual image
producer must consume Gemma4 hidden states through an Anima/GEMMANIMA synthesis
bridge. A plain SDXL checkpoint is not the default generator for this prototype.
