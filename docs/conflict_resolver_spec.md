# GEMMANIMA Conflict Resolver Spec

## Principle

Image reference and text instruction have different authority:

- Image reference answers: who or what should this look like?
- Text instruction answers: what should happen?

When they conflict, the engine must not guess. High-severity unresolved conflicts block generation and return a clarification request.

## Blocking Conditions

Generation should stop when:

- Identity conflict severity is high or critical.
- User intent is ambiguous.
- Reference image and text point to different characters.
- Preserve and modify requests conflict.
- Unsafe content risk is unresolved.

## Initial Contract

The first implementation uses `ConflictReport` from `gemmanima.core.protocol`:

- `preserve`
- `modify`
- `conflicts`
- `fields`
- `severity`
- `requires_user_confirmation`
- `proposed_questions`

`ConflictReport.blocks_generation()` is the renderer gate.

## Implemented First Gate

The current resolver detects high-severity conflicts for:

- `hair_color`
- `outfit`
- `style`
- `identity`
- `unsafe_content`

The resolver now accumulates multiple conflicts into one `ConflictReport`. The conductor writes an `ask_clarify` manifest, records the conflict under `renderer.conflict`, exposes the same conflict in the API response, and does not call the renderer.

Clarification replies are handled per field. A short follow-up such as `change it to black hair` can resume the blocked image request, clear the hair-color conflict, and keep unrelated unresolved conflicts blocked. The local GUI sends recent conversation history so the stateless API can resume the same blocked request.

The GUI now exposes preserve/change buttons for conflict fields. Those buttons submit ordinary clarification messages through the same conversation-history path, rather than adding a separate resolver API.

Next expansion targets are richer protocol extraction and more specific per-field question wording.
