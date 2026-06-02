# GEMMANIMA Architecture Summary

GEMMANIMA is a local multimodal conditioning bridge, not a prompt helper. The architecture keeps strong existing models frozen and trains only the smallest necessary translation layers.

## Target Flow

```text
Text / Image / Audio / Session State
-> Gemma-side understanding
-> GEMMANIMA Protocol
-> Conditioning Translator
-> Anima-compatible ConditioningBundle
-> Renderer
-> Image + RunManifest
```

## Current Repo Flow

```text
User text
-> GemmAnimaConductor
-> ContextRelevanceFilter
-> GemmaPlannerAdapter
-> HiddenStageExit
-> AnimaRendererAdapter
-> ManifestStore
```

## Current Protocol Foundation

`GemmanimaProtocol` now sits on `ContextCapsule` and flows into `HiddenStageExit`. The conductor also runs a `ConflictResolver` before renderer execution, so high-severity unresolved conflicts can return a clarification request without producing an image.

Protocol, conditioning bundle, conflict report, run manifest, and cache-build payloads now have schema validation helpers. Run manifests include observed-only `conditioning_metrics` with nullable `run_conditioning_mse`, nullable `bridge_val_mse`, and `measured=false` when no run-level teacher/student conditioning distance exists.

The API exposes `clarification_required` and `conflict` fields, and the local GUI highlights blocked conflict responses above the raw JSON result. The GUI also sends recent conversation history, allowing short clarification replies to resume the blocked image request instead of falling back to chat routing. Conflict preserve/change buttons fill in ordinary clarification messages, so the UI path uses the same conversation-history contract as typed replies.

Protocol extraction now sits behind `ProtocolParser`. The current parser is still lightweight and heuristic-based, but the boundary is ready for a model/adapter-backed parser without touching `ContextRelevanceFilter`.
