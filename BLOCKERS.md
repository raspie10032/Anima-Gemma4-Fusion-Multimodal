# GEMMANIMA Blockers

## Current Blockers

- Exact Anima conditioning interface shape, dtype, and sequence length must stay audited as renderer integration evolves.
- Gemma hidden-state source layer selection needs to be fixed before translator training expands.
- Gemma vision projection extraction point is not standardized yet.
- The 160k image-text dataset is internal experimental lineage; provenance and public release status are not clean.
- Unsafe-data review is incomplete.
- A clean public lineage dataset is required before public checkpoint release.
- Text rendering reinforcement data needs separate lineage tracking.
- VisionReferenceTranslator teacher path is still uncertain.

## Current Mitigation

- Keep Anima, VAE, base model, Gemma base, and vision tower frozen by default.
- Keep public artifacts limited to code, configs, schemas, manifests, logs, safe results, model cards, and dataset-card templates.
- Treat internal experimental checkpoints as non-public unless lineage is upgraded.
