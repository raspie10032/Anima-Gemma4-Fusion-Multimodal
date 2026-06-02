# GEMMANIMA Data Policy

## Lineage Classes

`internal_experimental`:

- Current private or unverified datasets may be used for performance probing.
- Raw dataset is not published.
- Checkpoints are not published.
- Results may be private or selectively shared only when safe.

`clean_public`:

- Source, license, and safety status are verified.
- Public release is allowed.
- Model card and dataset card are required.
- This is the only lineage eligible for public checkpoints.

## Manifest Requirements

Every dataset or run manifest must record:

- `lineage`
- `dataset_id`
- `dataset_release_status`
- `public_release_allowed`
- `contains_unverified_data`
- `contains_unsafe_data`
- `source_license_status`
- `intended_use`

## Current Policy

The 160k image-text dataset is treated as `internal_experimental` until provenance and safety review prove otherwise.
