# GemmAnima Composite Prototype License Notice

This repository is a prototype source tree and runtime scaffold for GemmAnima.
It is designed to work with external Gemma and Anima model assets plus
GemmAnima adapter/checkpoint files.

This notice is not legal advice. Verify current upstream terms before
redistribution, hosted use, commercial use, production use, or public model
release.

## Composite Runtime Posture

GemmAnima does not provide one simple permissive license for the complete
runtime stack. The usable runtime is a composite of:

- Gemma/GGUF base model terms and required notices.
- Anima base model terms.
- NVIDIA Open Model License terms referenced by Anima where applicable.
- GemmAnima adapter, bridge, projector, and runtime metadata notices.
- Dataset/source restrictions for any adapter training data.

Use the most restrictive applicable terms across the loaded stack.

## Anima Dependency

The current image-generation path depends on Anima. The upstream Anima page
currently presents Anima under the CircleStone Labs Non-Commercial License and
states that the model and derivatives are usable only for non-commercial
purposes. It also states that Anima is a derivative model of
Cosmos-Predict2-2B-Text2Image and references NVIDIA Open Model License terms
where applicable.

Because of that dependency, treat the GemmAnima v0.1.0 prototype runtime and
adapter/checkpoint bundle as non-commercial, non-production, restricted-use
prototype material unless a separate license review and any required upstream
permissions say otherwise.

## Gemma Dependency

Gemma/GGUF assets are external base-model dependencies and are not mirrored in
this source repository. Users should download those assets from the original
model pages and comply with their current license, terms, distribution notices,
and use restrictions.

GemmAnima logic that is designed around Gemma behavior should also be treated as
Gemma-dependent project material where applicable. This includes model-facing
conversation harnesses, routing ideas, prompt and output-contract logic,
LoRA/mmproj usage patterns, hidden-state bridge concepts, and adapter behavior
that depends on Gemma or Gemma-derived runtime behavior. Do not treat those
Gemma-dependent ideas or logic as separately relicensing or bypassing Gemma
terms.

## Source Repository

The source code in this repository is provided so the prototype can be inspected,
tested, and run locally with separately obtained model assets. This notice does
not relicense Gemma, Anima, NVIDIA Cosmos-derived assets, or any upstream base
model.

Do not remove upstream notices. Do not imply that GemmAnima is an official
Google, CircleStone Labs, Comfy Org, or NVIDIA product.

## Files Not Included

Large base model weights, generated images, run artifacts, caches, and private
local paths should not be committed to this repository.
