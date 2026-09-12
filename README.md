# bert-masked-lm-pipeline

DIMER pipeline scaffold for **google-bert/bert-base-uncased** — Masked-language modelling (fill-mask) + sentence embeddings.

| | |
|---|---|
| Upstream model | [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) |
| Pinned revision | `86b5e0934494bd15c9632b12f734a8a67f723594` (resolved 2026-09-12) |
| Upstream license | `apache-2.0` (verified on the Hub 2026-09-12; re-check at the pinned revision before release) |
| Weight files to stage | `model.safetensors` |
| Status | scaffold only — no pipeline code yet |

Weights are staged under `weights/` and are git-ignored. This repository follows the
MODEL_CARD_SPEC 1.1 / NOTEBOOK_SPEC 1.0 conventions used by the other `*-pipeline` repos.
