# BERT Masked-LM Pipeline

DIMER inference wrapper for **`google-bert/bert-base-uncased`** — masked-language modelling (fill-mask) and sentence embeddings — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot.

## Upstream alignment

- Model: `google-bert/bert-base-uncased` (BERT-Base, 12 layers, hidden 768, uncased English, 30 522-token WordPiece vocab)
- Revision: `86b5e0934494bd15c9632b12f734a8a67f723594`
- Upstream weight license: Apache-2.0
- Upstream task: masked language modelling (the raw checkpoint is intended for fine-tuning upstream; this repository exposes the frozen encoder only)
- Repository adaptation: **none**; inference only

## Quick start

```python
from bert_masked_lm_pipeline import BERTMaskedLMPipeline

pipe = BERTMaskedLMPipeline.from_pretrained()          # cuda:0 if available, else cpu
filled = pipe.fill_mask("The capital of France is [MASK].", top_k=5)
print(filled["candidates"][0]["token"], filled["candidates"][0]["score"])

vectors = pipe.embed(["The cat sat on the mat.", "A dog lay on the rug."], pooling="mean")
print(vectors["dim"], len(vectors["embeddings"]))
```

`fill_mask` requires exactly one `[MASK]` and returns a softmax-ranked candidate list (not calibrated probabilities). `embed` returns L2-normalised 768-d vectors pooled from `[CLS]` (`pooling="cls"`) or the attention-masked mean of the last layer (`pooling="mean"`). Ceilings: `MAX_TEXT_CHARS = 4000`, `MAX_TEXT_TOKENS = 512` (rejected, not truncated), `MAX_BATCH = 64`, `MAX_TOP_K = 100`.

## Weights layout

```
weights/bert-base-uncased/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + sha256 (verified on every load)
  config.json                # BertForMaskedLM architecture
  tokenizer.json, tokenizer_config.json, vocab.txt
  model.safetensors          # 440449768 bytes, git-ignored
  LICENSE, README.md, coreml/  # upstream files listed in the manifest; not used by the loader
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` and refuses to load if any manifest file is missing or its SHA-256 differs. On a fresh clone (manifest committed, weights git-ignored) `from_pretrained(allow_download=True)` fetches only the missing files at the pinned revision; the default is to refuse. Without any manifest, `allow_download=True` loads from the Hub with `revision=86b5e0934494bd15c9632b12f734a8a67f723594`.

## Tests and smoke

```
pip install -e . --no-deps
pytest -q -o addopts= tests      # offline, no weights needed
```

Smoke (loads the verified snapshot on CPU; measured numbers are in `MODEL_CARD.md` → Runtime):

```python
from bert_masked_lm_pipeline import BERTMaskedLMPipeline

pipe = BERTMaskedLMPipeline.from_pretrained(device="cpu")
print(pipe.fill_mask("The capital of France is [MASK].")["candidates"][0])
```

## Tutorials

None yet. This release is the card pass; a `NOTEBOOK_SPEC` 1.0 `TASK-INFERENCE` tutorial is a follow-up.

## Release status

**Candidate / source-complete.** Pipeline package, offline unit tests, one CPU smoke run and the MODEL_CARD_SPEC 1.1 card exist; no tutorial notebook and no clean-runtime execution evidence yet. See `STATUS.md`.

## Documents

- [`MODEL_CARD.md`](MODEL_CARD.md) — MODEL_CARD_SPEC 1.1 card
- [`docs/WEIGHTS.md`](docs/WEIGHTS.md) — weight provenance and hosting
- [`STATUS.md`](STATUS.md) — release status

## Licensing

Repository code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md`.
