---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: fill-mask
base_model: google-bert/bert-base-uncased
---

# BERT base uncased (DIMER package v0.1.0) — Masked Language Model (Fill-Mask & Sentence Embeddings)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google--bert%2Fbert--base--uncased-ffcc4d?style=flat)](https://huggingface.co/google-bert/bert-base-uncased)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-google--research%2Fbert-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/bert)
[![arXiv Paper](https://img.shields.io/badge/arXiv-1810.04805-b31b1b.svg)](https://arxiv.org/abs/1810.04805)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pipeline](https://img.shields.io/badge/Pipeline-bert--masked--lm--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/bert-masked-lm-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This release ships no tutorial notebook (`tutorials/` is absent). The package is exercised through its test suite (`tests/`) and the run instructions in the README; a `NOTEBOOK_SPEC` 1.0 `TASK-INFERENCE` notebook is a follow-up, not a claim this card makes.

---

###### Description

`google-bert/bert-base-uncased` is the original BERT-Base uncased checkpoint released by Google Research (Devlin et al., arXiv:1810.04805), redistributed on the Hub by the Hugging Face team and pinned here to revision `86b5e0934494bd15c9632b12f734a8a67f723594`. It is a bidirectional Transformer encoder: 12 layers, hidden size 768, 12 attention heads, intermediate size 3072, absolute position embeddings up to 512 positions and a 30 522-entry WordPiece vocabulary (snapshot `config.json`), about 110 M parameters (upstream README table). It was pre-trained with two self-supervised objectives — masked language modelling (15 % of tokens masked, predict them) and next-sentence prediction — on lower-cased English text. At inference this package runs the encoder once per call and does one of two things: `fill_mask` reads the output-vocabulary logits at the single `[MASK]` position through the `BertForMaskedLM` head and ranks them by softmax; `embed` returns the encoder's last hidden states pooled to one 768-d vector per text. Nothing is fine-tuned, adapted or conditioned in this repository, and the next-sentence-prediction head and pooler weights present in the checkpoint are not loaded (the loader reports them unused). What this repository adds is packaging: the `BERTMaskedLMPipeline` class in `src/bert_masked_lm_pipeline/pipeline.py`, digest verification of the local snapshot (`verify_snapshot`, `stage_missing_files`), input validation against named ceilings, and a fixed output contract.

#### Intended Use and Limitations

###### Primary Intended Uses

Two tasks over short English text. Fill-mask: input one string containing exactly one `[MASK]` token; output the `top_k` (default 5, ceiling `MAX_TOP_K = 100`) vocabulary candidates for that position with their softmax scores and the filled-in sequence — the probe the upstream card itself demonstrates, useful for lexical substitution, cloze-style diagnostics, spelling or word-choice suggestion, and inspecting what a masked language model has absorbed. Embed: input 1–`MAX_BATCH = 64` strings; output one L2-normalised 768-d vector each, pooled either from the `[CLS]` position (`pooling="cls"`) or as the attention-masked mean of the last layer (`pooling="mean"`), for nearest-neighbour search, clustering, deduplication and as frozen features for a downstream classifier trained by the operator. In a larger system this pipeline is a classic baseline and a feature extractor, not a decision engine; the upstream card states the raw model "is mostly intended to be fine-tuned on a downstream task", and fine-tuning is not exposed here.

###### Primary Intended Users

Machine-learning engineers, NLP researchers and application developers integrating a well-understood English encoder into research prototypes, internal enterprise tooling or the DIMER model workbench. The pipeline assumes its users understand that the model is uncased English only (accents are stripped and case is lost by the tokenizer, `do_lower_case: true`), that a fill-mask softmax score is a ranking signal and not a calibrated probability, that raw BERT embeddings — unlike models trained with a contrastive sentence objective — give only moderate semantic similarity and must be evaluated on the operator's own retrieval or clustering task before use, and that the documented gender and occupation bias in the upstream card's own examples will surface in both outputs. It is not designed for hobbyist "point and trust" use.

###### Out-of-scope use cases

1. **Capability boundary:** not text generation, translation, summarisation or question answering — the model is an encoder with no decoder (use the sibling `gpt2-text-generation-pipeline` or `t5-small-text2text-pipeline` for generation); not multi-token span filling — exactly one `[MASK]` is accepted and a sentence with zero or two masks is rejected with `ValueError`; not next-sentence prediction, whose head is not loaded; not a purpose-trained sentence-embedding model (the `qwen3-embedding-pipeline` sibling covers retrieval-grade embeddings); not fine-tuning.
2. **Input boundary:** `str` only (`TypeError` otherwise); empty or whitespace-only text rejected; more than `MAX_TEXT_CHARS = 4000` characters rejected before tokenisation and more than `MAX_TEXT_TOKENS = 512` WordPiece tokens (including `[CLS]`/`[SEP]`) rejected after it — inputs are never silently truncated; `embed` batches outside 1–64 rejected; `pooling` other than `cls`/`mean` rejected. Non-English text, code, and text whose meaning depends on casing degrade without any warning from the pipeline.
3. **Decision boundary:** not for autonomous or high-impact decisions — screening people, moderating content, ranking applicants — without a human reviewing the output and a locally measured error rate; not for inferring attributes of individuals from text.

#### Factors

###### Groups

The pipeline is human-centric in the sense that its inputs are natural-language text about, and often written by, people, and its outputs can name people, roles and attributes. The upstream card documents group-level behaviour directly: with the prompt "The man worked as a [MASK]." the top candidates are carpenter, waiter, barber, mechanic, salesman, while "The woman worked as a [MASK]." yields nurse, waitress, maid, prostitute, cook (upstream README "Limitations and bias", scores 0.097–0.038 and 0.220–0.030 respectively), and the card states "This bias will also affect all fine-tuned versions of this model." Neither the upstream card nor this repository reports any broader group audit of the BookCorpus and English Wikipedia pre-training corpora, and this repository ran no fairness evaluation. The obligation therefore transfers to the operator: before any deployment that touches people, run a template-based probe (paired prompts differing only in a group term, compared on the ranked candidates and on embedding distances) and a task-level evaluation stratified by the groups relevant to the application, and treat a material gap as a blocker.

###### Instrumentation

The training data was captured by no physical sensor: it is text. Upstream discloses BookCorpus ("11,038 unpublished books") and English Wikipedia "excluding lists, tables and headers", lower-cased and tokenised with WordPiece into a 30 522-entry vocabulary, formed into `[CLS] A [SEP] B [SEP]` pairs under 512 tokens (upstream README "Training data" and "Preprocessing"). The instrument characteristics that reach the model are therefore the text-collection and cleaning steps — web scraping of self-published fiction, Wikipedia markup stripping, sentence splitting — none of which are documented beyond those sentences. At inference the only instrument is the tokenizer shipped in the snapshot (`tokenizer.json`, `vocab.txt`); unusual Unicode, OCR errors, code, URLs and out-of-vocabulary words fragment into sub-word pieces or `[UNK]`, and the pipeline reports the token count (`n_tokens`) but does not detect encoding drift, language change or OCR noise.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `numpy==2.5.3` (exact pins in `pyproject.toml`), float32. `from_pretrained(device=None)` picks `cuda:0` when available, else CPU; this repository's smoke ran on CPU only (Windows venv, `CUDA_VISIBLE_DEVICES=-1`, `device="cpu"`): loading and digest-verifying the 441 MB snapshot took 4.34 s, one `fill_mask` call 0.13 s, one two-sentence `embed` call 0.02 s. The CUDA path is untested in this repository. Data environment: inputs are assumed to be lower-case-tolerant modern English prose resembling books and encyclopaedia text; the model's behaviour on social-media text, domain jargon (clinical, legal, code), other languages, or text older or newer than the pre-training corpus is not measured here and is expected to degrade — the vocabulary was frozen at pre-training time, so newer terms fragment into pieces.

#### Metrics

###### Performance Measures

The pipeline reports no performance measure. `fill_mask` emits `score` (a softmax over the 30 522-token vocabulary at the masked position) and `embed` emits unit-norm vectors; neither is a prediction against a ground truth the package could score, so no accuracy, pseudo-perplexity or similarity metric is computed and no helper is shipped. To evaluate fill-mask the caller must supply a labelled cloze set (sentence, mask position, gold token) and compute rank-based hits, for example top-1 and top-5 hit rate; to evaluate embeddings the caller must supply a similarity or retrieval benchmark with relevance labels and compute a ranking metric such as recall@k or Spearman correlation against human judgements. Upstream reports GLUE test results after fine-tuning (average 79.6; upstream README "Evaluation results"); those describe fine-tuned derivatives, not this frozen pipeline, which has not reproduced them.

###### Decision thresholds

`fill_mask` applies an implicit decision rule: candidates are ordered by descending softmax score and cut at `top_k`; the first entry is the argmax and no minimum score is required, so a nonsensical sentence still yields a ranked list. `embed` applies no threshold at all — it returns vectors, and any similarity cut-off (for duplicate detection, clustering, retrieval acceptance) is deliberately left to the caller because the raw cosine scale of BERT embeddings is not calibrated to any notion of "same meaning". No acceptance threshold was set during development. A deployment that needs one must choose it on its own labelled data, trading the cost of a wrong confident substitution or a false duplicate (false positive) against a missed candidate (false negative) for its application.

###### Approaches to uncertainty and variability

No metric is reported, so there is no estimation procedure or dispersion to state. Inference is deterministic given the same weights, device and library versions: dropout is disabled by `model.eval()`, there is no sampling, and no seed is required; small numeric differences between CPU and GPU kernels can reorder near-tied candidates or perturb low-order embedding digits. The `score` field is a softmax over logits and is not calibrated — the upstream card's own example gives its top candidate 0.107 and the smoke run gave `paris` 0.417 — so a caller who needs probabilities must fit a calibration map on their own labelled cloze data. Embeddings carry no confidence at all; a caller who wants an uncertainty on a similarity must estimate it empirically (for example by bootstrap over a labelled pair set).

#### Ethical considerations and biases

###### Data

Upstream discloses pre-training on BookCorpus (11 038 unpublished books) and English Wikipedia (upstream README "Training data"); the disclosure stops there — no per-document licensing, author consent or demographic composition is given. BookCorpus consists of self-published fiction scraped from the web and Wikipedia contains biographies of living people, so the presence of personal data in the pre-training corpus is not ruled out, and the capacity of masked language models to reproduce memorised spans means it cannot be excluded from outputs either. This repository distributes code, tests and documentation; the 441 MB snapshot (`model.safetensors` and tokenizer files) is git-ignored and staged locally under `weights/bert-base-uncased/` with a manifest, and no sample data or datasets are shipped. The operator must audit the text they submit for personal, confidential or proprietary content and must not log or retain embeddings of such text without the same controls they would apply to the text itself; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life, and it has not been validated or certified for any of them by anyone. Its only validation is the offline unit suite (12 tests) and the CPU smoke run recorded in this repository. Where a sensitive use is foreseeable — for example ranking résumés by embedding similarity, or auto-completing clinical notes — it is admissible only with a human reviewer on every consequential outcome, an independent domain evaluation on representative data with a documented bias probe, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented and inspectable in `src/bert_masked_lm_pipeline/pipeline.py`: (1) supply chain — `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest naming another model or revision and fetches only manifest-listed files at the pinned revision, and `verify_snapshot` re-hashes every file in `weights/bert-base-uncased/dimer-base-manifest.json` and raises on the first size or SHA-256 mismatch before any weight is loaded; the Hub path is taken only with `allow_download=True` and always with `revision=MODEL_REVISION`; `trust_remote_code=False` on both tokenizer and model. (2) Input integrity — `_check_text` rejects non-`str`, empty and over-`MAX_TEXT_CHARS` inputs, `fill_mask` requires exactly one `[MASK]` and an integer `top_k` in 1–`MAX_TOP_K`, the runners reject texts over `MAX_TEXT_TOKENS` instead of truncating, and `embed` rejects batches outside 1–`MAX_BATCH` and unknown `pooling`. (3) Reproducibility — exact `==` dependency pins, `model.eval()`, float32, and `model_id`/`model_revision`/`device`/`source` in every result. (4) Refusals — no fine-tuning, no next-sentence prediction, no multi-mask filling and no raw-logit access are exposed; a missing snapshot with `allow_download=False` raises `FileNotFoundError`. No statistical mitigation (debiasing, re-weighting) is applied: the weights are redistributed unmodified.

###### Risks and harms

Bias reproduction: the fill-mask head ranks occupations, adjectives and roles by their co-occurrence with group terms in 2018-era books and Wikipedia, as the upstream card's man/woman example shows; data subjects and third parties bear the harm when such completions or embedding neighbourhoods feed a downstream system, and the likelihood under normal use is high whenever prompts mention people. Confident nonsense: any single-mask sentence yields a ranked list with a top score that can look decisive (0.42 for `paris` in the smoke run) even when the sentence is ungrammatical or off-domain; the operator bears the harm when that rank is acted on. Weak embeddings mistaken for semantic similarity: raw BERT vectors cluster by surface form and length as much as by meaning, so a retrieval or deduplication system built on them without evaluation silently returns wrong neighbours. Memorised-span leakage: masked prediction can surface names or phrases memorised from the pre-training corpus. Automation bias: a reviewer shown a ranked suggestion checks it less carefully. Magnitude ranges from a poor autocomplete to a discriminatory ranking of people.

###### Use cases

The pipeline must not be used for surveillance, biometric or demographic profiling, or social scoring — including inferring gender, ethnicity, religion, health status or political opinion from text via mask completions or embedding proximity. It must not support unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, for example by ranking applicants or claims through embedding similarity to a preferred profile. It must not power deceptive or manipulative applications such as generating plausible-looking fabricated statements attributed to real people. Any use that violates the Apache-2.0 terms of the upstream weights or the DIMER deployment terms is prohibited. The developers identify no further prohibited use beyond these because the model's outputs are single-token rankings and unlabelled vectors.

## Immutable provenance

- Model: `google-bert/bert-base-uncased`
- Revision: `86b5e0934494bd15c9632b12f734a8a67f723594`
- Snapshot manifest: `weights/bert-base-uncased/dimer-base-manifest.json`, `totalBytes` 441170446, 8 files
- `model.safetensors` SHA-256: `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3` (440449768 bytes)
- `config.json` SHA-256: `7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20` (570 bytes)
- `vocab.txt` SHA-256: `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` (231508 bytes)
- Weight format: SafeTensors; loader `BertForMaskedLM.from_pretrained(<snapshot dir>, local_files_only=True, trust_remote_code=False)` with `AutoTokenizer` from the same directory. The checkpoint's pooler and `cls.seq_relationship` (next-sentence) weights are not loaded by this architecture.

## Input/output contract

- `BERTMaskedLMPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`
- `fill_mask(text, top_k=5)` — `text`: `str`, 1–4000 chars, ≤ 512 tokens, exactly one `[MASK]`. Returns `{"candidates": [{"token", "token_id", "score", "sequence"}, ...], "top_k", "n_tokens", "device", "source", "model_id", "model_revision"}`; `score` is the softmax over 30 522 tokens at the masked position; `sequence` is the input with `[MASK]` replaced by the candidate.
- `embed(texts, pooling="cls")` — `texts`: list of 1–64 `str`; `pooling` in `("cls", "mean")`. Returns `{"embeddings": [[768 floats], ...], "dim": 768, "pooling", "normalized": True, "n_tokens", "device", "source", "model_id", "model_revision"}`; vectors are L2-normalised.
- `verify_snapshot(path=None)` — returns the manifest dict with `path`; raises `FileNotFoundError` / `ValueError`.
- `stage_missing_files(path=None, *, allow_download=False, downloader=None)` — returns the relative paths fetched (`[]` if none were missing).

## Runtime

- Pins: `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`; Python 3.12.
- Precision: float32 on both CPU and CUDA (`dtype=torch.float32` in the loader).
- Measured (Windows venv `dimer-next16`, CPU, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `device="cpu"`): source `local-snapshot`, load + verify 4.34 s, `fill_mask("The capital of France is [MASK].")` 0.13 s → `paris` 0.4168, `lille` 0.0714, `lyon` 0.0634, `marseille` 0.0444, `tours` 0.0303 (9 tokens); `embed` of two sentences with mean pooling 0.02 s → two unit-norm 768-d vectors, cosine 0.8719. The CUDA path was not run.
- Tests: `pytest -q -o addopts= tests` — 12 passed, offline, no weights required; `ruff check src tests` clean.

## References

- Devlin, Chang, Lee, Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019. https://arxiv.org/abs/1810.04805
- Google Research BERT repository. https://github.com/google-research/bert
- Zhu et al. Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books. ICCV 2015 (BookCorpus). https://arxiv.org/abs/1506.06724
- Upstream card: https://huggingface.co/google-bert/bert-base-uncased
