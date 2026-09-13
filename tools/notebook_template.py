"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "bert_masked_lm_pipeline",
    "repo_name": "bert-masked-lm-pipeline",
    "stem": "bert_masked_lm",
    "notebook_name": "bert_masked_lm_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "BERTMaskedLMPipeline",
    "weights_key": "bert-base-uncased",
    "runtime_imports": ["torch", "transformers"],
    "title": "BERT-Base uncased — DIMER fill-mask and sentence-embedding tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/bert-masked-lm-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/bert-masked-lm-pipeline/blob/main/tutorials/bert_masked_lm_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google--bert%2Fbert--base--uncased-ffcc4d?style=flat",
            "https://huggingface.co/google-bert/bert-base-uncased",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-google--research%2Fbert-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/google-research/bert",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-1810.04805-b31b1b.svg", "https://arxiv.org/abs/1810.04805"),
    ],
    "capability": "masked-language modelling (fill-mask: ranked candidates for one `[MASK]`) and sentence embeddings (768-d, CLS or mean pooled, L2-normalised) using the pinned BERT-Base uncased weights",
    "intro": (
        "At inference the WordPiece tokenizer lower-cases the text and one forward pass of the 12-layer bidirectional "
        "encoder runs; `fill_mask` reads the output-vocabulary logits at the single `[MASK]` position through the "
        "masked-LM head and ranks them by a softmax over the 30,522-token vocabulary, while `embed` takes the encoder's "
        "last hidden states and pools them to one 768-d vector per text (the `[CLS]` position, or the attention-masked "
        "mean) and L2-normalises it. **No adaptation occurs:** no training, fine-tuning, in-context conditioning, or "
        "preprocessing fitting — the pinned checkpoint is used as published, with its next-sentence-prediction head and "
        "pooler weights left unloaded. What the upstream checkpoint supplies is the encoder, the masked-LM head and the "
        "tokenizer; what the carried pipeline module adds is manifest verification, input validation and ceilings "
        "(over-long texts are rejected, not truncated), the two task methods, fixed output contracts and the "
        "`validate_inputs` and `evaluation_report` stage helpers. **The fill-mask `score` is not a calibrated "
        "probability** (it is a softmax ranking signal), and **embeddings are representations, not predictions**; the "
        "pipeline ships no threshold for either."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, author a synthetic cloze sentence "
        "and a few sentences to embed (or upload your own), stage and digest-verify the immutable upstream snapshot, "
        "surface the pipeline's ceilings and validate both capabilities' inputs into one input manifest, run fill-mask "
        "and read its ranked candidates correctly, run embedding and read the vector contract correctly (shape, pooling, "
        "unit), read from the machine-readable evaluation report why no metric is reported and what labelled data each "
        "capability needs, and export identifiers alongside vectors plus provenance."
    ),
    "exclusions": (
        "fine-tuning or classification heads, next-sentence prediction, multi-mask filling, raw-logit access, text "
        "generation (BERT is an encoder), cased or non-English text (the checkpoint is uncased English), or "
        "contrastively trained sentence similarity (the `qwen3-embedding-pipeline` sibling covers retrieval-grade "
        "embeddings). The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (also float32; the pipeline loads the checkpoint in float32 on both). The model card's CPU smoke loaded and verified the snapshot in 4.34 s, filled one mask in 0.13 s and embedded two sentences in 0.02 s, so the default runs in seconds on a hosted CPU runtime. The pinned `torch==2.14.0` install and the 440 MB `model.safetensors` are the largest downloads of the run.",
        "- **Knowledge:** basic Python; what a softmax over a vocabulary is and why it is not a calibrated probability; what cosine similarity between unit vectors means.",
        "- **Data:** the default sample is one synthetic cloze sentence and three synthetic sentences authored in code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 text file whose first non-empty line is a cloze sentence containing exactly one `[MASK]` and whose remaining non-empty lines are the sentences to embed. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Author the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**, written in this cell: one cloze sentence with a single `[MASK]` (the "
                "model card's smoke sentence) together with the token its author expects, and three sentences to embed — "
                "two about a pet on a floor covering and one unrelated — each given a stable identifier (`s1`, `s2`, `s3`) "
                "so every vector can be mapped back to its text. The expected token is the author's intent, not a "
                "labelled dataset: whether it lands in the top-`k` is a smoke/sanity check that the code path works, "
                "never an accuracy figure and never benchmark evidence; the three sentences carry **no similarity "
                "labels**, so the cosine table they produce is a qualitative check only. `TOP_K` and `POOLING` are Colab "
                "form parameters checked against the carried module in Section 5.\n\n"
                "BYOD is optional and disabled by default. Expected BYOD input: one UTF-8 text file whose first non-empty "
                "line is a cloze sentence containing exactly one `[MASK]` and whose remaining non-empty lines (at most "
                "`MAX_BATCH`) are the sentences to embed, each at most `MAX_TEXT_CHARS` characters and at most "
                "`MAX_TEXT_TOKENS` WordPiece tokens (longer texts are rejected by the pipeline, not truncated). Text is "
                "lower-cased by the tokenizer, so case carries no information. The upload stays inside this runtime. If "
                "you also hold gold tokens or similarity judgements, keep them outside the notebook — Section 8 explains "
                "what to compute with them."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "TOP_K = 5  # @param {{type:\"integer\"}}\n"
                "POOLING = 'mean'  # @param [\"cls\", \"mean\"]\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    lines = [line.strip() for line in io.StringIO(uploaded[sample_name].decode('utf-8')) if line.strip()]\n"
                "    if len(lines) < 2:\n"
                "        raise ValueError(f'{{sample_name}}: expected a [MASK] sentence line followed by at least one sentence line')\n"
                "    cloze, sentences = lines[0], lines[1:]\n"
                "    expected_token = None\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    cloze = 'The capital of France is [MASK].'\n"
                "    expected_token = 'paris'\n"
                "    sentences = [\n"
                "        'The cat sat on the mat.',\n"
                "        'A dog lay on the rug.',\n"
                "        'Interest rates were raised by a quarter of a point.',\n"
                "    ]\n"
                "    sample_name = 'synthetic_cloze_and_sentences'\n"
                "    sample_kind = 'synthetic (authored in this cell; the model card smoke cloze sentence)'\n"
                "sentence_ids = [f's{{index + 1}}' for index in range(len(sentences))]\n"
                "sample_sha256 = hashlib.sha256('\\n'.join([cloze, *sentences]).encode('utf-8')).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'cloze': cloze, 'expected_token': expected_token, 'sentences': len(sentences), 'top_k': TOP_K, 'pooling': POOLING, 'text_sha256': sample_sha256}})\n"
                "for sentence_id, sentence in zip(sentence_ids, sentences, strict=True):\n"
                "    print(f'{{sentence_id}}: {{sentence[:100]}}')"
            ),
        },
        {
            "md": (
                "## 5. Validate the inputs → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage and covers **both** capabilities: it takes "
                "the batch `embed` would take, and every entry carrying a `[MASK]` token is additionally checked against "
                "`fill_mask`'s contract and marked `fill_mask_input` in the manifest. The checks are the methods' own — "
                "`_check_batch`/`_check_text` for `embed`, `_check_mask_count`/`_check_top_k` for `fill_mask` — so a "
                "rejection here is a rejection there. `MAX_TEXT_CHARS` is the character guard applied before "
                "tokenisation; `MAX_TEXT_TOKENS` (512, the checkpoint's position limit) is applied after tokenisation "
                "and **rejects** longer texts rather than truncating them, so it is enforced inside the pipeline and "
                "cannot be observed at this stage; `MAX_BATCH` bounds one `embed` call; `MAX_TOP_K` bounds `top_k`; "
                "`MASK_TOKEN` must occur exactly once in a fill-mask text; `POOLINGS` names the two pooling policies; "
                "`VOCAB_SIZE` and `HIDDEN_SIZE` are the output-vocabulary and vector widths the contracts promise. The "
                "manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell "
                "also validates a two-mask text and records the pipeline's own error message as a finding. The notebook "
                "never trims or alters the texts."
            ),
            "code": (
                "import json\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "ceilings = {{'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MAX_BATCH': MAX_BATCH, 'MAX_TOP_K': MAX_TOP_K, 'VOCAB_SIZE': VOCAB_SIZE, 'HIDDEN_SIZE': HIDDEN_SIZE, 'MASK_TOKEN': MASK_TOKEN, 'POOLINGS': POOLINGS, 'DEFAULT_TOP_K': DEFAULT_TOP_K}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs([cloze, *sentences], top_k=TOP_K, pooling=POOLING, names=['cloze', *sentence_ids])\n"
                "# Demonstrate the exactly-one-mask rejection; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs([f'A {{MASK_TOKEN}} and another {{MASK_TOKEN}}.'], top_k=TOP_K, pooling=POOLING)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'two-mask-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))\n"
                "print({{'token_ceiling': f'MAX_TEXT_TOKENS={{MAX_TEXT_TOKENS}} is checked by the pipeline after tokenisation and rejects, never truncates'}})"
            ),
        },
        {
            "md": (
                "## 6. Fill the mask and read the candidates correctly\n\n"
                "**Input/output contract.** `fill_mask(text, top_k=...)` takes one string with exactly one `[MASK]` and "
                "returns `candidates` — a list of `top_k` entries ordered by descending `score`, each with the WordPiece "
                "`token`, its `token_id`, the `score`, and the `sequence` with the mask substituted — plus `top_k`, "
                "`n_tokens` (WordPiece count including `[CLS]`/`[SEP]`), the device and the model identity. **Score "
                "semantics:** `score` is the softmax over the 30,522-token vocabulary at the masked position — a ranking "
                "signal that sums to 1 over the whole vocabulary, not a calibrated probability that the candidate is "
                "correct; the **default decision rule is `argmax`** (the first candidate) and the pipeline applies no "
                "minimum score, so a nonsensical sentence still yields a ranked list. Any acceptance threshold is owned "
                "by the caller and must be set on their own labelled cloze data. The model card's CPU smoke on this "
                "sentence ranked `paris` first with score 0.4168 (9 tokens); that is one observation, not an expected "
                "value — near-tied candidates can reorder between CPU and CUDA kernels. Inference is deterministic on a "
                "fixed device and dtype (`model.eval()`, no sampling, no seed needed)."
            ),
            "code": (
                "import time\n\n"
                "started = time.perf_counter()\n"
                "filled = pipe.fill_mask(cloze, top_k=TOP_K)\n"
                "fill_elapsed = time.perf_counter() - started\n"
                "scores = [candidate['score'] for candidate in filled['candidates']]\n"
                "fill_checks = {{\n"
                "    'top_k_candidates_returned': len(filled['candidates']) == TOP_K,\n"
                "    'scores_descending': all(a >= b for a, b in zip(scores, scores[1:], strict=False)),\n"
                "    'scores_in_unit_interval': all(0.0 < s <= 1.0 for s in scores),\n"
                "    'tokens_within_vocab': all(0 <= candidate['token_id'] < VOCAB_SIZE for candidate in filled['candidates']),\n"
                "    'n_tokens_within_ceiling': 1 <= filled['n_tokens'] <= MAX_TEXT_TOKENS,\n"
                "}}\n"
                "if not all(fill_checks.values()):\n"
                "    raise RuntimeError(f'fill_mask output failed a sanity check: {{fill_checks}}')\n"
                "print({{key: value for key, value in filled.items() if key != 'candidates'}})\n"
                "print({{'seconds': round(fill_elapsed, 3), 'checks': fill_checks, 'decision_rule': 'argmax = first candidate; no threshold shipped'}})\n"
                "print(f'cloze: {{cloze}}')\n"
                "for rank, candidate in enumerate(filled['candidates'], start=1):\n"
                "    print(f\"{{rank:>2}}. {{candidate['token']:<12}} id {{candidate['token_id']:>6}}  score {{candidate['score']:.4f}}  {{candidate['sequence']}}\")\n"
                "fill_sanity = {{}}\n"
                "if expected_token is not None:\n"
                "    fill_sanity = {{'expected_token_in_top_k': expected_token in [candidate['token'] for candidate in filled['candidates']]}}\n"
                "    print({{'sanity_check': fill_sanity, 'note': 'falsifiable plumbing check on one synthetic sentence; not a metric'}})"
            ),
        },
        {
            "md": (
                "## 7. Embed the sentences and read the vectors correctly\n\n"
                "**Input/output contract.** `embed(texts, pooling=...)` takes a list of 1..`MAX_BATCH` strings and "
                "returns `embeddings` — one list per input text, **in input order**, each of length `dim` (768) — plus "
                "`pooling` (`cls`: the `[CLS]` position of the last layer; `mean`: the attention-masked mean of the last "
                "layer, so padding never enters the average), `normalized` (`True`: every vector has unit L2 norm), "
                "`n_tokens` per text, the device and the model identity. The unit of embedding is **one vector per "
                "text**; there is no per-token or per-chunk output, and a text over `MAX_TEXT_TOKENS` is rejected rather "
                "than chunked or truncated. Missing data has no meaning here: empty strings are rejected, not embedded. "
                "**Embeddings are representations, not predictions:** they carry no label and no confidence, and their "
                "only meaning is relative — cosine between two vectors from the same model and the same pooling "
                "policy.\n\n"
                "**No intrinsic metric exists** for an embedding: the repository ships no metric helper and no labelled "
                "data, and quality can only be judged through a downstream task with labels — a similarity benchmark "
                "with human judgements (Spearman correlation), or a retrieval or clustering set with relevance labels "
                "(recall@k). Below, the pairwise cosine matrix is computed from the returned vectors as a **qualitative "
                "check** that the contract works: on the default sample the two pet sentences are expected to score "
                "higher with each other than with the unrelated one. Cosine values are similarities in `[-1, 1]` on this "
                "model's geometry; raw BERT was not trained with a sentence-similarity objective, so its cosine scale is "
                "compressed and uncalibrated (the model card's smoke gave 0.8719 for the two pet sentences with mean "
                "pooling — one observation, not an expected value), absolute values are not comparable across models or "
                "pooling policies, and any \"same meaning\" threshold is the caller's to set on labelled pairs. Look for "
                "`(N, 768)` unit-norm vectors, token counts within the ceiling, and no truncation (the pipeline cannot "
                "truncate)."
            ),
            "code": (
                "started = time.perf_counter()\n"
                "embedding_result = pipe.embed(sentences, pooling=POOLING)\n"
                "embed_elapsed = time.perf_counter() - started\n"
                "vectors = np.asarray(embedding_result['embeddings'], dtype=np.float32)\n"
                "norms = np.linalg.norm(vectors, axis=1)\n"
                "embed_checks = {{\n"
                "    'one_vector_per_text': vectors.shape == (len(sentences), HIDDEN_SIZE),\n"
                "    'dim_matches_contract': embedding_result['dim'] == HIDDEN_SIZE,\n"
                "    'pooling_as_requested': embedding_result['pooling'] == POOLING,\n"
                "    'unit_norm': bool(embedding_result['normalized']) and bool(np.allclose(norms, 1.0, atol=1e-4)),\n"
                "    'all_values_finite': bool(np.isfinite(vectors).all()),\n"
                "    'n_tokens_within_ceiling': all(1 <= n <= MAX_TEXT_TOKENS for n in embedding_result['n_tokens']),\n"
                "}}\n"
                "if not all(embed_checks.values()):\n"
                "    raise RuntimeError(f'embed output failed a sanity check: {{embed_checks}}')\n"
                "print({{key: value for key, value in embedding_result.items() if key != 'embeddings'}})\n"
                "print({{'seconds': round(embed_elapsed, 3), 'shape': vectors.shape, 'norms': [round(float(n), 4) for n in norms], 'unit': 'one vector per text', 'checks': embed_checks}})\n"
                "cosine = vectors @ vectors.T\n"
                "for sentence_id, row in zip(sentence_ids, cosine, strict=True):\n"
                "    print(sentence_id, {{other_id: round(float(value), 4) for other_id, value in zip(sentence_ids, row, strict=True)}})\n"
                "embed_sanity = {{}}\n"
                "if not USE_BYOD:\n"
                "    embed_sanity = {{'related_pair_outscores_unrelated': bool(cosine[0, 1] > max(cosine[0, 2], cosine[1, 2]))}}\n"
                "    print({{'sanity_check': embed_sanity, 'note': 'qualitative check on three synthetic sentences; not a metric'}})"
            ),
        },
        {
            "md": (
                "## 8. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report — even, as "
                "here, when nothing is measurable. The repository ships **no metric helper and reports no performance "
                "measure** for either capability, so the verdict is always `not-measurable` and the report states what "
                "would make each task measurable: for fill-mask, a labelled cloze set (sentence, mask position, gold "
                "token) over enough sentences to state a dispersion, scored with the caller's own top-1/top-k hit-rate "
                "code; for the embeddings, a judged similarity set (Spearman correlation) or a retrieval or clustering "
                "set with relevance labels (recall@k). Supplying the author's expected token does **not** change the "
                "verdict — one expected token is an intent, not a labelled cloze set, and scoring it would present a "
                "single observation as an accuracy — so the helper records that in `reason` instead. The sanity checks "
                "printed in Sections 6 and 7 remain falsifiable plumbing checks, not results. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "expected_tokens = None if expected_token is None else [expected_token]\n"
                "report = evaluation_report(filled, expected_tokens, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is reported: neither capability has a metric helper here; compute top-k hit rates on your own labelled cloze set and Spearman or recall@k on your own judged pairs.')"
            ),
        },
        {
            "md": (
                "## 9. Export identifiers alongside vectors, and provenance\n\n"
                "Two further files are written under `outputs/` beside the input manifest and the evaluation report. The "
                "vectors go to CSV (`outputs/{stem}_embeddings.csv`) with one row per sentence — `id`, `n_tokens`, "
                "`pooling`, then `e0000…e0767` — so every vector stays attached to its identifier for downstream use. One "
                "JSON record (`outputs/{stem}_result.json`) preserves the fill-mask result (the cloze, every candidate "
                "with token, id, score, rank and sequence, the decision rule, `n_tokens`), the embedding contract (`dim`, "
                "`pooling`, `normalized`, unit), the identified sentences with their token counts, the cosine table keyed "
                "by identifier, the sanity checks, the ceilings in force, the input manifest, the evaluation report, the "
                "sample identity and digest, the notebook's source (repository, revision, embedded module digest, "
                "generator), the model identifier, the immutable model revision, the model licence, the verified snapshot "
                "summary, and the runtime identity (Python, `torch`, `transformers`, device, dtype). No credentials are "
                "involved in any step, so none can reach the export."
            ),
            "code": (
                "import csv\n\n"
                "with open('outputs/{stem}_embeddings.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['id', 'n_tokens', 'pooling'] + [f'e{{i:04d}}' for i in range(HIDDEN_SIZE)])\n"
                "    for sentence_id, vector, n_tokens in zip(sentence_ids, embedding_result['embeddings'], embedding_result['n_tokens'], strict=True):\n"
                "        writer.writerow([sentence_id, n_tokens, embedding_result['pooling']] + [f'{{value:.7f}}' for value in vector])\n"
                "payload = {{\n"
                "    'fill_mask': {{\n"
                "        'cloze': cloze,\n"
                "        'expected_token': expected_token,\n"
                "        'candidates': [{{'rank': rank, **candidate}} for rank, candidate in enumerate(filled['candidates'], start=1)],\n"
                "        'decision_rule': 'argmax (first candidate); score is a vocabulary softmax, not a calibrated probability; no threshold shipped',\n"
                "        'top_k': filled['top_k'],\n"
                "        'n_tokens': filled['n_tokens'],\n"
                "        'seconds': round(fill_elapsed, 3),\n"
                "        'sanity_checks': fill_checks,\n"
                "        'plumbing_check': fill_sanity,\n"
                "    }},\n"
                "    'embed': {{\n"
                "        'contract': {{'dim': embedding_result['dim'], 'pooling': embedding_result['pooling'], 'normalized': embedding_result['normalized'], 'unit': 'one vector per text'}},\n"
                "        'texts': dict(zip(sentence_ids, sentences, strict=True)),\n"
                "        'n_tokens': dict(zip(sentence_ids, embedding_result['n_tokens'], strict=True)),\n"
                "        'cosine_similarity': {{sentence_id: {{other_id: float(value) for other_id, value in zip(sentence_ids, row, strict=True)}} for sentence_id, row in zip(sentence_ids, cosine, strict=True)}},\n"
                "        'vectors_file': 'outputs/{stem}_embeddings.csv',\n"
                "        'seconds': round(embed_elapsed, 3),\n"
                "        'sanity_checks': embed_checks,\n"
                "        'plumbing_check': embed_sanity,\n"
                "    }},\n"
                "    'ceilings': ceilings,\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'text_sha256': sample_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': snapshot['path'], 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes')}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The fill-mask candidates are a softmax ranking over the vocabulary at one masked position: the first entry is "
        "the argmax, the `score` is not a calibrated probability, and the pipeline applies no threshold — the caller owns "
        "any cut-off and must set it on labelled cloze data. The embeddings are unit vectors in this model's "
        "768-dimensional space: representations that predict nothing, whose only meaning is cosine between vectors from "
        "the same model and pooling policy; raw BERT's cosine scale is compressed and uncalibrated, and a "
        "retrieval-grade sentence encoder is a different model. On the synthetic sample both outputs are plumbing "
        "evidence only; the evaluation report is `not-measurable` because no metric can be computed without labelled "
        "data, and a real evaluation needs a labelled cloze set (top-k hit rates) and a judged similarity or retrieval "
        "set (Spearman, recall@k) with the caller's own code over enough items to state a dispersion. Text is "
        "lower-cased and accent-stripped by the tokenizer; texts over 512 WordPiece tokens are rejected, not truncated; "
        "the checkpoint is uncased English only and carries the gender and occupation associations the upstream card "
        "documents; the pipeline exposes no fine-tuning, generation, multi-mask filling or next-sentence prediction. "
        "Inference is deterministic on a fixed device and dtype, but CPU and CUDA kernels can reorder near-tied "
        "candidates and perturb low-order embedding digits.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model snapshot, validate the demonstrated inputs against the enforced "
        "ceilings, execute both public pipeline paths, and emit the shown machine-readable outputs in the tested runtime "
        "— without the repository being reachable. It does **not** establish benchmark superiority, cloze accuracy or "
        "similarity quality on any domain, a usable threshold, safety for high-consequence decisions, or production "
        "fitness on an unseen domain.\n\n"
        "**Next experiments.** Switch `POOLING` between `cls` and `mean` and watch the cosine table move — the same "
        "sentences, a different representation; write a cloze sentence whose answer is ambiguous and read how the "
        "softmax mass spreads across candidates; assemble a dozen labelled cloze sentences of your own and compute the "
        "top-1 and top-5 hit rates the evaluation report asks for; judge a handful of sentence pairs yourself and "
        "compute Spearman correlation against the cosine values. None of these turns the sample result into evidence of "
        "production fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/bert-masked-lm-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/bert-masked-lm-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/bert-masked-lm-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/bert\n"
        "- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding: https://arxiv.org/abs/1810.04805"
    ),
}
