"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E continued-pre-training workflow: the pinned BERT-Base uncased snapshot is
digest-verified and loaded, a digest-pinned real corpus (SciTLDR paper abstracts) is fetched, validated and
split, both inference contracts are exercised, the frozen model's masked-token metrics on held-out abstracts
are read beside a unigram floor, a bounded masked-language-model fine-tuning of the last encoder layers adapts
the model to the domain in the kernel, the held-out split is scored again, and the adapter is exported and
reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "bert_masked_lm_pipeline",
    "repo_name": "bert-masked-lm-pipeline",
    "stem": "bert_masked_lm",
    "notebook_name": "bert_masked_lm_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned BERT-Base uncased snapshot (safetensors, 440 MB), fetches the three digest-pinned SciTLDR-A files from the "
        "project repository (5.5 MB, no credential), reads the paper abstracts and draws 300 / 50 / 100 training, validation "
        "and test documents from the release's own paper-disjoint members, fills one mask in an unseen abstract's opening "
        "sentence and embeds two test abstracts through the inference contracts with an input manifest and a rejection "
        "probe, scores the frozen model on the test abstracts by masked-token perplexity and top-1 / top-5 accuracy at "
        "seeded masks beside the add-one unigram floor, runs a bounded continued masked-language-model training of the "
        "last four encoder layers with validation-perplexity epoch selection, scores the held-out split again, fills "
        "the same masks and re-embeds the same abstracts with the adapted model, exports the adapter as safetensors with a "
        "manifest, and reloads that artifact into a fresh pipeline to verify parity. The default path needs no repository "
        "clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 "
        "§5). On CPU the whole path takes about four minutes of model time after the downloads; a CUDA runtime is used "
        "automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "text corpus as a CSV (columns `id`, `text`), a JSON array or JSONL file of `{{id, text}}` records, or a plain "
        "`.txt` file in which every blank-line-separated paragraph is one document. It passes through the same validation, "
        "seeded text-disjoint split, unigram floor, frozen scoring, fine-tuning, held-out evaluation, cloze and embedding "
        "checks, artifact export and reload-parity cells as the SciTLDR sample. The expected schema and the ceilings are "
        "stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
    ),
    "pipeline_class": "BERTMaskedLMPipeline",
    "weights_key": "bert-base-uncased",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "transformers"],
    "title": "BERT-Base uncased — DIMER E2E continued-pre-training tutorial: masked-token prediction on paper abstracts (standalone)",
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
    "capability": "masked-language modelling (fill-mask: ranked candidates for one `[MASK]`), sentence embeddings (768-d, CLS or mean pooled, L2-normalised) and bounded continued masked-language-model training of the last encoder layers on a text corpus, measured by held-out masked-token perplexity and top-k accuracy, using the pinned BERT-Base uncased weights",
    "intro": (
        "At inference the WordPiece tokenizer lower-cases the text and one forward pass of the 12-layer bidirectional "
        "encoder runs; `fill_mask` reads the output-vocabulary logits at the single `[MASK]` position through the "
        "masked-LM head and ranks them by a softmax over the 30,522-token vocabulary, while `embed` takes the encoder's "
        "last hidden states and pools them to one 768-d vector per text (the `[CLS]` position, or the attention-masked "
        "mean) and L2-normalises it. The carried pipeline module adds manifest verification, input validation and "
        "ceilings (over-long texts are rejected, not truncated), the two task methods and fixed output contracts. **The "
        "fill-mask `score` is not a calibrated probability** (it is a softmax ranking signal), and **embeddings are "
        "representations, not predictions**; the pipeline ships no threshold for either.\n\n"
        "What this notebook adds to inference is **continued pre-training measured by the model's own objective**. The "
        "dataset is real: SciTLDR-A (Cachola et al., 2020; Apache-2.0) ships the abstracts of 3,229 computer-science "
        "papers as three digest-pinned JSON-Lines files fetched from the project repository at a pinned commit; only the "
        "abstract text is used, so every record is one document and no label is needed. The carried `metrics.py` masks a "
        "**seeded 15 %** of each held-out abstract's WordPiece tokens — the same positions for every model scored under "
        "the same seed — and reads the negative log-likelihood and the rank of the original token at every masked "
        "position, aggregated into **masked perplexity**, **bits per masked token** and **top-1 / top-5 accuracy**; an "
        "**add-one unigram model** fitted on the training tokens is the floor a model that ignores context reaches. The "
        "fine-tuning question is whether a bounded masked-language-model adaptation of the last encoder layers on 300 "
        "abstracts lowers that perplexity on abstracts the model has not seen. Nothing here is a quality claim: a lower "
        "masked perplexity says the adapted encoder predicts missing words of paper abstracts better, not that its "
        "embeddings are better for your task."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, dataset and metrics modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned real corpus and validate and split it "
        "without leakage; run fill-mask and embedding through the public API and read the candidate and vector contracts "
        "correctly; read masked perplexity and top-k accuracy beside a unigram floor and understand what they do and do "
        "not measure; run a bounded continued masked-language-model training with explicit hyperparameters and "
        "validation-based epoch selection; evaluate on an independent test split; compare cloze candidates and embedding "
        "similarities before and after; and export a safetensors adapter that reloads against the pinned base with "
        "verified parity."
    ),
    "exclusions": (
        "classification or other supervised heads, next-sentence prediction, multi-mask filling, raw-logit access, text "
        "generation (BERT is an encoder), cased or non-English text (the checkpoint is uncased English), contrastively "
        "trained sentence similarity (the `qwen3-embedding-pipeline` sibling covers retrieval-grade embeddings), "
        "full-model or embedding-table training, any similarity or downstream-task score, and any claim that a SciTLDR "
        "abstract split stands in for your corpus. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is adequate: the build record measured about 4 s to load and digest-verify the 440 MB snapshot, about 7 s to score the 100-abstract test split (2,983 masked positions) and about 102 s per training epoch over 300 abstracts plus a 50-abstract validation pass per epoch. The pinned `torch==2.14.0` install and the 440 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what masked-language modelling is; what a softmax over a vocabulary is and why it is not a calibrated probability; what perplexity is (the exponential of the mean negative log-likelihood) and why a masked perplexity is an intrinsic number, not a judgement of embedding quality; what cosine similarity between unit vectors means.",
        "- **Data contract:** records are `{{id, text}}` — one document of the domain, 1..4,000 characters and at most 512 WordPiece tokens including `[CLS]`/`[SEP]` (a longer record is refused, not truncated, everywhere), ids matching `[A-Za-z0-9_.:-]{{1,64}}` and unique; a dataset needs 8..20,000 records; texts are de-duplicated case-insensitively before splitting so the same document never sits in two splits; text is lower-cased by the tokenizer, so case carries no information. BYOD accepts CSV, JSON, JSONL or TXT in that shape.",
        "- **Validation is structural, not semantic:** nothing checks that a record belongs to the domain you mean — an off-topic corpus is trained on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — an internal document collection is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches three pinned objects (`train.jsonl` 3,155,015 bytes, `dev.jsonl` 1,124,865 bytes, `test.jsonl` 1,204,107 bytes; SHA-256 `b222771d…` / `3191fa98…` / `fb42dd6c…`) from `raw.githubusercontent.com` at the pinned `allenai/scitldr` commit over HTTPS, each refused on any mismatch before it is read; SciTLDR is Apache-2.0 (Cachola et al., 2020).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and split\n\n"
                "`fetch_corpus` downloads the three pinned SciTLDR-A files (or reads them from the cache), refuses a "
                "byte-size or SHA-256 mismatch per file before it is parsed, and `read_corpus` flattens each JSON-Lines "
                "member into records whose `text` is the abstract's sentences joined by a space — titles and TLDRs are "
                "left unread. `build_sample_dataset` keeps abstracts of 200..2,000 characters, drops repeated texts, and "
                "draws 300 training documents from the `train` member, 50 validation documents from `dev` and 100 test "
                "documents from `test` by a seeded shuffle — the release's own paper-disjoint partition. `validate_dataset` "
                "then checks every record against the contract, `check_split_disjoint` asserts no text appears in two "
                "splits, and the training split is written to `outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 1,992 + 619 + 618 raw papers, three digests, splits 300 / 50 / 100, and four refusal probes — "
                "a duplicate id, an empty text, a missing field and a dataset too small to split — each rejected before "
                "`torch` does anything."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_papers = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/scitldr'))\n"
                "    raw_papers = {{name: len(part) for name, part in corpus.items()}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_papers': raw_papers, 'splits': disjoint, 'file_sha256': {{k: v[2][:12] + '...' for k, v in CORPUS_FILES.items()}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_texts': manifest['unique_texts'], 'text_chars': manifest['text_chars'], 'total_chars': manifest['total_chars'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "print({{'example': {{'id': train_records[0]['id'], 'text': train_records[0]['text'][:200] + '...'}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'empty text': [{{**train_records[0], 'text': '   '}}, *train_records[1:8]],\n"
                "    'missing field': [{{'id': r['id']}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Fill a mask and embed through the inference contracts\n\n"
                "Before any adaptation, both inference contracts are exercised as they always were. A **cloze** is made "
                "from an unseen abstract's opening sentence by replacing its longest alphabetic word with `[MASK]` — the "
                "removed word is the *gold* word, which may be several WordPiece tokens, in which case no single "
                "candidate can match it (reported, never asserted). `validate_inputs` applies exactly the checks "
                "`fill_mask` and `embed` apply (text type and character ceiling, exactly one `[MASK]` for the cloze, "
                "`top_k` and `pooling` within their contracts) and returns an input manifest; the WordPiece ceiling "
                "`MAX_TEXT_TOKENS` (512) needs the real tokenizer and is enforced inside the pipeline, which **rejects, "
                "never truncates**. A two-mask input is validated too and its rejection recorded as a finding. `fill_mask` "
                "returns `top_k` candidates with a softmax `score` (a ranking signal, not a calibrated probability); "
                "`embed` returns one unit-norm 768-d vector per text under the requested pooling. Three unseen clozes and "
                "the embeddings of two test abstracts are kept as the *before* column for Section 9."
            ),
            "code": (
                "import re\n"
                "import time\n\n"
                "TOP_K = 5  # @param {{type:\"integer\"}}\n"
                "POOLING = 'mean'  # @param [\"cls\", \"mean\"]\n\n"
                "def cloze_of(record):\n"
                "    \"\"\"The opening sentence with its longest alphabetic word replaced by [MASK]; returns (cloze, gold word).\"\"\"\n"
                "    sentence = record['text'].split('. ')[0]\n"
                "    words = [w for w in re.findall(r'[A-Za-z]+', sentence) if len(w) >= 4]\n"
                "    gold = max(words, key=len)\n"
                "    return re.sub(r'\\b' + gold + r'\\b', MASK_TOKEN, sentence, count=1), gold.lower()\n\n"
                "if USE_BYOD:\n"
                "    unseen_records = [{{**r, 'id': f'unseen-{{i:02d}}'}} for i, r in enumerate(test_records[2:5])]\n"
                "else:\n"
                "    used = {{r['text'].lower() for part in splits.values() for r in part}}\n"
                "    unseen_records = [{{**r, 'id': f'unseen-{{i:02d}}'}} for i, r in enumerate([r for r in filter_records(corpus['dev']) if r['text'].lower() not in used][:3])]\n"
                "clozes = {{r['id']: cloze_of(r) for r in unseen_records}}\n"
                "cloze, gold = clozes[unseen_records[0]['id']]\n"
                "pair_texts = [test_records[0]['text'], test_records[1]['text']]\n"
                "ceilings = {{'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MAX_BATCH': MAX_BATCH, 'MAX_TOP_K': MAX_TOP_K, 'VOCAB_SIZE': VOCAB_SIZE, 'HIDDEN_SIZE': HIDDEN_SIZE, 'MASK_TOKEN': MASK_TOKEN, 'MASK_TOKEN_ID': MASK_TOKEN_ID, 'POOLINGS': POOLINGS, 'DEFAULT_TOP_K': DEFAULT_TOP_K}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs([cloze, *pair_texts], top_k=TOP_K, pooling=POOLING, names=['cloze', 'pair-a', 'pair-b'])\n"
                "try:\n"
                "    validate_inputs([f'A {{MASK_TOKEN}} and another {{MASK_TOKEN}}.'], top_k=TOP_K, pooling=POOLING)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'two-mask-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "started = time.perf_counter()\n"
                "filled = pipe.fill_mask(cloze, top_k=TOP_K)\n"
                "fill_seconds = round(time.perf_counter() - started, 3)\n"
                "started = time.perf_counter()\n"
                "embedding_result = pipe.embed(pair_texts, pooling=POOLING)\n"
                "embed_seconds = round(time.perf_counter() - started, 3)\n"
                "vectors = np.asarray(embedding_result['embeddings'], dtype=np.float32)\n"
                "scores = [c['score'] for c in filled['candidates']]\n"
                "checks = {{\n"
                "    'top_k_candidates_returned': len(filled['candidates']) == TOP_K,\n"
                "    'scores_descending_in_unit_interval': all(a >= b for a, b in zip(scores, scores[1:], strict=False)) and all(0.0 < s <= 1.0 for s in scores),\n"
                "    'tokens_within_vocab': all(0 <= c['token_id'] < VOCAB_SIZE for c in filled['candidates']),\n"
                "    'n_tokens_within_ceiling': 1 <= filled['n_tokens'] <= MAX_TEXT_TOKENS and all(1 <= n <= MAX_TEXT_TOKENS for n in embedding_result['n_tokens']),\n"
                "    'one_unit_vector_per_text': vectors.shape == (2, HIDDEN_SIZE) and bool(np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)),\n"
                "    'pooling_as_requested': embedding_result['pooling'] == POOLING and embedding_result['dim'] == HIDDEN_SIZE,\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'inference output failed a sanity check: {{checks}}')\n"
                "print({{'cloze': cloze, 'gold_word': gold, 'candidates': [(c['token'], round(c['score'], 4)) for c in filled['candidates']], 'gold_in_top_k': gold in [c['token'] for c in filled['candidates']], 'seconds': fill_seconds}})\n"
                "print({{'pair_cosine': round(float(vectors[0] @ vectors[1]), 4), 'pooling': POOLING, 'n_tokens': embedding_result['n_tokens'], 'seconds': embed_seconds, 'checks': checks, 'findings': len(input_manifest['findings']), 'score_semantics': 'softmax ranking signal, not a calibrated probability; no threshold shipped'}})\n"
                "before = {{rid: [c['token'] for c in pipe.fill_mask(text, top_k=TOP_K)['candidates']] for rid, (text, _gold) in clozes.items()}}\n"
                "vectors_before = vectors\n"
                "print({{'unseen_clozes_filled_by_the_frozen_model': len(before)}})"
            ),
        },
        {
            "md": (
                "## 6. The unigram floor and the frozen model's masked-token metrics on the test split\n\n"
                "Two numbers frame the adaptation. `pipe.unigram_baseline` fits an add-one-smoothed unigram model over the "
                "30,522-token vocabulary on the training tokens and predicts every masked position of the test abstracts "
                "with it: the perplexity and top-k accuracy a model that knows the domain's word frequencies but ignores "
                "every context reaches — expect a perplexity in the thousands and a top-1 accuracy of a few percent (the "
                "commonest tokens). `pipe.evaluate` masks a seeded 15 % of each test abstract's interior WordPiece tokens "
                "(`record_seed` makes the positions a function of the record id and `SEED`, so every model scored here "
                "sees the same masks), runs the frozen encoder and masked-LM head once per record, and reads the "
                "negative log-likelihood and the rank of the original token at every masked position: **masked "
                "perplexity** (`exp` of the mean NLL), **bits per masked token** and **top-1 / top-5 accuracy**. Records "
                "over `MAX_TEXT_TOKENS` are refused, never truncated. The build record saw the frozen model near 15 with "
                "top-1 accuracy just above 50 % on these abstracts; about seven seconds on CPU."
            ),
            "code": (
                "SEED = 0  # @param {{type:\"integer\"}}\n"
                "MASK_RATE = 0.15  # @param {{type:\"number\"}}\n\n"
                "def brief(m):\n"
                "    return {{'perplexity': round(m['perplexity'], 2), 'bits_per_token': round(m['bits_per_token'], 3), 'top1_accuracy': round(m['top1_accuracy'], 4), 'top5_accuracy': round(m['top5_accuracy'], 4), 'n_masked': m['n_masked']}}\n\n"
                "t0 = time.perf_counter()\n"
                "unigram = pipe.unigram_baseline(train_records, test_records, mask_rate=MASK_RATE, seed=SEED)\n"
                "print({{'unigram_floor': brief(unigram), 'baseline': unigram['baseline'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, mask_rate=MASK_RATE, seed=SEED)\n"
                "print({{'frozen_model_test': brief(frozen_test), 'n_records': frozen_test['n_records'], 'verdict': frozen_test['verdict'], 'adapted': frozen_test['adapted'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "assert frozen_test['n_masked'] == unigram['n_masked'] and frozen_test['perplexity'] < unigram['perplexity']"
            ),
        },
        {
            "md": (
                "## 7. Bounded continued masked-language-model training\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_LAYERS` encoder layers — four by default, "
                "28,351,488 of 109,514,298 parameters; the word, position and token-type embeddings, the earlier layers "
                "and the masked-LM head with its decoder tied to the word embeddings stay frozen — with the model's own "
                "pre-training objective: every epoch re-draws a seeded `MASK_RATE` of each training abstract's interior "
                "tokens, replaces them with `[MASK]` and applies cross-entropy at those positions only (100 % `[MASK]`, "
                "not the upstream 80/10/10 replacement mix), AdamW at a fixed learning rate, gradient clipping at 1.0, "
                "seeded shuffling and no scheduler. Epoch 0 records the frozen model's validation metrics under the "
                "evaluation masking; every epoch is scored the same way, and the epoch with the lowest validation masked "
                "perplexity is kept.\n\n"
                "Watch validation perplexity fall from about 13 by a point or two over two epochs while top-1 accuracy "
                "creeps up (about 102 s of training plus a validation pass per epoch on CPU). The build record's sweep on this sample: two layers at 1e-4 reached 12.96, four layers at 1e-4 reached 12.51 and four layers at 5e-5 reached 12.59 in about the same time — the default keeps the standard 5e-5."
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_LAYERS = 4  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_perplexity'] = round(entry['val']['perplexity'], 2)\n"
                "        row['val_top1'] = round(entry['val']['top1_accuracy'], 4)\n"
                "        row['val_top5'] = round(entry['val']['top5_accuracy'], 4)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_layers=TRAINABLE_LAYERS, mask_rate=MASK_RATE, seed=SEED, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'objective': adapt_result['objective'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'train_tokens': adapt_result['n_train_tokens'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or epoch selection, and no abstract in it appears in the "
                "training or validation splits. The adapted model is scored exactly as the frozen model was in Section 6 "
                "— same masks, same seed — and the three rows are put side by side. Look for a masked perplexity a point "
                "or two below the frozen one with top-1 and top-5 accuracy up by about a point, all far from the unigram "
                "floor; the cell asserts the adapted perplexity is lower than the frozen. One hundred abstracts from one "
                "seeded split of one corpus give no dispersion estimate; the delta is sample-sanity evidence that the "
                "adaptation contract works, not a benchmark, and a lower masked perplexity on paper abstracts says nothing "
                "about your corpus until you measure it there."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, mask_rate=MASK_RATE, seed=SEED)\n"
                "adapted_val = pipe.evaluate(val_records, mask_rate=MASK_RATE, seed=SEED)\n"
                "comparison = {{\n"
                "    metric: {{'unigram_floor': round(unigram[metric], 4), 'frozen': round(frozen_test[metric], 4), 'adapted': round(adapted_test[metric], 4)}}\n"
                "    for metric in ('perplexity', 'bits_per_token', 'top1_accuracy', 'top5_accuracy')\n"
                "}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 4) for metric in ('perplexity', 'bits_per_token', 'top1_accuracy', 'top5_accuracy')}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'masking': {{'mask_rate': MASK_RATE, 'seed': SEED, 'n_masked_test': frozen_test['n_masked']}},\n"
                "    'baselines': {{'unigram_floor': unigram}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['perplexity'] < frozen_test['perplexity']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Clozes and embeddings before and after, export the adapter and reload it\n\n"
                "The three unseen clozes filled by the frozen model in Section 5 are filled again by the adapted model "
                "through the same `fill_mask` contract, printed side by side with the gold word and whether it appears in "
                "the top-`k` before and after; the two test abstracts are re-embedded and their cosine similarity and "
                "each vector's cosine to its frozen self are printed. Read these as observations: the adapter moves the "
                "last encoder layers, so **every embedding changes** — how much, and whether for the better on your "
                "similarity task, nothing here measures. The single-input `evaluation_report` helper — the inference-stage "
                "helper — is written for the first cloze and stays `not-measurable`, because one cloze has no metric.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last four encoder layers, about 113 MB — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the "
                "training configuration and the epoch history (OUT8). `BERTMaskedLMPipeline.from_artifact` re-verifies "
                "the base snapshot, checks the artifact manifest and digest **before** deserialising, refuses any tensor "
                "that is not an encoder-layer tensor of the base, and overlays the tensors onto a freshly loaded base — a "
                "new object from files, not the in-memory model (VER2). The cell asserts an identical test perplexity on "
                "ten records, identical candidates and identical embeddings (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "rows = []\n"
                "for record in unseen_records:\n"
                "    text, gold_word = clozes[record['id']]\n"
                "    after = [c['token'] for c in pipe.fill_mask(text, top_k=TOP_K)['candidates']]\n"
                "    rows.append({{'id': record['id'], 'cloze': text, 'gold_word': gold_word, 'frozen_candidates': ' '.join(before[record['id']]), 'adapted_candidates': ' '.join(after), 'gold_in_top_k_frozen': gold_word in before[record['id']], 'gold_in_top_k_adapted': gold_word in after}})\n"
                "    print({{k: rows[-1][k] for k in ('id', 'gold_word', 'frozen_candidates', 'adapted_candidates', 'gold_in_top_k_frozen', 'gold_in_top_k_adapted')}})\n"
                "vectors_after = np.asarray(pipe.embed(pair_texts, pooling=POOLING)['embeddings'], dtype=np.float32)\n"
                "embedding_shift = {{'pair_cosine_frozen': round(float(vectors_before[0] @ vectors_before[1]), 4), 'pair_cosine_adapted': round(float(vectors_after[0] @ vectors_after[1]), 4), 'self_cosine_frozen_vs_adapted': [round(float(vectors_before[i] @ vectors_after[i]), 4) for i in range(2)]}}\n"
                "single_report = evaluation_report(pipe.fill_mask(cloze, top_k=TOP_K), [gold], sample_kind='one unseen SciTLDR abstract cloze' if not USE_BYOD else 'one BYOD test record')\n"
                "print({{'embedding_shift': embedding_shift, 'single_input_report_verdict': single_report['verdict'], 'clozes_with_changed_candidates': sum(r['frozen_candidates'] != r['adapted_candidates'] for r in rows), 'of': len(rows)}})\n"
                "with open('outputs/{stem}_cloze.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(rows)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = BERTMaskedLMPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "parity_records = test_records[:10]\n"
                "ppl_pair = (pipe.evaluate(parity_records, mask_rate=MASK_RATE, seed=SEED)['perplexity'], reloaded.evaluate(parity_records, mask_rate=MASK_RATE, seed=SEED)['perplexity'])\n"
                "cand_pair = [(r['adapted_candidates'], ' '.join(c['token'] for c in reloaded.fill_mask(r['cloze'], top_k=TOP_K)['candidates'])) for r in rows]\n"
                "vectors_reloaded = np.asarray(reloaded.embed(pair_texts, pooling=POOLING)['embeddings'], dtype=np.float32)\n"
                "parity = {{'perplexity_in_memory': round(ppl_pair[0], 6), 'perplexity_reloaded': round(ppl_pair[1], 6), 'identical_candidates': sum(a == b for a, b in cand_pair), 'of': len(cand_pair), 'embeddings_identical': bool(np.array_equal(vectors_after, vectors_reloaded))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert abs(ppl_pair[0] - ppl_pair[1]) < 1e-6 and parity['identical_candidates'] == parity['of'] and parity['embeddings_identical']\n\n"
                "weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'files': {{k: {{'name': v[0], 'bytes': v[1], 'sha256': v[2]}} for k, v in CORPUS_FILES.items()}}, 'license': CORPUS_LICENSE}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'cloze': cloze, 'gold_word': gold, 'fill_mask': {{k: filled[k] for k in ('candidates', 'top_k', 'n_tokens')}}, 'embed': {{k: embedding_result[k] for k in ('dim', 'pooling', 'normalized', 'n_tokens')}}, 'seconds': {{'fill_mask': fill_seconds, 'embed': embed_seconds}}}},\n"
                "    'comparison': comparison,\n"
                "    'clozes_before_after': rows,\n"
                "    'embedding_shift': embedding_shift,\n"
                "    'single_input_report': single_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen encoder already predicts the masked words of paper abstracts far better than a unigram model does "
        "(a masked perplexity near 15 against a floor in the thousands, top-1 accuracy above 50 % against a few percent "
        "— BooksCorpus and Wikipedia contain technical prose), and a bounded continuation of its own pre-training "
        "objective on 300 abstracts, touching only the last four encoder layers, lowers held-out masked perplexity by a "
        "few points in a few minutes on CPU, with a 113 MB adapter that reloads to identical likelihoods, candidates "
        "and embeddings. That is the claim: the adaptation contract can continue pre-training end to end on a real "
        "corpus, and the number it produces is read against the frozen model and a context-free floor rather than in "
        "isolation.\n\n"
        "Masked perplexity is intrinsic: it says how well the encoder predicts missing tokens of text it did not write, "
        "at one seeded 15 % masking of one seeded split of one corpus with no dispersion estimate. It is not embedding "
        "quality, retrieval performance or downstream accuracy, and a lower value does not make the candidates or the "
        "cosine table in Section 9 better — those are observations. The adapter changes the last layers, which every "
        "input shares, so every embedding shifts (Section 9 prints each vector's cosine to its frozen self) and the "
        "fill-mask `score` stays a softmax ranking signal, not a probability. Text is lower-cased and accent-stripped by "
        "the tokenizer; texts over 512 WordPiece tokens are rejected everywhere, never truncated; the checkpoint carries "
        "the gender and occupation associations the upstream card documents, and continued pre-training on a corpus does "
        "not remove them.\n\n"
        "Three things to carry to real data. **Floors first:** the unigram floor and the frozen masked perplexity on "
        "*your* held-out documents are the numbers to read before any adapted one. **Leakage:** de-duplicate texts "
        "across splits (the contract does this case-insensitively) and split by document collection or author when your "
        "documents come from one. **Downstream:** a better masked perplexity on your domain is a reason to *measure* "
        "your similarity or classification task with the adapted encoder, not evidence that it improved.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real corpus, validate "
        "the demonstrated dataset contract without leakage, execute both inference contracts and a bounded continued "
        "pre-training, evaluate by masked-token metrics against a trivial floor and the frozen model on an independent "
        "split, and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** "
        "establish benchmark superiority, embedding or cloze quality on any domain, a usable acceptance threshold, or "
        "production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_LAYERS = 1` and watch the gain "
        "shrink; set `EPOCHS = 4` and watch whether validation perplexity keeps falling or turns (the best epoch is kept "
        "either way); change `SEED` in Section 6 and read how much the masked metrics move with a different set of masked "
        "positions; switch `POOLING` and re-read the cosine shift; or bring your own documents through BYOD and read the "
        "unigram floor before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/bert-masked-lm-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/bert-masked-lm-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/bert-masked-lm-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/bert\n"
        "- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., NAACL 2019): https://arxiv.org/abs/1810.04805\n"
        "- TLDR: Extreme Summarization of Scientific Documents (Cachola et al., EMNLP Findings 2020; SciTLDR, Apache-2.0): https://arxiv.org/abs/2004.15011\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
