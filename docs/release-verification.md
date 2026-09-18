# Release verification

`tutorials/bert_masked_lm_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned corpus
  commit is the one allowed second hash);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `BERTMaskedLMPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_dataset` per split,
  `check_split_disjoint`, `write_dataset_csv`, `validate_inputs` with the two-mask refusal probe, `pipe.fill_mask` and `pipe.embed` with the sanity
  checks plus the frozen candidates of the unseen clozes, `pipe.unigram_baseline`, `pipe.evaluate` on the frozen
  model with the floor assertion and on the validation and test splits after adaptation with the perplexity
  assertion, `pipe.adapt` with its explicit hyperparameters, `trainable_layers=TRAINABLE_LAYERS` and
  `mask_rate=MASK_RATE`, the after-adaptation `pipe.embed`, the single-input `evaluation_report`,
  `pipe.save_artifact`, `BERTMaskedLMPipeline.from_artifact` and the reload-parity assertion, and the provenance fields `weight_format`, `weight_sha256` and the `corpus` block), the six expected
  `outputs/` paths, the learner-facing statements (uncalibrated fill-mask score, embeddings are representations,
  continued pre-training measured by the model's own objective, seeded 15 % masks, the add-one unigram floor,
  reject not truncate, no dispersion estimate, every embedding changes, named exclusions, the Apache-2.0 corpus
  licence) and the gated-off BYOD
  default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary
  path, a mutable `revision='main'`, direct `from transformers import` / `BertForMaskedLM` / `AutoTokenizer` /
  `model(**` / `from huggingface_hub import` / `urllib.request` / `safetensors` / `torch.optim` /
  `.backward(` / `pipe._model` / `log_softmax(` use **outside the carried module cells**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI installs only `pytest`, `ruff` and `numpy` plus the package without its model dependencies (no torch, no
transformers), runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`, `tests/test_import_boundary.py`,
`tests/test_notebook_parity.py`; injected runner, tokenizer, scorer and corpus fetcher, temporary manifests, no weights
— `tests/test_model_backed.py` is skipped without `transformers` and the snapshot). These are source/provenance and unit
checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; float32 either way) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy or torch that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/bert-base-uncased/` or the corpus cache `weights/scitldr/` (the standalone path writes the
   manifest itself, stages the missing file from the Hub, and fetches the three pinned SciTLDR-A files from the
   project repository, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `TOP_K = 5`, `POOLING = 'mean'`, `SEED = 0`, `MASK_RATE = 0.15`,
   `EPOCHS = 2`, `LEARNING_RATE = 5e-5`, `BATCH_SIZE = 8`, `TRAINABLE_LAYERS = 4`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `huggingface-hub==0.36.2`,
   `safetensors==0.8.0`, `numpy==2.5.3` (an interpreter restart after the install is expected where the runtime's
   preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `BERTMaskedLMPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `fetch_corpus`, `read_corpus`,
     `build_sample_dataset`, `filter_records`, `validate_dataset`, `check_split_disjoint`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv`, `record_seed`, `mask_positions`, `masked_metrics`,
     `unigram_baseline` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `google-bert/bert-base-uncased` at the immutable revision, and `verify_snapshot` returning its dict (8 files);
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with `source` `local-snapshot`;
   - Section 4: `fetch_corpus` fetching the three pinned files (3,155,015 / 1,124,865 / 1,204,107 bytes) from
     `raw.githubusercontent.com` into `weights/scitldr/`, 1,992 + 619 + 618 raw papers read, and the seeded draw of
     300 / 50 / 100 records with `check_split_disjoint` reporting no shared text and the three dataset digests
     `__DIG_TRAIN__` / `__DIG_VAL__` / `__DIG_TEST__`; `outputs/…_train.csv` written; the four dataset refusal probes each
     raising `ValueError`;
   - Section 5: the ceilings (`MAX_TEXT_CHARS` 4000, `MAX_TEXT_TOKENS` 512, `MAX_BATCH` 64, `MAX_TOP_K` 100,
     `VOCAB_SIZE` 30522, `HIDDEN_SIZE` 768, `MASK_TOKEN_ID` 103) surfaced; `validate_inputs` writing
     `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the two-mask probe);
     `fill_mask` on the unseen cloze and `embed` of two test abstracts with all six sanity checks `True`, then three
     unseen dev-abstract clozes filled by the frozen model;
   - Section 6: the unigram floor (≈ 1,174.6 perplexity, top-1 ≈ 4.2 % on the sample) and the frozen
     model's test metrics (≈ 15.09 perplexity, top-1 ≈ 53.3 %, top-5 ≈ 70.1 %, 2,983 masked
     positions) on CPU float32, with the cell's assertion that the masked counts agree and the frozen perplexity is
     below the floor;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 28,351,488 trainable of 109,514,298
     parameters, 300 training abstracts, and a two-epoch history with validation masked perplexity falling
     (12.89 → 11.15 → 10.87 in the recorded run; `best_epoch` 2);
   - Section 8: `pipe.evaluate` on the validation and test splits with the three-way comparison and
     `outputs/…_evaluation_report.json` written (the cell asserts the adapted test perplexity is below the frozen
     one — on the sample ≈ 12.59 versus ≈ 15.09);
   - Section 9: the three unseen clozes filled again by the adapted model and printed beside the frozen candidates
     and the gold word, the two abstracts re-embedded with the pair cosine and each vector's cosine to its frozen
     self printed, the single-input `evaluation_report` verdict `not-measurable`, `outputs/…_cloze.csv` written;
     `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors, manifest.json}` (64 tensors, about
     113 MB) and `BERTMaskedLMPipeline.from_artifact` reloading it with an identical ten-record perplexity,
     3/3 identical candidate lists and identical embeddings (the cell asserts all three); `outputs/…_result.json`
     written with `NOTEBOOK_SOURCE`, the model identity and licence, the snapshot block (`weight_format`,
     `weight_sha256`), the `corpus` block, the inference-contract items, the comparison, the before/after clozes, the
     embedding shift, the artifact digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `bert_masked_lm_colab.ipynb` (`E2E`) | `__LOCAL_ROW__` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/bert_masked_lm_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/bert_masked_lm_colab.ipynb`). Wall times are the sum of per-cell times reported by
the executor and include the model download where it occurred; they are measurements for the stated runtime, not
general estimates. No hosted run of the earlier `TASK-INFERENCE` notebook was ever recorded for this repository.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | `__LOCAL_ROW__` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | `__LOCAL_EXEC__` |

## Current status

The `E2E` notebook source is complete and passes all static checks, including the generator parity checks
(`--check` OK). A local pre-flight execution of the committed blob completed the whole default path on CPU — corpus
read from the cache, validation and split, both inference contracts, the unigram floor and the frozen masked metrics,
two epochs of continued masked-language-model training, held-out evaluation, before/after clozes and embeddings,
adapter export and reload parity — which catches defects but is **not** a supported runtime under REL1/REL10, and it
ran with the snapshot and the three SciTLDR files pre-staged, so neither the 440 MB Hub fetch nor the corpus download
has been exercised by this notebook end to end, and no hosted run of any revision of this notebook exists. The
repository stays at **Candidate** until a Colab or fresh-container run of the exact `E2E` release revision is
recorded above.
