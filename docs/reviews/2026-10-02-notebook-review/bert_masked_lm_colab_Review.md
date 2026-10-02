# BERT-Base Uncased Masked-LM E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/bert-masked-lm-pipeline`  
**Notebook:** `tutorials/bert_masked_lm_colab.ipynb`  
**Reviewed commit:** `ab7b54b0766b68a74aadc2c996a2855258de5edb` (`main`, confirmed with `gh api repos/kurtvalcorza/bert-masked-lm-pipeline/commits/main`)  
**Notebook Git blob:** `1d474b2ff63122dac48de715e7516217bd892546`, the blob committed at `1c8c93b` and executed in the recorded Kaggle run (later commits change the model card, tests and docs only; generator `--check` exits 0 at the reviewed commit)  
**Finding prefix:** `MLM`

## Executive assessment

The technical core is sound. The notebook carries its three modules byte for byte (generator `--check` and the release validator both exit 0). It pins and digest-verifies the model snapshot and a real, licensed, digest-pinned corpus, and it uses the release's own paper-disjoint split. It fits the unigram floor on training tokens only and scores the floor, the frozen model and the adapted model at identical seeded masks. Validation is used for epoch selection only. The metrics are explained as intrinsic, and reload parity is asserted. A direct CPU execution of the 11 code cells, verbatim and with the real BERT-Base weights, reproduced the recorded local pre-flight exactly: adapted test masked perplexity 12.5943 against frozen 15.093 and floor 1,174.57, and reload parity 14.966474 both ways.

Four problems stand in the way of `Ready for intended use`:

1. `Run all` in a fresh hosted runtime does not finish in one pass. The in-kernel pinned install replaces the preloaded NumPy, the guard stops the run with a restart instruction, and the recorded release run passed only on a second pass after a restart. The repository still registers the notebook as `Release-grade` (MLM-M1).
2. Every rerun the notebook prescribes reuses the already-adapted `pipe`. The BYOD instruction and three of the five optional experiments (Sections 6 and 7) score the adapted model as "frozen" and stack training under an epoch-0 "frozen model" label. The `TRAINABLE_LAYERS = 1` experiment reports the same gain as the default run and then stops at the reload-parity `AssertionError`. This was shown with the real model (MLM-M2).
3. The BYOD contract says "8..20,000 records", but the real minimum is 50, and the rejection messages name neither the split nor that minimum. An over-ceiling record passes Section 4 and fails in Section 6. A document whose opening sentence has no four-letter word crashes Section 5 with `max() iterable argument is empty` (MLM-M3).
4. The notebook is declared `GUIDED`, but the guided layer is absent. There is no how-to-use, roadmap, Input → Model → Output contract, prediction, checkpoint, worked answer, glossary or troubleshooting section, and nothing asks the learner to do more than run cells. The objectives are procedural, and 1,251 lines of carried code are not labelled as infrastructure (MLM-M4).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, and the opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** (2026-09-26), `ml-worker` `origin/main` |
| Intended audience | Not stated. The Prerequisites ask for basic Python plus knowledge of masked-language modelling, softmax, perplexity and cosine similarity |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU float32 by default, CUDA when present |
| Promised outcomes | One-pass `Run all` with no configuration edit; pinned install; carried modules; staged, digest-verified snapshot; digest-pinned SciTLDR-A, 300 / 50 / 100 paper-disjoint split, four refusal probes; fill-mask and embedding contracts with an input manifest and a two-mask refusal; unigram floor and frozen masked metrics; bounded continued MLM training of the last four layers with validation selection; held-out three-way comparison; clozes and embeddings before and after; safetensors adapter with reload parity; six `outputs/` files; BYOD "through the same … cells" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; carried modules from `src/bert_masked_lm_pipeline/` @ `ce404f4` |

### Evidence actually obtained

- **Source inspection:** all 25 cells (11 code), the carried `pipeline.py` (`evaluate`, `unigram_baseline`, `adapt`, `save_artifact`, `load_artifact`), `samples.py` (`validate_dataset`, `split_dataset`, `load_byod_dataset`), the generator and template, `README.md`, `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md`.
- **Documented execution evidence:** `docs/release-verification.md` and the archived run summary in the workspace (`.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-bert-masked-lm/v4/evidence/run_summary.json`). The run was on Kaggle Tesla T4, 2026-09-19, **blob `1d474b2f`, the reviewed blob**. Pass 1 failed in the install cell with `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` Pass 2, after the restart, ran 11/11 cells (`restarted_after_install_cell: true`): adapted test perplexity 12.66 against frozen 15.09, reload parity exact. No Colab run of this blob is recorded. The executed `.ipynb` was not opened; the run summary was.
- **Direct execution (this review):** `run_probes.py` on Windows, Python 3.12.14 (`eo-notebook-test` env, torch 2.13.0+cpu, transformers 4.57.6), CPU only. It used **the real BERT-Base weights** from the local digest-verified snapshot (hard-linked into a scratch working directory) and the local digest-verified SciTLDR cache, so no downloads happened and the runtime was not clean. Cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, which skips the install; the pins were not installed (torch 2.13 rather than 2.14). Every code cell was executed **verbatim from the notebook JSON** in one namespace. Only form-field literals were substituted, as an executor sets fields, and `google.colab.files.upload` was replaced by a fake for BYOD. Static checks: JSON parse, compile of all 11 code cells, blob id, generator `--check` (exit 0), `tools/validate_release_assets.py` (exit 0).
- **Not verified:** a Colab run of any kind; a one-pass hosted `Run all`; BYOD through the real upload widget; the pinned torch 2.14 build; learner understanding.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| Technical correctness | Strong on the default path. The revision is immutable, every file is digest-checked, the corpus is digest-pinned, no remote code runs, validation precedes model work, `adapt` is transactional, the artifact manifest and tensor set are checked before load, and parity is asserted. The default path reproduced on CPU with the real model (P1). Defects: the in-kernel install forces a restart on the recorded hosted image (MLM-M1). `pipe` is not reset before a prescribed rerun, and a changed `TRAINABLE_LAYERS` exports an artifact that is not the evaluated model (MLM-M2). BYOD split sizes and the token ceiling are checked after Section 4 or by a later stage (MLM-M3). |
| Scientific / experimental validity | Good default design: paper-disjoint release partition, training-only floor, identical seeded masks across the three systems with `n_masked` asserted equal, validation-only selection, test used once, an explicit no-dispersion caveat, and the intrinsic metric not confused with embedding quality. Weakness: every non-default run replaces the "frozen" reference with the adapted model (MLM-M2). |
| Promise fulfilment | Every default stage ran in the recorded Kaggle run and in this review's CPU run. Not met: "Run all … no configuration edit" in one pass (MLM-M1); the optional experiments "do not affect the default path" (MLM-M2); BYOD "through the same … cells" at the stated minimum (MLM-M3); BYOD "unseen" clozes (MLM-m4). |
| Learner experience | The prose is careful and accurate, and three "Look for" notes describe normal output. There is little else: no audience statement, how-to-use, roadmap, task contract, predictions, checkpoints, worked answers, glossary, troubleshooting, infrastructure labels or conclusion scaffold. The objectives read as procedures (MLM-M4). Runtime figures name no environment (MLM-m2). Doubled braces appear in the data contract (MLM-m5). |
| Spec conformance | Unmet applicable `MUST`s: RUN1, RUN10, ENV6, REL2, REL11 (MLM-M1); DAT13, DAT14, UX7, RUN9 (MLM-M2); DAT12, DAT19, VAL1, VAL6, REL12 (MLM-M3); UX1 (MLM-M4); UX12 (MLM-m2); VAL7 (MLM-m3); INF2 (MLM-m4). `SHOULD` gaps: GDL1–GDL15, UX5, UX8, UX9 (MLM-M4); EXE1, EXE2, UX10 (MLM-m3). `tutorials/README.md` marks the notebook `release-grade` although its `Run all` needs a restart (§27). |

## 3. Promise and objective tracing

| Claim (opening / section text) | Implementation | Observable result | Learner interpretation | Holds? |
|---|---|---|---|---|
| `Run all` in a fresh runtime, no intervention | Cell 3, in-kernel `pip install` + stale-import guard | Kaggle pass 1: `RuntimeError … Restart the runtime`; pass 2 ok | Learner must restart and rerun | **No** (MLM-M1) |
| Snapshot staged and digest-verified, no fallback | Cell 11 | Kaggle: 21 files fetched into a clean cache; P1: 8 files verified | Clear | Yes |
| Corpus digest-pinned; 300 / 50 / 100 paper-disjoint | Cell 13 | P1: three digests, splits 300/50/100, four refusals | Clear ("Look for") | Yes |
| Fill-mask and embedding contracts with input manifest and two-mask refusal | Cell 15 | P1: six sanity checks `True`, one finding | Clear | Yes |
| Floor and frozen model at the same masks | Cell 17 | P1: floor 1,174.57, frozen 15.093, `n_masked` equal | Clear | Yes (default path); **No** after a prescribed rerun (MLM-M2) |
| Bounded adaptation, validation selection | Cell 19 | P1: validation 12.89 → 11.15 → 10.87, best epoch 2, 224 s CPU | Clear | Yes (default path) |
| Held-out three-way comparison | Cell 21 | P1: 15.093 → 12.594; Kaggle T4: 15.09 → 12.66 | Clear | Yes |
| Adapter reloads with parity | Cell 23 | P1: 14.966474 both ways, 3/3, embeddings identical | Clear | Yes (default path); **No** after the `TRAINABLE_LAYERS = 1` experiment (MLM-M2) |
| Optional experiments "do not affect the default path" | Interpretation cell 24 | P2a, P2b: reruns score the adapted model as frozen; parity `AssertionError` | — | **No** (MLM-M2) |
| BYOD "passes through the same … cells"; "a dataset needs 8..20,000 records" | Cell 13 BYOD branch | P3a: 8–49 records rejected; P3b, P3c: failures in Sections 5 and 6 | — | **No** (MLM-M3) |
| Section 5 / 9 clozes are "unseen" | Cell 15 | Under BYOD they are `test_records[2:5]` (P3d) | — | **No** under BYOD (MLM-m4) |

| Learning objective (opening cell) | Learner activity | Evidence the objective is exercised |
|---|---|---|
| Install the pinned runtime; read what the carried modules guarantee; stage and digest-verify the snapshot | Run cells | Procedure, not a learning outcome (MLM-M4) |
| Fetch, validate and split a corpus without leakage | Run cell 13; read the "Look for" note | Observation only |
| Read the candidate and vector contracts correctly | Read cell 15 output | Observation only; nothing asks the learner to interpret a score |
| Read masked perplexity and top-k beside a floor; "understand what they do and do not measure" | Read Sections 6 and 8 prose | Exercised by reading; no prediction or checkpoint |
| Run bounded continued MLM training; evaluate on an independent split | Run cells 19 and 21 | Observation; the optional experiments that would exercise it are invalid (MLM-M2) |
| Compare clozes and embeddings before and after; export and reload an adapter | Run cell 23 | Observation |

## 4. Prioritized findings

### MLM-M1 — Major: fresh-runtime `Run all` needs a manual restart after the install cell, yet the notebook is registered `Release-grade`

**Cell/section:** Section 1 install cell (cell 3; `_INSTALL_GUARD` and the install-cell builder in `tools/build_notebook.py`, lines 47–69 and 452–471); the opening `Run all` paragraph (`tools/notebook_template.py` line 35); `README.md`, `STATUS.md` and `tutorials/README.md` release status; `docs/release-verification.md` procedure step 4 and the executor table.

**Observed issue:** Cell 3 runs `pip install` into the live kernel. It then compares every already-imported distribution with the newly installed version and raises `RuntimeError(... 'Restart the runtime, then rerun from the top.')` on any mismatch. Hosted images preload NumPy, and on Kaggle also `cuda-bindings`, at versions other than the pins (`numpy==2.5.3`), so the first pass stops there. The repository documents this as expected ("an interpreter restart after the install is expected where the runtime's preinstalled torch or numpy differ from the pins"). It still records the two-pass Kaggle run as a `PASSED` `Run all` and marks the notebook `Release-grade`. The opening promises the reverse: "no configuration edit … (NOTEBOOK_SPEC 2.0 §5)".

**Consequence:** A learner who chooses **Run all** on a fresh hosted runtime gets a `RuntimeError` in the first code cell and has to restart and run everything again. The release record presents a run that needed this intervention as one-pass evidence.

**Evidence:** Documented execution: `run_summary.json`, pass 1 `ok: false` with the error quoted in §1, then pass 2 `ok: true`, `restarted_after_install_cell: true`; `docs/release-verification.md` row "11/11 code cells ok (1 restart after install cell)". Source inspection of cell 3. Not re-executed here, because no hosted runtime was used.

**Recommended correction:** Adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass. The setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment. The kernel's preloaded NumPy and torch are never replaced, so no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement the pattern in the repository's notebook generator (`tools/build_notebook.py`, install cell and guard), regenerate, and re-qualify with a one-pass hosted `Run all`. Correct the release record so a restart-dependent run is not reported as a `Run all` PASS: until then, the status in `README.md`, `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md` returns to `Candidate`.

**Acceptance check:** A fresh Colab or Kaggle runtime, with the stock image's preloaded NumPy and torch, completes `Run all` of the regenerated blob in one pass with no restart and no `RuntimeError`. The run is recorded with its blob id. No document calls a run with `restarted_after_install_cell: true` a `Run all` PASS.

**Spec:** RUN1, RUN10, ENV6, REL2, REL11 (MUST); §27 (release-grade marking); §25.7.

### MLM-M2 — Major: prescribed reruns reuse the adapted `pipe`: "frozen" scores are the adapted model, training stacks, and the `TRAINABLE_LAYERS = 1` experiment reports the old gain and then fails reload parity

**Cell/section:** The opening BYOD instruction ("set `USE_BYOD = True` in Section 4 and re-run from that cell", `tools/notebook_template.py` line 39). "Optional experiments (they do not affect the default path)" in the Interpretation (line 467: `TRAINABLE_LAYERS = 1`, `EPOCHS = 4`, change `SEED` in Section 6, switch `POOLING`, BYOD). Sections 5–9 (cells 15–23). Carried `adapt` and `save_artifact` (`src/bert_masked_lm_pipeline/pipeline.py`, around lines 554–740).

**Observed issue:** `pipe` is built once, in Section 3. `adapt()` starts from the model's **current** weights. It uses `initial_state` only to roll back on an exception, and it still labels epoch 0 `"frozen model"`. After the default run:
- rerunning Section 6 (the `SEED` experiment, or BYOD rerun "from that cell") scores the adapted model as `frozen_test`. Only the printed `adapted: True` field shows it;
- Section 5's "before" clozes and embeddings, rerun under BYOD or the `POOLING` experiment, come from the adapted model;
- rerunning Section 7 with `TRAINABLE_LAYERS = 1` (rerunning with `EPOCHS = 4` stacks the same way) trains on top of the four already-adapted layers. Epoch 0, "frozen model", reports the adapted model's validation perplexity. If the new epochs do not improve on it, `best_epoch` is 0 and the "1-layer" result equals the 4-layer one, so the gain the learner was told to watch shrink does not move. `save_artifact` writes only layer 11, while `pipe` still carries the trained layers 8–10. The reloaded artifact therefore differs from the evaluated model, and cell 23 stops at `assert` with an empty `AssertionError` after it has written the artifact.

Only the cell-level instruction "restart and Run all" would give valid numbers, and neither the experiments nor BYOD say to do that.

**Consequence:** Each experiment the notebook suggests, and the BYOD transfer path, yields "frozen" and "adapted" numbers that mean something other than their labels. The `TRAINABLE_LAYERS` experiment either teaches the wrong conclusion (no change with fewer layers) or crashes with a bare assertion.

**Evidence:** Direct execution with the real model on CPU, cells run verbatim:
- **P2a:** after the default path, cells 19–23 were rerun with `TRAINABLE_LAYERS = 1`, `EPOCHS = 1`. Layers 8–11 were already adapted before the rerun. Epoch 0 "frozen model" val perplexity was 10.868 against a true frozen 12.887. Epoch 1 reached 10.92, so `best_epoch` was 0. Section 8 printed the default run's comparison unchanged (adapted 12.5943). The artifact holds layer `[11]` only. Parity: 14.966 in memory against 15.580 reloaded, 0/3 candidates identical, embeddings differ, `AssertionError`.
- **P2b:** rerunning cell 17 with `SEED = 1` gave `adapted: True` and a "frozen" perplexity of 11.98. A freshly loaded pretrained model scores 14.22 at the same seed.
- **P2c (control):** with `pipe` rebuilt from the verified files before Section 6, the same 1-layer, 1-epoch setting gives epoch 0 at 12.887, adapted test 13.74 (delta −1.35 against −2.50 for four layers), and parity exact, 3/3.

**Recommended correction:** Make each pass start from the pretrained base. Either have Section 6 (or each rerun entry point) rebuild `pipe` with `BERTMaskedLMPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` from the already-verified files, as the P2c control does, or add a `reset_adapter()` that restores base tensors and clears `self.adapter`. Alternatively, `adapt()` can refuse when `self.adapter is not None`, and Section 6 can refuse to report an adapted pipe as frozen. Make `save_artifact` refuse, or include every encoder tensor that differs from the base, when the adapted set and `trainable_names` disagree, and give the parity assert a message that names the cause. For each experiment and for BYOD, state exactly which section to rerun from. Generator: `tools/notebook_template.py` lines 39 and 467 and the Section 5/6/7 cells; `pipeline.py` `adapt` / `save_artifact`.

**Acceptance check:** After the default path, rerunning as each instruction says gives Section 6 `adapted: False`, with perplexity equal to a freshly loaded pretrained model at the same seed. A `TRAINABLE_LAYERS = 1` rerun starts from the base (epoch 0 equal to the frozen validation perplexity), its comparison differs from the 4-layer run, and cell 23 passes parity.

**Spec:** DAT13, DAT14, UX7, RUN9 (MUST); VER5 (context); GDL10, UX5 (SHOULD).

### MLM-M3 — Major: the BYOD contract's stated limits are not the enforced ones, and bad input fails after Section 4 with unhelpful messages

**Cell/section:** The Prerequisites data contract ("a dataset needs 8..20,000 records", template line 120) and the Section 4 BYOD branch (cell 13, around template line 150); `split_dataset` and `validate_dataset` (`samples.py`); `_record_ids` (`pipeline.py`); `cloze_of` in cell 15.

**Observed issue:**
1. Cell 13 calls `split_dataset` (test 20 %, validation 15 %) and then `validate_dataset(part)` on **each split**, and each split needs at least 8 records. Every file of 8 to 49 records is rejected, although the stated contract accepts it. The messages are `split leaves 5 training records; at least 8 are required` or `4 records; 8..20000 are required`. The second names no split and contradicts the file's actual size.
2. Section 4 checks the 4,000-character guard but not the 512-token ceiling, which the Prerequisites call "refused, not truncated, everywhere". A 3,900-character record of 3,686 tokens passes Section 4 and Section 5, which call the model. It then fails in Section 6 (`unigram_baseline`) with `record u0059 has 3686 tokens; ceiling is 512`. The message names the record, but it comes after model execution has begun and without a recovery step.
3. `cloze_of` takes `max()` over the opening sentence's words of four or more letters. A BYOD document whose first sentence has none (for example "We do it.") crashes Section 5 with `ValueError: max() iterable argument is empty`.

**Consequence:** A learner following the stated contract with a small corpus is told it is too small with a number that does not match their file. A learner with long or terse documents fails mid-run without being told what to fix. The BYOD promise is not met at its own boundaries.

**Evidence:** Direct execution, cell 13 verbatim with `USE_BYOD = True` and a fake JSONL upload of real abstracts (P3a): 8, 10, 20, 30, 37, 38, 40, 45 and 49 records were rejected, and 50 is the smallest accepted. P3b: the over-ceiling record landed in `train`; cells 13 and 15 passed, and cell 17 raised the error quoted above. P3c: 60 records each starting "We do it." passed cell 13, and cell 15 raised `max() iterable argument is empty`. Positive control P3d: a 60-paragraph `.txt` file (12 / 9 / 39 split) ran cells 13–23 to the end with exact parity. Real upload widget: **not verified**.

**Recommended correction:** Do all of this in Section 4, before any model call:
- state and enforce the real minimum (or validate validation and test with `min_records=1` while keeping the training minimum), and name the split in the message;
- tokenize every BYOD text (the tokenizer is loaded in Section 3) and reject over-ceiling records by id, with a recovery hint (split the document);
- make `cloze_of` skip records without a usable word, or choose clozes from records that have one, and say so.

Generator: cell 13 and cell 15 in `tools/notebook_template.py`, and the Prerequisites text at line 120.

**Acceptance check:** A 20-record BYOD file either runs through Section 4 or is rejected in Section 4 with a message naming the split and the real minimum. A file with one record over 512 tokens is rejected in Section 4 naming its id. A file whose documents open with a short sentence runs Section 5 or is rejected in Section 4 with a named reason.

**Spec:** DAT12, DAT19, VAL1, VAL6, REL12 (MUST); UX10 (SHOULD).

### MLM-M4 — Major: declared `GUIDED`, but the guided layer and any structured learner activity are absent; objectives are procedural; carried code is not marked as infrastructure

**Cell/section:** The opening and Prerequisites (cells 0–1), every section, the Interpretation (cell 24); the carried module cells 5, 7 and 9 (117 + 837 + 297 lines); `tools/notebook_template.py` and the opening builder in `tools/build_notebook.py` (line 434).

**Observed issue:** A marker scan of all 14 markdown cells (P0) found no intended-learner statement, no **How to use this notebook**, no roadmap, no Input → Model/System → Output contract, no prediction prompt, no "What to notice" or "Check your reasoning", no `<details>` worked answer, no glossary, no troubleshooting section, no "Infrastructure" label, no `cellView: form`, and no conclusion template. There are three "Look for" notes. The learning objectives begin "install the pinned runtime; read what the carried … modules guarantee; stage and digest-verify the immutable upstream snapshot", which are procedures, and one is "understand". The optional experiments are a single sentence. Their rerun instructions are missing, and following them gives invalid results (MLM-M2). Section 2 puts 1,251 lines of carried code in front of the first model call with no label saying the learner may run it without reading it.

**Consequence:** The notebook works as a careful reference, but a self-paced learner of the stated `GUIDED` mode only runs cells. They are never asked to predict, change, explain or conclude anything, and they cannot tell which of 1,251 lines matter. Objectives such as reading perplexity "beside a floor" are not exercised. This is a learner-facing gap, separate from spec conformance: NOTEBOOK_SPEC 2.2 says existing notebooks do not become nonconformant solely for missing the guided layer, but UX1 (objectives that correspond to code) is a MUST.

**Evidence:** Source inspection of all 25 cells; P0 `guided_markers_in_markdown`, `cellView_form_cells: []`, `carried_cell_lines`.

**Recommended correction:** Follow NOTEBOOK_SPEC 2.2 §3.5 and the §25.13 reference notebook:
- state the intended learner, then add how-to-use, a roadmap and the Input → Model → Output contract;
- rewrite the objectives as observable actions, for example "explain why the unigram floor is far above the frozen perplexity" or "predict and then check how many perplexity points four trainable layers gain over one";
- add a prediction before Sections 6 and 8, a "What to notice" note after them, and collapsible worked answers;
- after MLM-M2 is fixed, turn one experiment (for example `TRAINABLE_LAYERS`) into a Predict → Change one thing → Run → Observe → Explain activity with exact rerun instructions;
- add a glossary (masked perplexity, bits per token, top-k, WordPiece, unigram floor, continued pre-training), a troubleshooting section (install, download, memory, BYOD rejections) and a conclusion template;
- title the carried and setup cells `# @title Infrastructure: …` with `cellView: form`.

Generator: `tools/notebook_template.py` and `tools/build_notebook.py`.

**Acceptance check:** The regenerated notebook has an audience statement, how-to-use, roadmap and task contract. Each objective names an observable action that some cell or prompt exercises. There is a prediction and a worked answer at each principal result, at least one Predict → Change → Run → Observe → Explain activity with rerun instructions, a glossary, troubleshooting and a conclusion template, and the carried cells are labelled Infrastructure and collapsed.

**Spec:** UX1 (MUST); UX5, UX8, UX9, GDL1–GDL15 (SHOULD).

### MLM-m1 — Minor: a result-dependent assert stops a legitimate negative experiment before export

**Cell/section:** Section 8, cell 21 (`assert adapted_test['perplexity'] < frozen_test['perplexity']`, template line 358).

**Observed issue:** An experiment the notebook invites can legitimately fail to lower test perplexity: fewer layers, a different seed, a different learning rate, or a BYOD corpus. The assert then stops the run before Section 9, so nothing is exported or reloaded, and the error is a bare `AssertionError`. P2a shows that a non-improving epoch is realistic: the 1-layer epoch 1 (10.92) was worse than its starting point (10.87).

**Consequence:** A negative result, which the interpretation section should help the learner read, is presented as a crash.

**Evidence:** Source inspection; P2a history. A fresh-start negative result was not produced in this review.

**Recommended correction:** Replace the assert with a printed verdict (improved / not improved), and carry on to export and reload.

**Acceptance check:** A run in which adapted perplexity ≥ frozen completes Section 9 and prints a "not improved" verdict.

**Spec:** RUN9 (MUST, for optional runs); GDL8, GDL14 (SHOULD).

### MLM-m2 — Minor: runtime and result figures name no environment and disagree with the recorded runs

**Cell/section:** Opening ("about four minutes of model time" on CPU, template line 35); Prerequisites ("about 102 s per training epoch", "about 7 s to score"); Section 6 ("about seven seconds on CPU"); Section 7 ("about 102 s of training … per epoch on CPU"); `README.md` ("15.09 → 12.59 masked perplexity in the recorded run").

**Observed issue:** The figures come from "the build record", which is the local Windows pre-flight, but none says which environment it was. The release run (Kaggle T4) measured adapted 12.66, not the 12.59 that `README.md` attributes to "the recorded run". This review's CPU run took 224 s for two epochs (about 112 s per epoch). No hosted-CPU run exists.

**Consequence:** A learner on a hosted CPU cannot judge whether a slow run is normal, and a reader cannot reconcile the README number with the release record.

**Evidence:** Source inspection; `docs/release-verification.md`; P1 timings.

**Recommended correction:** Label each figure with its environment (for example "local Windows CPU pre-flight" or "Kaggle T4"), mark hosted-CPU times as estimates, and cite the release run's 12.66 (GPU) next to the CPU 12.59.

**Acceptance check:** Every runtime or result figure in the notebook and README names its environment or is labelled an estimate, and matches a recorded run.

**Spec:** UX12 (MUST); ENV8.

### MLM-m3 — Minor: BYOD is a Colab-only upload with no location field; an empty upload gives a bare `StopIteration`; de-duplication is not reported

**Cell/section:** Section 4, cell 13 BYOD branch; `split_dataset` (`samples.py`).

**Observed issue:** BYOD imports `google.colab.files`, although "Jupyter" is a stated runtime, and has no path field an executor or Jupyter user can set. Cancelling or submitting an empty upload raises `StopIteration` from `next(iter(uploaded.items()))`. `split_dataset` silently drops case-insensitive duplicate texts, and the count dropped is never printed.

**Consequence:** Jupyter users cannot use BYOD. An empty upload gives no recovery hint. A learner does not learn that some of their records were dropped.

**Evidence:** Source inspection; P3a `empty_upload: StopIteration`.

**Recommended correction:** Add `BYOD_PATH = ''  # @param {type:"string"}`, read it when set and fall back to the upload widget only in Colab; guard the empty upload with a message; print the number of duplicates removed.

**Acceptance check:** With `BYOD_PATH` set, cell 13 reads the file without importing `google.colab`. An empty upload prints a recovery message. A file with duplicates prints the number dropped.

**Spec:** VAL7 (MUST); EXE1, EXE2, DAT16, UX10 (SHOULD).

### MLM-m4 — Minor: under BYOD, the "unseen" clozes are test records

**Cell/section:** Section 5, cell 15 (`unseen_records = … test_records[2:5]` when `USE_BYOD`, template line 213), reused in Section 9.

**Observed issue:** With BYOD, the three clozes ids `unseen-00…02` are taken from the test split, and the prose never says so. On the sample path they come from `dev` records outside every split.

**Consequence:** A BYOD learner is told that the before/after cloze demonstration uses unseen text when it reuses scored test documents.

**Evidence:** Source inspection; P3d `unseen_clozes_are_test_records: true`.

**Recommended correction:** Hold three BYOD records out of the split for the clozes, or label them as test records in the ids and the prose.

**Acceptance check:** Under BYOD, the cloze ids and text say where the records came from, and none is claimed to be unseen unless it is outside every split.

**Spec:** INF2 (MUST where task semantics permit).

### MLM-m5 — Minor: doubled braces in the learner-facing data contract

**Cell/section:** The opening BYOD paragraph and Prerequisites (cells 0 and 1; `tools/notebook_template.py` lines 40 and 120).

**Observed issue:** Markdown strings that are never passed through `str.format` keep the escaped braces, so the rendered text reads `{{id, text}}` and the id pattern reads `[A-Za-z0-9_.:-]{{1,64}}`. That pattern is wrong as written: `{{1,64}}` is not the enforced `{1,64}`.

**Consequence:** The schema a BYOD user is asked to match is shown with literal doubled braces and an incorrect regex.

**Evidence:** Source inspection; P0 `doubled_braces_in_markdown: [0, 1]`.

**Recommended correction:** Use single braces in the template strings that are not formatted, or format them consistently.

**Acceptance check:** No markdown cell in the regenerated notebook contains `{{` or `}}`.

**Spec:** DAT12 (MUST: the schema must be stated correctly).

### Suggestions

- **MLM-S1:** Update the declared `notebook_spec` from 2.0 to 2.2 in metadata, the opening, `NOTEBOOK_SOURCE` and the validator, once the guided-layer work in MLM-M4 lands.
- **MLM-S2:** In Section 8, note that the adapted perplexity differs slightly between CPU and CUDA (12.59 and 12.66 in the recorded runs), so learners do not read the difference as a defect (ENV8).
- **MLM-S3:** Extend reload parity from ten records' perplexity, three clozes and two embeddings to the whole test split, plus a direct tensor-equality check of the artifact against `pipe` (VER4).
- **MLM-S4:** Print a bootstrap interval over the 100 test abstracts for the frozen-to-adapted perplexity and accuracy deltas, which makes the no-dispersion caveat concrete.

## 5. Readiness

**Needs revision.** Four Major findings are open. Unmet applicable `MUST`s: RUN1, RUN10, ENV6, REL2, REL11 (MLM-M1); DAT13, DAT14, UX7, RUN9 (MLM-M2); DAT12, DAT19, VAL1, VAL6, REL12 (MLM-M3); UX1 (MLM-M4); RUN9 (MLM-m1); UX12 (MLM-m2); VAL7 (MLM-m3); INF2 (MLM-m4); DAT12 (MLM-m5). The `Release-grade` registry status is not supported, because the only hosted run needed a restart.

What is met: on the recorded Kaggle T4 run (after the restart) and on this review's CPU run with the real model, every default stage completes, with the stated metrics and exact reload parity.

Gates remaining after the fixes:
- a one-pass hosted `Run all` of the regenerated blob, using the uv isolated environment;
- a BYOD positive run and a BYOD negative run through Section 9 on a hosted runtime (REL12);
- a rerun of an optional experiment, as instructed, showing a frozen reference equal to a fresh load and passing reload parity;
- a corrected release record and status in all four documents.

## 6. Verified versus inferred

- **Verified by direct execution (CPU, this review, real BERT-Base weights, pre-staged snapshot and corpus, install skipped):** notebook parse and compile (11/11 code cells), blob `1d474b2f`, generator `--check` exit 0, release validator exit 0 (P0). Default path cells 3–23 verbatim, 11/11, with adapted 12.5943 against frozen 15.093 and parity exact (P1). Rerun state, the 1-layer parity failure and the fresh-start control (P2a–P2c). BYOD minimum 50, empty upload, token ceiling, short opening sentence (P3a–P3c). BYOD positive path through Section 9 and test-record clozes (P3d).
- **Verified from documented execution:** the Kaggle T4 run of this exact blob, which failed pass 1 at the install guard and passed pass 2 after a restart, read from the archived `run_summary.json`.
- **Inferred, not executed:** that a fresh Colab runtime also trips the install guard (the Colab image preloads NumPy; no Colab run of this blob exists); behaviour through the real upload widget; behaviour with the pinned torch 2.14; learner understanding.
- **Finding most likely to be wrong:** MLM-M4's severity. The prose is accurate and dense, and a reader could argue that for a technically literate audience the missing guided layer is a Minor conformance gap rather than a Major learner problem. It is rated Major because the notebook declares `GUIDED` and none of its objectives is exercised by anything other than running cells.

Probe ZIP: `bert_masked_lm_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
