# Pre-registration: configuration re-selection on validation and a fresh test (2026-10-17)

This file is committed **before** any grid configuration is trained or any fresh-test abstract is scored. It
answers review finding T5-1 (tutorial-triage 2026-10-15, row 5): the notebook's `TRAINABLE_LAYERS = 4` was picked by
a three-configuration sweep scored on the **test** split (two layers at 1e-4: 12.96, four layers at 1e-4: 12.51,
four layers at 5e-5: 12.59, adapted test perplexity, local Windows CPU; text added in `3ff35cd`). That sweep is
superseded by the protocol below. The owner's ruling of 2026-10-17 fixes every choice here; nothing in it is tuned
after the run.

## Fixed for every configuration

- `EPOCHS` 2, `BATCH_SIZE` 8, `MASK_RATE` 0.15, `SEED` 0, `SPLIT_SEED` 42.
- AdamW (weight decay 0.01), gradient clipping at 1.0, no scheduler, 100 % `[MASK]` replacement.
- The same 300 training and 50 validation abstracts (`build_sample_dataset(seed=42)`, unchanged).
- Epoch selection inside `adapt()` is unchanged: the epoch with the lowest validation masked perplexity is kept.

## The grid (three configurations, in preference order)

| Config | `TRAINABLE_LAYERS` | `LEARNING_RATE` | Role |
|---|---|---|---|
| **B** | 2 | 5e-5 | smaller model change (most preferred) |
| **A** | 4 | 5e-5 | the previous default |
| **C** | 4 | 1e-4 | the earlier test-sweep argmin, re-checked on validation |

## Selection rule

1. **Metric:** each configuration's best-epoch validation masked perplexity, `min(history[*].val.perplexity)` over
   epochs 1..2, on the 50 validation abstracts at `SEED` 0.
2. **Margin:** let v* be the lowest of the three. Pick the **first** configuration in the order **B → A → C** whose
   metric is at most v* + τ, with **τ = 0.10** perplexity points.
3. **Keep frozen unless beaten:** if the picked configuration's metric is not below the frozen model's validation
   perplexity (epoch 0) minus τ, the frozen model is kept ("no adaptation kept").
4. Configurations that are not picked are **never scored on any test split**.

## Fresh test (scored only under the pre-registered settings)

- **Pool:** SciTLDR-A `dev.jsonl` (the pinned commit `5ccad9c`) after `filter_records` (200..2,000 characters,
  case-insensitive de-duplication), minus every text (case-insensitive) in the 300 training, 50 validation and 100
  old-test abstracts, minus the three Section 4 cloze abstracts. The review counted **559** abstracts in that pool
  (`pools.json`); the code asserts the count.
- **Draw:** 100 abstracts, a seeded shuffle of that pool with `random.Random(20261017)`; ids `fresh-test-0000..0099`.
- **Disjointness (asserted in code):** no paper id and no lower-cased text of the fresh test appears in the training,
  validation or old-test splits or among the clozes. SciTLDR-A's members share no paper id but do share four exact
  abstract texts (2 train/test, 1 train/dev, 1 dev/test; review T5-7); excluding every used text covers them.
- **Pinning:** the sorted fresh-test paper ids and their SHA-256 are written into `samples.py` after the draw and
  before any scoring; the notebook refuses a draw that does not reproduce that digest.
- **Never-scored status:** no recorded perplexity run scored a dev abstract outside the 50 validation abstracts; the
  three clozes were only filled with `fill_mask`. This rests on the code of every revision since `3ff35cd` (the
  sample draws validation only from `dev`, and the 2026-10-03 BYOD check used SciTLDR **test** abstracts). The
  build record of the original sweep (`sci-build-2026-09-18`) no longer exists, so this cannot be checked against it.
- **Scoring:** the picked configuration (or the frozen model, under rule 3) is scored on the fresh test only under the
  settings above. That number is the reported test result; a later run with identical settings is a reproduction of
  it, and a run with any other setting is exploratory and does not score the fresh test.

## The old test

The 100-abstract test split drawn from `test.jsonl` stays in the notebook, relabelled the **exposed development
test**: it was seen by the superseded configuration sweep. The picked configuration is scored on it under the same
settings and the number is reported with that label, never as the held-out result.

## Run budget

The grid runs **on** by default if the estimated hosted T4 time for Run all stays under about 55 minutes (estimate
from the 2026-10-03 Colab T4 run: 25.5 s per four-layer adaptation, 132.7 s for the whole notebook, so three grid
adaptations plus the re-training of the pick add roughly 100 s). The CPU build record is produced on a local
Windows CPU with the pinned `torch 2.14.0+cpu` and is labelled as such, pending a hosted run.

## Wording amendment (2026-10-17, review round 2)

The phrases "scored once" above were reworded to "scored only under the pre-registered settings" (fleet
score-once rule, 2026-10-17). No setting, grid entry, rule, seed or pool changed; the record below is unchanged.

## Record

The three validation numbers, v*, τ, the picked configuration, the frozen validation perplexity, the single
fresh-test score and the exposed development test score are recorded in `docs/release-verification.md`.
