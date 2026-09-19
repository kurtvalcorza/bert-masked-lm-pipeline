"""Masked-language-model metrics over a held-out text corpus, seeded masking, and the unigram baseline.

A record is scored by masking a seeded 15 % of its WordPiece tokens (never `[CLS]`/`[SEP]`, at least one
token) and reading, at every masked position, the model's negative log-likelihood of the original token and
the original token's rank in the model's prediction. Those aggregate into **masked perplexity**
(`exp(mean NLL)` over all masked positions, token-weighted), **bits per masked token** (`mean NLL / ln 2`)
and **top-1 / top-5 accuracy** (the original token is the first / among the first five candidates). The
masks are a fixed function of the seed and the record's id, so the frozen and adapted models are scored on
exactly the same positions. The **unigram baseline** predicts every masked position with an add-one-smoothed
unigram distribution fitted on the training tokens — the floor a model that ignores context reaches.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

DEFAULT_MASK_RATE = 0.15
METRIC_DEFINITIONS = {
    "perplexity": (
        "exp of the mean negative log-likelihood (natural log) of the original token at every masked "
        "position of the held-out records, all masked positions weighted equally; lower is better"
    ),
    "bits_per_token": "the same mean negative log-likelihood divided by ln 2",
    "mean_nll": "the mean negative log-likelihood at the masked positions in nats",
    "top1_accuracy": "fraction of masked positions whose original token is the model's first candidate",
    "top5_accuracy": "fraction of masked positions whose original token is among the first five candidates",
    "masking": (
        "a seeded 15 % of each record's WordPiece tokens (never [CLS]/[SEP], at least one token) replaced "
        "by [MASK] — the same positions for every model scored under the same seed"
    ),
}


def record_seed(record_id: str, seed: int) -> int:
    """A stable per-record seed so the masked positions depend only on the record and the base seed."""
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def mask_positions(n_tokens: int, *, rate: float = DEFAULT_MASK_RATE, seed: int) -> list[int]:
    """Sorted positions to mask among 1..n_tokens-2 (the interior; position 0 is [CLS], the last is [SEP])."""
    if not 0.0 < rate <= 1.0:
        raise ValueError("rate must be in (0, 1]")
    interior = n_tokens - 2
    if interior < 1:
        raise ValueError("a record needs at least one token between [CLS] and [SEP]")
    count = max(1, round(interior * rate))
    return sorted(random.Random(seed).sample(range(1, n_tokens - 1), count))


def masked_metrics(scores: Sequence[Sequence[tuple[float, int]]]) -> dict[str, Any]:
    """Aggregate per-record lists of (NLL in nats, rank of the original token, 1 = first) pairs."""
    if not scores:
        raise ValueError("no records to score")
    total = 0.0
    n_masked = 0
    top1 = 0
    top5 = 0
    for record_scores in scores:
        if not record_scores:
            raise ValueError("a record scored no masked positions")
        for nll, rank in record_scores:
            if rank < 1:
                raise ValueError("rank must be 1 or more")
            total += float(nll)
            n_masked += 1
            top1 += rank == 1
            top5 += rank <= 5
    mean_nll = total / n_masked
    return {
        "n_records": len(scores),
        "n_masked": n_masked,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "bits_per_token": mean_nll / math.log(2),
        "top1_accuracy": top1 / n_masked,
        "top5_accuracy": top5 / n_masked,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def unigram_masked_scores(
    train_ids: Sequence[Sequence[int]], targets: Sequence[Sequence[int]], vocab_size: int
) -> list[list[tuple[float, int]]]:
    """(NLL, rank) of each masked original token under an add-one unigram model fitted on `train_ids`.
    The rank counts vocabulary entries with a strictly higher count plus one, so ties favour the target."""
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    counts: Counter[int] = Counter()
    for ids in train_ids:
        counts.update(int(t) for t in ids)
    total = sum(counts.values()) + vocab_size
    log_total = math.log(total)
    ordered = sorted(counts.values(), reverse=True)
    out = []
    for record_targets in targets:
        record = []
        for target in record_targets:
            count = counts.get(int(target), 0)
            higher = sum(1 for c in ordered if c > count)
            record.append((log_total - math.log(count + 1), higher + 1))
        out.append(record)
    return out


def unigram_baseline(
    train_ids: Sequence[Sequence[int]], targets: Sequence[Sequence[int]], vocab_size: int
) -> dict[str, Any]:
    """Masked metrics of the context-free unigram model — the floor a masked language model must beat."""
    result = masked_metrics(unigram_masked_scores(train_ids, targets, vocab_size))
    result["baseline"] = "add-one-smoothed unigram model fitted on the training tokens (ignores context)"
    return result
