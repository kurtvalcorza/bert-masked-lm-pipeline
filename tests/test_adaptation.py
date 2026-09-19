"""Offline tests for the text-corpus dataset contract, the pinned SciTLDR abstract reader, seeded masking,
masked-token metrics and the unigram baseline, BYOD loaders (CSV / JSON / JSONL / TXT), the injected-scorer
evaluation path, artifact-manifest rejections and adapt() argument validation. Nothing here imports torch or
transformers; the corpus is three crafted JSON-Lines files served through an injected fetcher."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from bert_masked_lm_pipeline import (
    ARTIFACT_FORMAT,
    CLS_TOKEN_ID,
    ENCODER_LAYERS,
    MASK_TOKEN_ID,
    MAX_TEXT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_SPLIT,
    SEP_TOKEN_ID,
    VOCAB_SIZE,
    WEIGHT_SHA256,
    BERTMaskedLMPipeline,
    build_sample_dataset,
    check_split_disjoint,
    dataset_digest,
    fetch_sample_dataset,
    load_byod_dataset,
    mask_positions,
    masked_metrics,
    record_seed,
    split_dataset,
    unigram_baseline,
    validate_dataset,
    write_dataset_csv,
)
from bert_masked_lm_pipeline import metrics as mt
from bert_masked_lm_pipeline import pipeline as pl
from bert_masked_lm_pipeline import samples as sm
from bert_masked_lm_pipeline.samples import fetch_corpus, filter_records, read_corpus

ABSTRACTS = [
    (
        "p01",
        [
            "We study graph neural networks for molecule property prediction.",
            "Our model beats three baselines on two benchmarks.",
            "Code is released.",
        ],
    ),
    (
        "p02",
        [
            "Transformers are hard to train on small data.",
            "We propose a curriculum that orders examples by length.",
            "Accuracy improves by four points.",
        ],
    ),
    (
        "p03",
        [
            "Reinforcement learning agents forget old tasks.",
            "We add a replay buffer with prioritised sampling.",
            "Forgetting drops by half.",
        ],
    ),
    (
        "p04",
        [
            "Speech recognition degrades with accents.",
            "We fine-tune on accented data with adapters.",
            "Word error rate falls.",
        ],
    ),
    (
        "p05",
        [
            "Image captioning models hallucinate objects.",
            "We penalise captions naming absent objects.",
            "Hallucination rate halves.",
        ],
    ),
    (
        "p06",
        [
            "Sparse attention scales to long documents.",
            "We route tokens to experts by locality-sensitive hashing.",
            "Memory drops fourfold.",
        ],
    ),
    (
        "p07",
        [
            "Tabular data resists deep learning.",
            "We pretrain a transformer on synthetic tables.",
            "It matches gradient boosting.",
        ],
    ),
    (
        "p08",
        [
            "Machine translation for low-resource languages lacks data.",
            "We back-translate monolingual text.",
            "BLEU rises by six.",
        ],
    ),
    (
        "p09",
        [
            "Protein structure prediction is costly.",
            "We distil a large model into a small one.",
            "Speed improves ten times.",
        ],
    ),
    (
        "p10",
        [
            "Robots grasp unfamiliar objects poorly.",
            "We learn grasps from simulated point clouds.",
            "Success rate reaches ninety percent.",
        ],
    ),
    (
        "p11",
        [
            "Recommender systems amplify popularity bias.",
            "We reweight the loss by item frequency.",
            "Long-tail recall improves.",
        ],
    ),
    (
        "p12",
        [
            "Code models struggle with long files.",
            "We add retrieval over the repository.",
            "Completion accuracy improves.",
        ],
    ),
]


def _rows(prefix="", papers=ABSTRACTS):
    """Crafted SciTLDR rows; the prefix goes into the first sentence so the three members are text-disjoint
    like the real release. `target` and `title` are carried but unused by this row."""
    return [
        {
            "paper_id": f"{prefix}{pid}",
            "source": [f"{prefix.upper()}{src[0]}", *src[1:]] if prefix else src,
            "target": ["unused"],
            "title": f"Title {pid}",
        }
        for pid, src in papers
    ]


def _records(prefix="r"):
    return [{"id": f"{prefix}{i:03d}", "text": " ".join(src)} for i, (_pid, src) in enumerate(ABSTRACTS)]


def _files():
    def jsonl(rows):
        return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")

    return {"train": jsonl(_rows("tr-")), "dev": jsonl(_rows("dv-")), "test": jsonl(_rows("te-"))}


def _pin(monkeypatch, files):
    monkeypatch.setattr(
        sm,
        "CORPUS_FILES",
        {k: (f"{k}.jsonl", len(v), hashlib.sha256(v).hexdigest()) for k, v in files.items()},
    )
    monkeypatch.setattr(sm, "CORPUS_PAPERS", {k: 12 for k in files})
    monkeypatch.setattr(sm, "MIN_SAMPLE_TEXT_CHARS", 50)


def _encode(text: str) -> list[int]:
    """Word tokeniser over a stable hash into the vocabulary range, wrapped in [CLS] .. [SEP]."""
    ids = [
        int(hashlib.sha1(w.lower().encode()).hexdigest()[:6], 16) % (VOCAB_SIZE - 1000) + 1000
        for w in text.split()
    ]
    return [CLS_TOKEN_ID, *ids, SEP_TOKEN_ID]


def _scorer(ids: list[int], positions: list[int], targets: list[int]) -> list[tuple[float, int]]:
    """Fake MLM scorer: every masked position costs ln 2; even targets rank first, odd ones seventh."""
    assert all(ids[p] == MASK_TOKEN_ID for p in positions) and len(positions) == len(targets)
    return [(math.log(2.0), 1 if t % 2 == 0 else 7) for t in targets]


def _pipeline_without_model(scorer=_scorer):
    return BERTMaskedLMPipeline(
        lambda text: (None, 0),
        lambda texts: (None, None),
        lambda i: f"tok{i}",
        "cpu",
        "injected",
        _encode=_encode,
        _mlm_scorer=scorer,
    )


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert sm.CORPUS_BASE_URL.startswith("https://raw.githubusercontent.com/allenai/scitldr/5ccad9c0")
    assert {k: v[1] for k, v in sm.CORPUS_FILES.items()} == {
        "train": 3_155_015,
        "dev": 1_124_865,
        "test": 1_204_107,
    }
    assert all(len(v[2]) == 64 for v in sm.CORPUS_FILES.values())
    assert sum(SAMPLE_SPLIT.values()) == 450 and set(SAMPLE_SPLIT) == {"train", "validation", "test"}


def test_fetch_corpus_verifies_each_file_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    calls = []

    def fetcher(url):
        calls.append(url)
        return files[url.rsplit("/", 1)[1].removesuffix(".jsonl")]

    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files
    assert len(calls) == 3 and all(u.startswith(sm.CORPUS_BASE_URL) for u in calls)
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_read_corpus_joins_abstract_sentences_and_checks_counts(monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    corpus = read_corpus(files)
    assert len(corpus["train"]) == 12 and corpus["train"][0]["id"] == "train-tr-p01"
    assert corpus["train"][0]["text"].startswith("TR-We study graph neural networks")
    assert corpus["train"][0]["text"].endswith("Code is released.")
    assert set(corpus["dev"][0]) == {"id", "text", "paper_id"} and corpus["dev"][0]["paper_id"] == "dv-p01"
    with pytest.raises(ValueError, match="missing the test file"):
        read_corpus({"train": files["train"], "dev": files["dev"]})
    monkeypatch.setattr(sm, "CORPUS_PAPERS", {"train": 99, "dev": 12, "test": 12})
    with pytest.raises(ValueError, match="expected 99"):
        read_corpus(files)


def test_filter_and_sample_split_are_seeded_and_disjoint(monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    corpus = read_corpus(files)
    noisy = [
        *corpus["train"],
        {**corpus["train"][0], "id": "dup"},
        {**corpus["train"][1], "id": "short", "text": "Tiny."},
        {**corpus["train"][2], "id": "long", "text": "x" * (sm.MAX_SAMPLE_TEXT_CHARS + 1)},
    ]
    assert len(filter_records(noisy)) == 12
    sizes = {"train": 8, "validation": 3, "test": 4}
    splits = build_sample_dataset(corpus, seed=1, sizes=sizes)
    assert {k: len(v) for k, v in splits.items()} == sizes
    assert splits["train"][0]["id"] == "train-0000" and set(splits["test"][0]) == {"id", "text", "paper_id"}
    assert check_split_disjoint(splits) == sizes
    assert build_sample_dataset(corpus, seed=1, sizes=sizes) == splits
    assert build_sample_dataset(corpus, seed=2, sizes=sizes) != splits
    with pytest.raises(ValueError, match="only"):
        build_sample_dataset(corpus, sizes={"train": 100, "validation": 1, "test": 1})
    leaky = {"train": splits["train"], "test": [{**splits["train"][0], "id": "leak"}]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaky)


def test_fetch_sample_dataset_end_to_end_with_injected_fetcher(tmp_path, monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    splits = fetch_sample_dataset(
        cache_dir=tmp_path,
        fetcher=lambda url: files[url.rsplit("/", 1)[1].removesuffix(".jsonl")],
        sizes={"train": 8, "validation": 2, "test": 2},
    )
    assert validate_dataset(splits["train"])["n_records"] == 8


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(forbid_model_imports):
    report = validate_dataset(_records())
    assert report["n_records"] == 12 and report["unique_texts"] == 12
    assert report["text_chars"]["min"] > 50 and report["total_chars"] == sum(
        len(r["text"]) for r in _records()
    )
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    tagged = validate_dataset([{**_records()[0], "paper_id": 7}, *_records()[1:]])
    assert tagged["records"][0]["paper_id"] == "7" and "paper_id" not in tagged["records"][1]
    good = _records()
    for bad, message in (
        (good[:7], "8..20000"),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match"),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id"),
        ([{**good[0], "text": " "}, *good[1:]], "text is empty"),
        ([{**good[0], "text": 5}, *good[1:]], "text must be a string"),
        ([{**good[0], "text": "x" * 4_001}, *good[1:]], "MAX_TEXT_CHARS"),
        ([{"id": "a"}, *good[1:]], "missing 'text'"),
        (["not a mapping", *good[1:]], "must be a mapping"),
        ({"a": 1}, "must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad)


def test_split_dataset_deduplicates_and_is_seeded(forbid_model_imports):
    records = [*_records(), {**_records()[0], "id": "dup"}]
    splits = split_dataset(records, val_fraction=0.1, test_fraction=0.2, seed=3)
    assert sum(len(v) for v in splits.values()) == 12 and len(splits["test"]) == 2
    assert check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.1, test_fraction=0.2, seed=3) == splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="at least"):
        split_dataset(records, val_fraction=0.0, test_fraction=0.9)


# --- metrics and baseline -----------------------------------------------------------------------------


def test_mask_positions_are_seeded_interior_and_at_least_one(forbid_model_imports):
    positions = mask_positions(42, rate=0.15, seed=record_seed("r001", 0))
    assert positions == sorted(positions) and len(positions) == 6 and all(1 <= p <= 40 for p in positions)
    assert positions == mask_positions(42, rate=0.15, seed=record_seed("r001", 0))
    assert positions != mask_positions(42, rate=0.15, seed=record_seed("r001", 1))
    assert positions != mask_positions(42, rate=0.15, seed=record_seed("r002", 0))
    assert mask_positions(3, rate=0.15, seed=1) == [1]
    with pytest.raises(ValueError, match="at least one token"):
        mask_positions(2, rate=0.15, seed=1)
    with pytest.raises(ValueError, match="rate"):
        mask_positions(10, rate=0.0, seed=1)


def test_masked_metrics_aggregate_over_positions(forbid_model_imports):
    metrics = masked_metrics([[(math.log(2.0), 1), (math.log(2.0), 5)], [(math.log(8.0), 9)]])
    assert metrics["n_records"] == 2 and metrics["n_masked"] == 3
    assert metrics["perplexity"] == pytest.approx(math.exp((2 * math.log(2) + math.log(8)) / 3))
    assert metrics["bits_per_token"] == pytest.approx(5 / 3)
    assert metrics["top1_accuracy"] == pytest.approx(1 / 3) and metrics["top5_accuracy"] == pytest.approx(
        2 / 3
    )
    assert "masking" in metrics["definitions"]
    with pytest.raises(ValueError, match="no records"):
        masked_metrics([])
    with pytest.raises(ValueError, match="no masked positions"):
        masked_metrics([[]])
    with pytest.raises(ValueError, match="rank"):
        masked_metrics([[(0.1, 0)]])


def test_unigram_baseline_is_add_one_smoothed_with_ranks(forbid_model_imports):
    scores = mt.unigram_masked_scores([[1, 1, 2]], [[9, 1, 2]], vocab_size=4)
    # counts {1: 2, 2: 1}, total 3 + 4 = 7; ranks: unseen 9 -> 3, top token 1 -> 1, token 2 -> 2
    assert scores == [
        [
            (pytest.approx(math.log(7 / 1)), 3),
            (pytest.approx(math.log(7 / 3)), 1),
            (pytest.approx(math.log(7 / 2)), 2),
        ]
    ]
    result = unigram_baseline([[1, 1, 2]], [[9, 1, 2]], vocab_size=4)
    assert (
        result["n_masked"] == 3
        and result["top1_accuracy"] == pytest.approx(1 / 3)
        and "unigram" in result["baseline"]
    )
    with pytest.raises(ValueError, match="vocab_size"):
        mt.unigram_masked_scores([[1]], [[1]], vocab_size=0)
    pipe = _pipeline_without_model()
    baseline = pipe.unigram_baseline(_records()[:8], _records()[8:])
    assert baseline["n_records"] == 4 and 1.0 < baseline["perplexity"] <= VOCAB_SIZE
    assert baseline["n_masked"] == sum(
        len(mask_positions(len(_encode(r["text"])), seed=record_seed(r["id"], 0))) for r in _records()[8:]
    )


# --- BYOD loaders and CSV -----------------------------------------------------------------------------


def test_byod_csv_json_jsonl_txt_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records()
    csv_path = write_dataset_csv(records, tmp_path / "data.csv")
    assert load_byod_dataset(csv_path) == records
    (tmp_path / "data.json").write_text(json.dumps(records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.json") == records
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.jsonl") == records
    (tmp_path / "data.txt").write_text("\r\n\r\n".join(r["text"] for r in records) + "\n", encoding="utf-8")
    txt = load_byod_dataset(tmp_path / "data.txt")
    assert [r["text"] for r in txt] == [r["text"] for r in records] and txt[0]["id"] == "doc-00000"
    (tmp_path / "bad.csv").write_text("id,source\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(tmp_path / "bad.csv")
    (tmp_path / "obj.json").write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="array of records"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "data.csv.bak").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="csv, .json, .jsonl or .txt"):
        load_byod_dataset(tmp_path / "data.csv.bak")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.csv")


# --- evaluation, adaptation and artifacts without a model ---------------------------------------------


def test_evaluate_uses_the_injected_scorer_and_refuses_oversized_records(forbid_model_imports):
    pipe = _pipeline_without_model()
    metrics = pipe.evaluate(_records())
    n_masked = sum(
        len(mask_positions(len(_encode(r["text"])), seed=record_seed(r["id"], 0))) for r in _records()
    )
    assert metrics["n_records"] == 12 and metrics["n_masked"] == n_masked
    assert metrics["perplexity"] == pytest.approx(2.0) and metrics["bits_per_token"] == pytest.approx(1.0)
    assert 0.0 < metrics["top1_accuracy"] < 1.0 and metrics["top1_accuracy"] == metrics["top5_accuracy"]
    assert metrics["verdict"] == "measured-small-sample" and metrics["adapted"] is False
    assert metrics["mask_rate"] == 0.15 and metrics["seed"] == 0 and metrics["model_id"] == MODEL_ID
    assert pipe.evaluate(_records(), seed=3)["n_masked"] == n_masked  # the count is seed-independent
    big = {"id": "big", "text": "w " * (MAX_TEXT_TOKENS - 1)}
    with pytest.raises(ValueError, match="ceiling is 512"):
        pipe.evaluate([big])
    with pytest.raises(ValueError, match="from_pretrained"):
        _pipeline_without_model(scorer=None).evaluate(_records())


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(_records(), batch_size=0)
    with pytest.raises(ValueError, match="mask_rate"):
        pipe.adapt(_records(), mask_rate=0.9)
    with pytest.raises(ValueError, match="trainable_layers"):
        pipe.adapt(_records(), trainable_layers=ENCODER_LAYERS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(_records())
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["bert.encoder.layer.11.output.dense.weight"],
        "adapter": {"trainable_layers": 1},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)


def test_load_artifact_refuses_unsupported_versions_extra_files_and_traversal(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    good = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": [],
        "adapter": {"trainable_layers": 1},
    }

    def write(manifest):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))

    write({**good, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": good["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "other.safetensors"}]})
    with pytest.raises(ValueError, match="must name exactly"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**good, "base_model": {**good["base_model"], "weight_file": "other.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {}})
    with pytest.raises(ValueError, match="trainable_layers"):
        pipe.load_artifact(tmp_path)
    write(good)  # every manifest check passes; the weights file is still missing, and no model was imported
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
