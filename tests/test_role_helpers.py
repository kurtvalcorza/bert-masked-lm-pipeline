"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from bert_masked_lm_pipeline import (
    DEFAULT_TOP_K,
    HIDDEN_SIZE,
    INPUT_SCHEMA,
    MASK_TOKEN,
    MAX_BATCH,
    MAX_TEXT_CHARS,
    MAX_TEXT_TOKENS,
    MAX_TOP_K,
    MODEL_ID,
    MODEL_REVISION,
    POOLINGS,
    VOCAB_SIZE,
    evaluation_report,
    validate_inputs,
)

CLOZE = f"The capital of France is {MASK_TOKEN}."


def _filled(tokens: list[str]) -> dict:
    return {
        "candidates": [
            {"token": token, "token_id": 1000 + i, "score": 0.5 / (i + 1), "sequence": CLOZE}
            for i, token in enumerate(tokens)
        ],
        "top_k": len(tokens),
        "n_tokens": 9,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(
        [CLOZE, "The cat sat on the mat."], top_k=3, pooling="mean", names=["c1", "s1"]
    )
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["batch"] == [1, MAX_BATCH]
    assert manifest["schema"]["text_chars"] == [1, MAX_TEXT_CHARS]
    assert manifest["schema"]["text_tokens"] == [1, MAX_TEXT_TOKENS]
    assert manifest["schema"]["top_k"] == [1, MAX_TOP_K]
    assert manifest["schema"]["pooling"] == list(POOLINGS)
    assert manifest["schema"]["vocab_size"] == VOCAB_SIZE
    assert manifest["schema"]["embedding_dim"] == HIDDEN_SIZE
    assert manifest["inputs"] == [
        {"id": "c1", "chars": len(CLOZE), "masks": 1, "fill_mask_input": True},
        {"id": "s1", "chars": 23, "masks": 0, "fill_mask_input": False},
    ]
    assert (manifest["top_k"], manifest["pooling"]) == (3, "mean")
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids_and_top_k() -> None:
    manifest = validate_inputs(["A sentence."])
    assert [entry["id"] for entry in manifest["inputs"]] == ["text-0"]
    assert manifest["top_k"] == DEFAULT_TOP_K
    assert manifest["pooling"] == "cls"


def test_validate_inputs_rejects_like_the_core_methods() -> None:
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs("a bare string")
    with pytest.raises(ValueError, match=f"MAX_BATCH={MAX_BATCH}"):
        validate_inputs(["x"] * (MAX_BATCH + 1))
    with pytest.raises(ValueError, match="is empty"):
        validate_inputs(["  "])
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        validate_inputs(["x" * (MAX_TEXT_CHARS + 1)])
    with pytest.raises(ValueError, match="pooling must be one of"):
        validate_inputs(["a sentence"], pooling="max")
    with pytest.raises(TypeError, match="top_k must be an int"):
        validate_inputs(["a sentence"], top_k=True)
    with pytest.raises(ValueError, match=f"MAX_TOP_K={MAX_TOP_K}"):
        validate_inputs(["a sentence"], top_k=MAX_TOP_K + 1)
    with pytest.raises(ValueError, match="exactly one"):
        validate_inputs([f"{MASK_TOKEN} and {MASK_TOKEN}"])
    with pytest.raises(ValueError, match="names must have one entry per text"):
        validate_inputs(["a sentence"], names=["a", "b"])


def test_evaluation_report_is_always_not_measurable() -> None:
    report = evaluation_report(_filled(["paris", "lyon", "nice"]))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_candidates"] == 3
    assert report["sample_kind"] == "synthetic"
    assert "no gold tokens and no similarity labels" in report["reason"]
    assert "labelled cloze set" in report["needs"]
    assert "Spearman correlation" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_stays_not_measurable_when_expected_tokens_are_supplied() -> None:
    report = evaluation_report(_filled(["paris"]), ["paris"], sample_kind="BYOD upload")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["sample_kind"] == "BYOD upload"
    assert "an expected token was supplied" in report["reason"]
    assert "single observation as an accuracy" in report["reason"]
