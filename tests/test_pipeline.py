import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from bert_masked_lm_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    HIDDEN_SIZE,
    MAX_BATCH,
    MAX_TEXT_CHARS,
    MAX_TOP_K,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    VOCAB_SIZE,
    BERTMaskedLMPipeline,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def _fake_mask_runner(text: str) -> tuple[np.ndarray, int]:
    logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
    logits[3000] = 6.0  # stand-in for the winning token
    logits[2000] = 3.0
    return logits, len(text.split()) + 2


def _fake_embed_runner(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    n, t = len(texts), 5
    hidden = np.zeros((n, t, HIDDEN_SIZE), dtype=np.float32)
    hidden[:, 0, 0] = 1.0  # CLS position points along axis 0
    hidden[:, 1:, 1] = 2.0  # other tokens point along axis 1
    mask = np.ones((n, t), dtype=np.int64)
    mask[:, -1] = 0  # last position is padding
    return hidden, mask


def _pipeline() -> BERTMaskedLMPipeline:
    return BERTMaskedLMPipeline(_fake_mask_runner, _fake_embed_runner, lambda i: f"tok{i}", "cpu", "injected")


def _write_snapshot(root: Path, payload: bytes = b"weights") -> Path:
    (root / "model.safetensors").write_bytes(payload)
    manifest = {
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "model.safetensors",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identity_constants_are_40_hex_and_named():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "google-bert/bert-base-uncased"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY


def test_identity_matches_local_manifest_when_present():
    manifest_path = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local snapshot manifest not staged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["modelKey"] == MODEL_KEY


def test_verify_snapshot_accepts_matching_manifest(tmp_path: Path):
    result = verify_snapshot(_write_snapshot(tmp_path).parent)
    assert result["revision"] == MODEL_REVISION and result["path"] == str(tmp_path)


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["files"][0]["sha256"]
    manifest["files"][0]["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_size_missing_file_and_identity(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        BERTMaskedLMPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    assert len(verify_snapshot(tmp_path)["files"]) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_fill_mask_rejects_bad_inputs():
    pipe = _pipeline()
    with pytest.raises(TypeError):
        pipe.fill_mask(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        pipe.fill_mask("   ")
    with pytest.raises(ValueError, match="exactly one"):
        pipe.fill_mask("no mask here")
    with pytest.raises(ValueError, match="exactly one"):
        pipe.fill_mask("[MASK] twice [MASK]")
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        pipe.fill_mask("a" * (MAX_TEXT_CHARS + 1) + " [MASK]")
    with pytest.raises(TypeError):
        pipe.fill_mask("x [MASK]", top_k=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_k"):
        pipe.fill_mask("x [MASK]", top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        pipe.fill_mask("x [MASK]", top_k=MAX_TOP_K + 1)


def test_fill_mask_output_fields():
    result = _pipeline().fill_mask("The capital of France is [MASK].", top_k=3)
    assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
    assert result["top_k"] == 3 and len(result["candidates"]) == 3
    best = result["candidates"][0]
    assert set(best) == {"token", "token_id", "score", "sequence"}
    assert best["token_id"] == 3000 and best["token"] == "tok3000"
    assert best["sequence"] == "The capital of France is tok3000."
    assert result["candidates"][1]["token_id"] == 2000
    assert best["score"] > result["candidates"][1]["score"] > 0.0
    assert sum(c["score"] for c in result["candidates"]) <= 1.0 + 1e-9


def test_embed_rejects_bad_inputs():
    pipe = _pipeline()
    with pytest.raises(TypeError):
        pipe.embed("a single string")
    with pytest.raises(ValueError, match="MAX_BATCH"):
        pipe.embed([])
    with pytest.raises(ValueError, match="MAX_BATCH"):
        pipe.embed(["x"] * (MAX_BATCH + 1))
    with pytest.raises(TypeError):
        pipe.embed(["ok", 3])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="empty"):
        pipe.embed(["ok", " "])
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        pipe.embed(["a" * (MAX_TEXT_CHARS + 1)])
    with pytest.raises(ValueError, match="pooling"):
        pipe.embed(["ok"], pooling="max")


def test_embed_output_fields_and_pooling():
    pipe = _pipeline()
    cls = pipe.embed(["one", "two"], pooling="cls")
    assert cls["dim"] == HIDDEN_SIZE and cls["pooling"] == "cls" and cls["normalized"] is True
    assert cls["model_id"] == MODEL_ID and cls["model_revision"] == MODEL_REVISION
    vectors = np.asarray(cls["embeddings"])
    assert vectors.shape == (2, HIDDEN_SIZE)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert vectors[0, 0] == pytest.approx(1.0)  # CLS pooling picks position 0
    assert cls["n_tokens"] == [4, 4]
    mean = np.asarray(pipe.embed(["one"], pooling="mean")["embeddings"])
    assert mean.shape == (1, HIDDEN_SIZE)
    assert mean[0, 1] > mean[0, 0] > 0.0  # masked mean mixes CLS with the three real tokens, not the pad
