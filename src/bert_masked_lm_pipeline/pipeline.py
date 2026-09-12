"""Masked-language modelling and sentence embeddings over the pinned ``google-bert/bert-base-uncased``.

Weights load only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly allowed,
from the Hugging Face Hub at the pinned revision. Two task methods: ``fill_mask`` (one ``[MASK]`` token ->
ranked vocabulary candidates) and ``embed`` (CLS or mean pooled, L2-normalised 768-d representations).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "google-bert/bert-base-uncased"
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "bert-base-uncased"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Ceilings. 512 is max_position_embeddings in the pinned config.json and model_max_length in
# tokenizer_config.json; longer inputs are rejected (not truncated) so a caller never silently loses [MASK].
MAX_TEXT_TOKENS = 512
MAX_TEXT_CHARS = 4_000  # pre-tokenisation guard; ~4 chars per WordPiece token on English text
MAX_BATCH = 64  # texts per embed() call
MAX_TOP_K = 100
VOCAB_SIZE = 30522  # config.json vocab_size
HIDDEN_SIZE = 768  # config.json hidden_size
MASK_TOKEN = "[MASK]"
POOLINGS = ("cls", "mean")
DEFAULT_TOP_K = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _check_text(text: Any, name: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"{name} must be str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError(f"{name} is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"{name} has {len(text)} chars; ceiling is MAX_TEXT_CHARS={MAX_TEXT_CHARS}")
    return text


@dataclass
class BERTMaskedLMPipeline:
    """``_mask_runner`` maps one text to (vocab logits at the [MASK] position, n_tokens);
    ``_embed_runner`` maps texts to (last hidden states (N, T, 768), attention mask (N, T)); both injectable.
    ``_decode`` maps a token id to its string."""

    _mask_runner: Callable[[str], tuple[np.ndarray, int]]
    _embed_runner: Callable[[list[str]], tuple[np.ndarray, np.ndarray]]
    _decode: Callable[[int], str]
    device: str = "cpu"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> BERTMaskedLMPipeline:
        import torch
        from transformers import AutoTokenizer, BertForMaskedLM

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
            origin = "local-snapshot"
        elif allow_download:
            source, kwargs, origin = MODEL_ID, {}, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = BertForMaskedLM.from_pretrained(
            source, revision=MODEL_REVISION, dtype=torch.float32, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()
        mask_id = tokenizer.mask_token_id

        def mask_runner(text: str) -> tuple[np.ndarray, int]:
            batch = tokenizer(text, return_tensors="pt", truncation=False).to(resolved_device)
            n_tokens = int(batch["input_ids"].shape[1])
            if n_tokens > MAX_TEXT_TOKENS:
                raise ValueError(f"text tokenises to {n_tokens} tokens; MAX_TEXT_TOKENS={MAX_TEXT_TOKENS}")
            position = (batch["input_ids"][0] == mask_id).nonzero().flatten()
            with torch.inference_mode():
                logits = model(**batch).logits[0, position[0]]
            return logits.float().cpu().numpy(), n_tokens

        def embed_runner(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
            if batch["input_ids"].shape[1] > MAX_TEXT_TOKENS:
                raise ValueError(f"a text tokenises past MAX_TEXT_TOKENS={MAX_TEXT_TOKENS}")
            batch = batch.to(resolved_device)
            with torch.inference_mode():
                hidden = model.bert(**batch).last_hidden_state
            return hidden.float().cpu().numpy(), batch["attention_mask"].cpu().numpy()

        return cls(mask_runner, embed_runner, tokenizer.convert_ids_to_tokens, resolved_device, origin)

    def fill_mask(self, text: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """Rank candidates for exactly one ``[MASK]``; ``score`` is a softmax over the 30 522-token vocab."""
        text = _check_text(text, "text")
        if text.count(MASK_TOKEN) != 1:
            raise ValueError(f"text must contain exactly one {MASK_TOKEN}, found {text.count(MASK_TOKEN)}")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an int")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and MAX_TOP_K={MAX_TOP_K}")
        logits, n_tokens = self._mask_runner(text)
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != (VOCAB_SIZE,):
            raise RuntimeError(f"backend returned {logits.shape}, expected ({VOCAB_SIZE},)")
        shifted = np.exp(logits - logits.max())
        probs = shifted / shifted.sum()
        order = np.argsort(-probs, kind="stable")[:top_k]
        candidates = [
            {
                "token": self._decode(int(i)),
                "token_id": int(i),
                "score": float(probs[i]),
                "sequence": text.replace(MASK_TOKEN, self._decode(int(i)), 1),
            }
            for i in order
        ]
        return {
            "candidates": candidates,
            "top_k": top_k,
            "n_tokens": n_tokens,
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def embed(self, texts: Sequence[str], pooling: str = "cls") -> dict[str, Any]:
        """L2-normalised 768-d representations: CLS token or attention-masked mean of the last layer."""
        if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a list of str, not a single string")
        if not 1 <= len(texts) <= MAX_BATCH:
            raise ValueError(f"texts must hold 1..MAX_BATCH={MAX_BATCH} items, got {len(texts)}")
        clean = [_check_text(t, f"texts[{i}]") for i, t in enumerate(texts)]
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        hidden, mask = self._embed_runner(clean)
        hidden = np.asarray(hidden, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if hidden.ndim != 3 or hidden.shape[0] != len(clean) or hidden.shape[2] != HIDDEN_SIZE:
            raise RuntimeError(f"backend returned {hidden.shape}, expected ({len(clean)}, T, {HIDDEN_SIZE})")
        if pooling == "cls":
            pooled = hidden[:, 0]
        else:
            counts = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
            pooled = (hidden * mask[:, :, None]).sum(axis=1) / counts
        normalized = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return {
            "embeddings": normalized.tolist(),
            "dim": HIDDEN_SIZE,
            "pooling": pooling,
            "normalized": True,
            "n_tokens": [int(v) for v in mask.sum(axis=1)],
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
