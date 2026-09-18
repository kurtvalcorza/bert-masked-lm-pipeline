"""Masked-language modelling and sentence embeddings over the pinned ``google-bert/bert-base-uncased``.

Weights load only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly allowed,
from the Hugging Face Hub at the pinned revision. Two task methods: ``fill_mask`` (one ``[MASK]`` token ->
ranked vocabulary candidates) and ``embed`` (CLS or mean pooled, L2-normalised 768-d representations).

The adaptation contract (``evaluate``, ``unigram_baseline``, ``adapt``, ``save_artifact``, ``from_artifact``)
scores a validated ``{id, text}`` corpus by masked-token prediction at seeded positions, continues the
masked-language-model objective on it for the last encoder layers with validation-perplexity epoch selection,
and exports the trained tensors as a safetensors adapter bound to the pinned base weights. The two inference
methods are unchanged by it, but both read the adapted encoder once ``adapt`` or ``load_artifact`` has run.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
PAD_TOKEN_ID = 0  # vocab.txt [PAD]
CLS_TOKEN_ID = 101  # vocab.txt [CLS]
SEP_TOKEN_ID = 102  # vocab.txt [SEP]
MASK_TOKEN_ID = 103  # vocab.txt [MASK]
WEIGHT_FILE = "model.safetensors"
WEIGHT_SHA256 = (
    "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"  # manifest digest of WEIGHT_FILE
)
PARAMETER_COUNT = 109_514_298
ENCODER_LAYERS = 12  # config.json num_hidden_layers
DEFAULT_TRAINABLE_LAYERS = 4  # the last four encoder layers (28,351,488 parameters)
MAX_EVAL_RECORDS = 2_000
MAX_RECORDS_FIT = 20_000  # the unigram baseline may be fitted on a whole training split
MIN_SCORED_RECORDS = 50  # below this a scored corpus is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.bert-base-uncased.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"


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


def _check_mask_count(text: str) -> str:
    """`fill_mask` accepts exactly one [MASK]; raise naming the count found."""
    if text.count(MASK_TOKEN) != 1:
        raise ValueError(f"text must contain exactly one {MASK_TOKEN}, found {text.count(MASK_TOKEN)}")
    return text


def _check_top_k(top_k: Any) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and MAX_TOP_K={MAX_TOP_K}")
    return top_k


def _check_batch(texts: Any, pooling: Any) -> list[str]:
    """`embed`'s batch contract; raise naming the first violated ceiling."""
    if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a list of str, not a single string")
    if not 1 <= len(texts) <= MAX_BATCH:
        raise ValueError(f"texts must hold 1..MAX_BATCH={MAX_BATCH} items, got {len(texts)}")
    clean = [_check_text(t, f"texts[{i}]") for i, t in enumerate(texts)]
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}")
    return clean


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "sequence of non-empty str; every entry carrying a [MASK] token is also a fill_mask input, "
        "every entry is an embed input (one vector per text)"
    ),
    "batch": [1, MAX_BATCH],
    "text_chars": [1, MAX_TEXT_CHARS],
    "text_tokens": [1, MAX_TEXT_TOKENS],
    "top_k": [1, MAX_TOP_K],
    "pooling": list(POOLINGS),
    "mask_token": MASK_TOKEN,
    "masks_per_fill_mask_text": 1,
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": HIDDEN_SIZE,
    "preprocessing": (
        "WordPiece tokenisation that lower-cases and strips accents; texts past MAX_TEXT_TOKENS are "
        "rejected, never truncated, so a [MASK] can never be silently lost; embed pools the last "
        "layer (cls position or attention-masked mean) and L2-normalises"
    ),
}


def validate_inputs(
    texts: Sequence[str],
    *,
    top_k: int = DEFAULT_TOP_K,
    pooling: str = "cls",
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    ``texts`` is the batch ``embed`` would take; every entry that carries a ``[MASK]`` token is
    additionally checked against ``fill_mask``'s contract (exactly one mask, ``top_k`` in range) and
    marked in the manifest. Both capabilities' checks run through the same private functions the
    methods use — ``_check_batch``/``_check_text`` for ``embed``, ``_check_mask_count``/``_check_top_k``
    for ``fill_mask`` — so a rejection here is a rejection there. ``MAX_TEXT_TOKENS`` is enforced
    after tokenisation inside the pipeline and therefore cannot be observed at this stage.
    """
    checked = _check_batch(texts, pooling)
    _check_top_k(top_k)
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per text")
    inputs = []
    for i, text in enumerate(checked):
        masks = text.count(MASK_TOKEN)
        if masks:
            _check_mask_count(text)
        inputs.append(
            {
                "id": names[i] if names else f"text-{i}",
                "chars": len(text),
                "masks": masks,
                "fill_mask_input": bool(masks),
            }
        )
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": inputs,
        "top_k": top_k,
        "pooling": pooling,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any], expected_tokens: Sequence[str] | None = None, *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even though no metric exists here.

    Neither capability has a metric helper in this repository, so the verdict is always
    ``not-measurable`` (EVAL9). ``expected_tokens`` exists for interface parity with the fleet's
    other pipelines and is recorded in ``reason`` rather than scored: one author-expected token is
    an intent, not a labelled cloze set, and computing a hit rate from it would present a single
    observation as an accuracy. ``result`` is the ``fill_mask`` result; the embedding half is a
    representation and is covered by the same verdict.
    """
    candidates = result.get("candidates", [])
    supplied = expected_tokens is not None
    return {
        "task": "masked-language modelling (fill-mask) and sentence embedding",
        "score_semantics": (
            f"fill_mask `score` is a softmax over the {VOCAB_SIZE}-token vocabulary at the masked "
            "position — a ranking signal, not a calibrated probability, with argmax as the decision "
            f"rule and no shipped threshold; embed returns {HIDDEN_SIZE}-d unit vectors whose only "
            "meaning is cosine within the same model and pooling policy"
        ),
        "sample_kind": sample_kind,
        "n_candidates": len(candidates),
        "metrics": [],
        "baselines": [],
        "verdict": "not-measurable",
        "reason": (
            "the repository ships no metric helper for either capability"
            + (
                "; an expected token was supplied, but one author-expected token is an intent rather "
                "than a labelled cloze set, so scoring it would present a single observation as an accuracy"
                if supplied
                else "; the evaluated sample carries no gold tokens and no similarity labels"
            )
        ),
        "needs": (
            "for fill-mask, a labelled cloze set (sentence, mask position, gold token) over enough "
            "sentences to state a dispersion, scored with the caller's own top-1/top-k hit-rate code; "
            "for the embeddings, a judged similarity set (Spearman correlation) or a retrieval or "
            "clustering set with relevance labels (recall@k) — neither of which this repository ships"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


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
    _encode: Callable[[str], list[int]] | None = field(default=None, repr=False)
    _mlm_scorer: Callable[[list[int], list[int], list[int]], list[tuple[float, int]]] | None = field(
        default=None, repr=False
    )
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> BERTMaskedLMPipeline:
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
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import AutoTokenizer, BertForMaskedLM

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
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

        def encode(text: str) -> list[int]:
            return [int(i) for i in tokenizer(text, truncation=False)["input_ids"]]

        def mlm_scorer(ids: list[int], positions: list[int], targets: list[int]) -> list[tuple[float, int]]:
            """(NLL of the original token, its rank) at every masked position of one already-masked record."""
            input_ids = torch.tensor([ids], dtype=torch.long, device=resolved_device)
            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits[0]
            rows = logits[positions].float()
            log_probs = torch.log_softmax(rows, dim=-1)
            target = torch.tensor(targets, dtype=torch.long, device=resolved_device)
            nll = -log_probs.gather(1, target[:, None])[:, 0]
            rank = (rows > rows.gather(1, target[:, None])).sum(dim=1) + 1
            return list(zip(nll.tolist(), rank.tolist(), strict=True))

        return cls(
            mask_runner,
            embed_runner,
            tokenizer.convert_ids_to_tokens,
            resolved_device,
            origin,
            _encode=encode,
            _mlm_scorer=mlm_scorer,
            _model=model,
            _tokenizer=tokenizer,
        )

    def fill_mask(self, text: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """Rank candidates for exactly one ``[MASK]``; ``score`` is a softmax over the 30 522-token vocab."""
        text = _check_mask_count(_check_text(text, "text"))
        top_k = _check_top_k(top_k)
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
        clean = _check_batch(texts, pooling)
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

    # ---- adaptation -----------------------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._tokenizer

    def _record_ids(self, records: Sequence[Mapping[str, Any]]) -> list[list[int]]:
        """Tokenise validated records with [CLS]/[SEP]; a record is refused (never truncated) above
        MAX_TEXT_TOKENS or without at least one interior token."""
        if self._encode is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        out = []
        for record in records:
            ids = list(self._encode(record["text"]))
            if len(ids) > MAX_TEXT_TOKENS:
                raise ValueError(f"record {record['id']} has {len(ids)} tokens; ceiling is {MAX_TEXT_TOKENS}")
            if len(ids) < 3:
                raise ValueError(f"record {record['id']} has no token between [CLS] and [SEP]")
            out.append(ids)
        return out

    @staticmethod
    def _masked(
        records: Sequence[Mapping[str, Any]], ids: Sequence[Sequence[int]], *, rate: float, seed: int
    ) -> list[tuple[list[int], list[int], list[int]]]:
        """(masked ids, positions, original tokens) per record under the seeded per-record masking."""
        from .metrics import mask_positions, record_seed

        out = []
        for record, record_ids in zip(records, ids, strict=True):
            positions = mask_positions(len(record_ids), rate=rate, seed=record_seed(record["id"], seed))
            masked = list(record_ids)
            for position in positions:
                masked[position] = MASK_TOKEN_ID
            out.append((masked, positions, [record_ids[p] for p in positions]))
        return out

    def evaluate(
        self, records: Sequence[Mapping[str, Any]], *, mask_rate: float = 0.15, seed: int = 0
    ) -> dict[str, Any]:
        """Masked-token prediction on a validated corpus: a seeded `mask_rate` of each record's interior
        tokens is replaced by [MASK] and the original tokens are scored by NLL and rank."""
        from .metrics import masked_metrics
        from .samples import validate_dataset

        if self._mlm_scorer is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        scores = [
            self._mlm_scorer(masked, positions, targets)
            for masked, positions, targets in self._masked(
                checked, self._record_ids(checked), rate=mask_rate, seed=seed
            )
        ]
        metrics = masked_metrics(scores)
        metrics.update(
            {
                "mask_rate": mask_rate,
                "seed": seed,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def unigram_baseline(
        self,
        train: Sequence[Mapping[str, Any]],
        test: Sequence[Mapping[str, Any]],
        *,
        mask_rate: float = 0.15,
        seed: int = 0,
    ) -> dict[str, Any]:
        """The add-one unigram model fitted on `train` (interior tokens), scored at the same masked positions
        of `test` that `evaluate` uses under the same seed."""
        from .metrics import unigram_baseline
        from .samples import validate_dataset

        train_checked = validate_dataset(train, min_records=1, max_records=MAX_RECORDS_FIT)["records"]
        test_checked = validate_dataset(test, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        train_ids = [ids[1:-1] for ids in self._record_ids(train_checked)]
        targets = [
            record_targets
            for _masked, _positions, record_targets in self._masked(
                test_checked, self._record_ids(test_checked), rate=mask_rate, seed=seed
            )
        ]
        result = unigram_baseline(train_ids, targets, VOCAB_SIZE)
        result.update({"mask_rate": mask_rate, "seed": seed})
        return result

    def _trainable_names(self, trainable_layers: int) -> list[str]:
        if not isinstance(trainable_layers, int) or not 1 <= trainable_layers <= ENCODER_LAYERS:
            raise ValueError(f"trainable_layers must be an int in 1..{ENCODER_LAYERS}")
        model, _ = self._require_model()
        first = ENCODER_LAYERS - trainable_layers
        prefixes = tuple(f"bert.encoder.layer.{k}." for k in range(first, ENCODER_LAYERS))
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 2,
        lr: float = 5e-5,
        batch_size: int = 8,
        trainable_layers: int = DEFAULT_TRAINABLE_LAYERS,
        mask_rate: float = 0.15,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded continued masked-language-model training on a validated text corpus.

        Only the last `trainable_layers` encoder layers train (4 by default; the word, position and
        token-type embeddings, the earlier layers, the pooler and the MLM head with its tied decoder stay
        frozen). Every epoch re-draws a seeded `mask_rate` of each record's interior tokens, replaces them
        with [MASK] and applies cross-entropy at those positions only; AdamW at a fixed learning rate,
        gradient clipping at 1.0, no scheduler; records over MAX_TEXT_TOKENS are refused, never truncated.
        Epoch 0 records the frozen model's validation metrics under the evaluation masking (`seed`); the
        epoch with the lowest validation masked perplexity is kept."""
        from .samples import validate_dataset

        if not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        if not (0.0 < mask_rate <= 0.5):
            raise ValueError("mask_rate must be in (0, 0.5]")
        names = self._trainable_names(trainable_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )
        train_ids = self._record_ids(train_checked)
        import torch

        torch.manual_seed(seed)
        model, _ = self._require_model()
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        device = torch.device(self.device)

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            keep = ("perplexity", "bits_per_token", "top1_accuracy", "top5_accuracy", "n_masked")
            return {
                k: v
                for k, v in self.evaluate(val_checked, mask_rate=mask_rate, seed=seed).items()
                if k in keep
            }

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_ppl = entry["val"]["perplexity"] if entry["val"] else math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(1, epochs + 1):
            model.train()
            masked = self._masked(train_checked, train_ids, rate=mask_rate, seed=seed + 1_000_003 * epoch)
            order = torch.randperm(len(masked), generator=generator).tolist()
            losses = []
            for start in range(0, len(order), batch_size):
                batch = [masked[i] for i in order[start : start + batch_size]]
                width = max(len(ids) for ids, _p, _t in batch)
                input_ids = torch.full((len(batch), width), PAD_TOKEN_ID, dtype=torch.long)
                attention = torch.zeros((len(batch), width), dtype=torch.long)
                labels = torch.full((len(batch), width), -100, dtype=torch.long)
                for row, (ids, positions, targets) in enumerate(batch):
                    input_ids[row, : len(ids)] = torch.tensor(ids)
                    attention[row, : len(ids)] = 1
                    labels[row, positions] = torch.tensor(targets)
                out = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention.to(device),
                    labels=labels.to(device),
                )
                optimiser.zero_grad(set_to_none=True)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimiser.step()
                losses.append(float(out.loss.detach()))
            model.eval()
            entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
            history.append(entry)
            if progress:
                progress(entry)
            current = entry["val"]["perplexity"] if entry["val"] else -math.inf
            if current < best_ppl or not entry["val"]:
                best_ppl = current
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                best_epoch = epoch
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "objective": "masked language modelling (continued pre-training)",
            "trainable_layers": trainable_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "lowest validation masked perplexity"
            if val_checked
            else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "mask_rate": mask_rate,
            "n_train": len(train_checked),
            "n_train_tokens": sum(len(ids) - 2 for ids in train_ids),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted encoder-layer tensors as safetensors with a manifest naming the base."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest and digest, then overwrite exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        entry = manifest["files"][0]
        weights_path = root / entry["path"]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != manifest["tensors"]:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith("bert.encoder.layer."):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable encoder-layer tensor of the base"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key}: shape {tuple(value.shape)} != {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> BERTMaskedLMPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline
