# ruff: noqa: E501  -- assertion messages quote notebook text on one line
"""Regression tests for the Notebook Review Framework v1 findings on tutorials/bert_masked_lm_colab.ipynb
(review PR #8: MLM-M1..M4, MLM-m1..m5).

Everything here runs in CI's lightweight install (pytest, ruff, numpy, the package without its model
dependencies): the notebook's own cells are executed from the committed JSON with injected stand-ins (a
word tokenizer, a fake pipeline factory, fake metric dicts), never with torch or the real weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

import bert_masked_lm_pipeline.metrics as metrics_module
import bert_masked_lm_pipeline.pipeline as pipeline_module
import bert_masked_lm_pipeline.samples as samples_module
from bert_masked_lm_pipeline import (
    CLS_TOKEN_ID,
    MAX_TEXT_TOKENS,
    SEP_TOKEN_ID,
    BERTMaskedLMPipeline,
    byod_minimum_records,
    check_byod_tokens,
    split_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "bert_masked_lm_colab.ipynb"
sys.path.insert(0, str(ROOT / "tools"))
from notebook_template import TEMPLATE  # noqa: E402


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def _markdown(notebook: dict) -> str:
    return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _cell(notebook: dict, marker: str) -> str:
    found = [src for src in _code_cells(notebook) if marker in src]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _encode(text: str) -> list[int]:
    return [CLS_TOKEN_ID, *(1000 + (len(w) % 500) for w in text.split()), SEP_TOKEN_ID]


def _tokenizer_only_pipe() -> BERTMaskedLMPipeline:
    return BERTMaskedLMPipeline(
        lambda text: (None, 0),
        lambda texts: (None, None),
        lambda i: f"tok{i}",
        "cpu",
        "injected",
        _encode=_encode,
    )


def _carried_namespace() -> dict:
    """The kernel globals the carried module cells define (the notebook embeds the three modules verbatim)."""
    ns: dict = {}
    for module in (metrics_module, pipeline_module, samples_module):
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    ns.update(os=os, Path=Path, json=json)
    return ns


def _doc(i: int, opening: str = "Graph models predict molecule properties") -> dict:
    return {"id": f"d{i:03d}", "text": f"{opening} in study number {i}. We report results on benchmark {i}."}


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _run_section4(notebook: dict, tmp_path: Path, monkeypatch, *, byod_path: str = "", use_byod: bool = True) -> dict:
    source = _cell(notebook, "USE_BYOD = False  # @param")
    source = source.replace("USE_BYOD = False  # @param", f"USE_BYOD = {use_byod}  # @param", 1)
    source = source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {byod_path!r}  # @param", 1)
    monkeypatch.chdir(tmp_path)
    ns = _carried_namespace()
    ns["pipe"] = _tokenizer_only_pipe()
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


# --- MLM-M1: isolated uv environment, no in-kernel install, no restart ----------------------------------


def test_isolated_runtime_replaces_the_in_kernel_install(notebook):
    code = _code_cells(notebook)
    kernel = [i for i, src in enumerate(code) if "# dimer: kernel cell" in src]
    assert kernel == [0, 1], f"only the first two code cells may run in the kernel, got {kernel}"
    install, router = code[0], code[1]
    assert '"venv", "--quiet", "--managed-python", "--python", MANAGED_PYTHON' in install
    assert f"MANAGED_PYTHON = {TEMPLATE['managed_python']!r}" in install
    assert '"--require-hashes", "--only-binary", ":all:"' in install
    assert 'platform.machine() != "x86_64"' in install
    assert f"UV_SHA256 = {TEMPLATE['uv']['sha256']!r}" in install and f"UV_BYTES = {TEMPLATE['uv']['bytes']}" in install
    assert "sys.executable" not in install.split("SKIP_INSTALL = ", 1)[1]  # no pip into the kernel's Python
    assert '"pip", "install", "--quiet", "--python", str(ISOLATED_PYTHON), "--require-hashes"' in install
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in router
    assert 'DIMER_NOTEBOOK_CI_PREINSTALLED="1"' in router


def test_carried_lock_equals_the_repository_lock_and_pins_every_dependency(notebook):
    install = _code_cells(notebook)[0]
    lock_text = (ROOT / TEMPLATE["lock"]).read_text(encoding="utf-8")
    carried = install.split("LOCK_TEXT = r'''", 1)[1].split("'''", 1)[0]
    assert carried == lock_text
    digest = re.search(r"LOCK_SHA256 = '([0-9a-f]{64})'", install).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    import build_notebook

    build_notebook.check_lock(build_notebook._pins(ROOT, TEMPLATE), lock_text)  # raises on a missing pin or hash


def test_no_learner_text_asks_for_a_restart_and_status_is_candidate(notebook):
    markdown = _markdown(notebook)
    assert "Restart the runtime, then rerun" not in markdown and "installs the pinned dependencies" not in markdown
    assert "no restart is needed" in markdown and "Linux x86_64" in markdown
    for name in ("STATUS.md", "README.md", "tutorials/README.md", "docs/release-verification.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "1 restart after install cell)" not in text, name
        assert "**Candidate" in text, name
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert record.count("not a one-pass `Run all`, not promotion evidence (review MLM-M1)") == 2
    assert "**PASSED** — 11/11 code cells ok (1 restart" not in record


# --- MLM-M2: every pass starts from the pretrained model ------------------------------------------------


def _reset_function(notebook: dict, ns: dict):
    source = _cell(notebook, "def reset_to_pretrained():")
    block = source.split("def reset_to_pretrained():", 1)[1].split("\nreset_to_pretrained()\n", 1)[0]
    exec(compile("def reset_to_pretrained():" + block, "<section 5 reset>", "exec"), ns)
    return ns["reset_to_pretrained"]


def test_sections_5_6_and_7_reset_before_using_the_model(notebook):
    for marker, first_use in (
        ("def reset_to_pretrained():", "pipe.fill_mask("),
        ("SEED = 0  # @param", "pipe.unigram_baseline("),
        ("EPOCHS = 2  # @param", "pipe.adapt("),
    ):
        source = _cell(notebook, marker)
        assert "\nreset_to_pretrained()\n" in source, marker
        assert source.index("\nreset_to_pretrained()\n") < source.index(first_use), marker
    section6 = _cell(notebook, "SEED = 0  # @param")
    assert "if frozen_test['adapted']:\n    raise RuntimeError(" in section6


def test_reset_to_pretrained_reloads_only_an_adapted_pipeline(notebook):
    loads: list[str] = []

    class Fresh:
        adapter = None

    def from_pretrained(*, weights_dir):
        loads.append(str(weights_dir))
        return Fresh()

    torch_stub = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=None))
    adapted = types.SimpleNamespace(adapter={"best_epoch": 2})
    ns = {
        "gc": __import__("gc"),
        "torch": torch_stub,
        "BERTMaskedLMPipeline": types.SimpleNamespace(from_pretrained=from_pretrained),
        "WEIGHTS_DIR": Path("weights/bert-base-uncased"),
        "pipe": adapted,
    }
    reset = _reset_function(notebook, ns)
    reset()
    assert isinstance(ns["pipe"], Fresh) and loads == [str(Path("weights/bert-base-uncased"))]
    reset()  # already pretrained: no second load
    assert len(loads) == 1


def test_adapt_refuses_an_already_adapted_pipeline(forbid_model_imports):
    pipe = _tokenizer_only_pipe()
    pipe.adapter = {"best_epoch": 1, "trainable_names": []}
    with pytest.raises(ValueError, match="already adapted"):
        pipe.adapt([_doc(i) for i in range(8)])


def test_reload_parity_failure_explains_the_cause(notebook):
    section9 = _cell(notebook, "artifact_dir = Path(")
    assert "\nassert " not in "\n" + section9
    assert "raise RuntimeError(f'Reload parity failed: {parity}." in section9 and "Re-run from Section 7" in section9


# --- MLM-m1: a negative result is a printed verdict, not a crash ------------------------------------------


def _metrics(perplexity: float) -> dict:
    return {"perplexity": perplexity, "bits_per_token": 1.0, "top1_accuracy": 0.5, "top5_accuracy": 0.7, "n_masked": 9}


def test_section_8_reports_not_improved_and_keeps_a_run_history(notebook, tmp_path, monkeypatch, capsys):
    source = _cell(notebook, "adapted_test = pipe.evaluate(test_records")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    results = iter([_metrics(16.0), _metrics(11.0), _metrics(12.0), _metrics(10.5)])
    ns = {
        "json": json,
        "pipe": types.SimpleNamespace(evaluate=lambda records, **kw: next(results)),
        "test_records": [], "val_records": [], "unigram": _metrics(1175.0), "frozen_test": _metrics(15.0),
        "MASK_RATE": 0.15, "SEED": 0, "MODEL_ID": "m", "MODEL_REVISION": "r", "MODEL_KEY": "k",
        "data_source": "sample", "dataset_manifests": {}, "disjoint": {}, "USE_BYOD": False,
        "adapt_result": {"n_trainable": 7_087_872, "best_epoch": 0, "history": []}, "adapt_seconds": 1.0,
        "TRAINABLE_LAYERS": 1, "EPOCHS": 2, "LEARNING_RATE": 5e-5,
    }
    exec(compile(source, "<section 8>", "exec"), ns)  # adapted 16.0 > frozen 15.0: must not raise
    assert ns["verdict"].startswith("not improved")
    ns["TRAINABLE_LAYERS"] = 4
    exec(compile(source, "<section 8>", "exec"), ns)
    assert ns["verdict"].startswith("improved")
    assert [(h["run"], h["trainable_layers"], h["delta"]) for h in ns["run_history"]] == [(1, 1, 1.0), (2, 4, -3.0)]
    report = json.loads((tmp_path / "outputs" / "bert_masked_lm_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == ns["verdict"]


# --- MLM-M3: BYOD limits are the enforced ones, checked in Section 4 before any model call -----------------


def test_byod_minimum_matches_the_split_arithmetic():
    minimum = byod_minimum_records()
    assert minimum == 12
    docs = [_doc(i) for i in range(80)]
    for n in range(1, 81):
        if n < minimum:
            with pytest.raises(ValueError, match=r"split has \d+ records .* at least 12 distinct texts"):
                split_dataset(docs[:n])
        else:
            splits = split_dataset(docs[:n])
            assert len(splits["train"]) >= 8 and len(splits["validation"]) >= 1 and len(splits["test"]) >= 2


def test_check_byod_tokens_names_the_split_and_record():
    splits = {"train": [_doc(1)], "test": [{"id": "long-doc", "text": "w " * MAX_TEXT_TOKENS}]}
    with pytest.raises(ValueError, match=r"test:long-doc \(514 tokens\).*split each long document"):
        check_byod_tokens(splits, _tokenizer_only_pipe().count_tokens, max_tokens=MAX_TEXT_TOKENS)
    ok = check_byod_tokens({"train": [_doc(1)]}, _tokenizer_only_pipe().count_tokens, max_tokens=MAX_TEXT_TOKENS)
    assert ok["longest_tokens"] == len(_encode(_doc(1)["text"])) and ok["records_checked"] == 1


def test_section_4_accepts_twenty_records_and_reports_duplicates(notebook, tmp_path, monkeypatch, capsys):
    path = _write_jsonl(tmp_path / "mine.jsonl", [_doc(i) for i in range(20)] + [{**_doc(3), "id": "dup"}])
    ns = _run_section4(notebook, tmp_path, monkeypatch, byod_path=str(path))
    assert {k: len(v) for k, v in ns["splits"].items()} == {"test": 4, "validation": 3, "train": 13}
    assert ns["dropped_duplicates"] == 1 and "'duplicate_texts_dropped': 1" in capsys.readouterr().out
    assert list(ns["clozes"]) == ["byod-test-00", "byod-test-01", "byod-test-02"]
    assert "not unseen" in ns["cloze_source"]


def test_section_4_refuses_a_small_corpus_naming_the_split_and_minimum(notebook, tmp_path, monkeypatch):
    path = _write_jsonl(tmp_path / "small.jsonl", [_doc(i) for i in range(11)])
    with pytest.raises(ValueError, match=r"the train split has 7 records .* at least 12 distinct texts"):
        _run_section4(notebook, tmp_path, monkeypatch, byod_path=str(path))


def test_section_4_refuses_an_over_ceiling_record_by_id(notebook, tmp_path, monkeypatch):
    records = [_doc(i) for i in range(20)] + [{"id": "huge", "text": "word " * 600}]
    path = _write_jsonl(tmp_path / "long.jsonl", records)
    with pytest.raises(ValueError, match=r":huge \(602 tokens\)"):
        _run_section4(notebook, tmp_path, monkeypatch, byod_path=str(path))


def test_section_4_skips_or_refuses_documents_without_a_usable_cloze_word(notebook, tmp_path, monkeypatch):
    mixed = [_doc(i, opening="We do it") for i in range(16)] + [_doc(i) for i in range(16, 20)]
    ns = _run_section4(notebook, tmp_path, monkeypatch, byod_path=str(_write_jsonl(tmp_path / "mixed.jsonl", mixed)))
    assert ns["clozes"] and all(c[0].count("[MASK]") == 1 for c in ns["clozes"].values())
    terse = [{"id": f"t{i:02d}", "text": f"We do it. Run {i} is ok."} for i in range(20)]
    with pytest.raises(ValueError, match="no cloze to fill"):
        _run_section4(notebook, tmp_path, monkeypatch, byod_path=str(_write_jsonl(tmp_path / "terse.jsonl", terse)))


# --- MLM-m3: BYOD path field and upload guards ----------------------------------------------------------


def test_byod_path_works_without_google_colab_and_bad_paths_are_explained(notebook, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(FileNotFoundError, match="is not a file in this runtime"):
        _run_section4(notebook, tmp_path, monkeypatch, byod_path=str(tmp_path / "missing.csv"))
    with pytest.raises(RuntimeError, match="needs Google Colab.*set BYOD_PATH"):
        _run_section4(notebook, tmp_path, monkeypatch)


def test_empty_upload_gives_a_recovery_message(notebook, tmp_path, monkeypatch):
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {})
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(RuntimeError, match=r"received 0\).*set BYOD_PATH"):
        _run_section4(notebook, tmp_path, monkeypatch)


# --- MLM-M4 / MLM-m2 / MLM-m5: guided layer, labelled figures, single braces ------------------------------


def test_guided_layer_is_present(notebook):
    markdown = _markdown(notebook)
    for marker, least in (
        ("**Who this is for.**", 1),
        ("**Input → Model → Output.**", 1),
        ("**How to use this notebook.**", 1),
        ("**Roadmap:**", 1),
        ("**Predict before running:**", 6),
        ("**What to notice:**", 6),
        ("<summary>Check your reasoning</summary>", 6),
        ("**Predict → Change → Run → Observe → Explain**", 1),
        ("## Troubleshooting", 1),
        ("## Glossary", 1),
        ("## Conclusion (your notes)", 1),
        ("> **Infrastructure.**", 3),
    ):
        assert markdown.count(marker) >= least, marker
    closing = notebook["cells"][-1]["source"]
    closing = "".join(closing)
    for setting, section in (("EPOCHS = 4", "Section 7"), ("`SEED`", "Section 6"), ("`POOLING`", "Section 5")):
        line = next(x for x in closing.splitlines() if setting in x)
        assert section in line and "Run after" in line, setting
    assert "set `TRAINABLE_LAYERS = 1`" in closing and "select the Section 7 cell and choose **Runtime → Run after**" in closing
    objectives = markdown.split("**Learning objectives:**", 1)[1].split("\n", 1)[0]
    assert len(re.findall(r"Sections? \d", objectives)) >= 6 and "install the pinned runtime" not in objectives


def test_figures_name_their_environment_and_markdown_has_single_braces(notebook):
    markdown = _markdown(notebook)
    assert "{{" not in markdown and "}}" not in markdown
    assert "[A-Za-z0-9_.:-]{1,64}" in markdown and "`{id, text}`" in markdown
    for stale in ("about four minutes of model time", "the build record measured", "about seven seconds on CPU"):
        assert stale not in markdown
    assert "12.66" in markdown and "local Windows CPU pre-flight" in markdown
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "12.59 masked perplexity in the recorded run" not in readme and "12.66 on the Kaggle Tesla T4" in readme


def test_no_learner_cell_uses_a_bare_assert(notebook):
    learner = [
        src
        for src in _code_cells(notebook)
        if "# dimer: kernel cell" not in src and "carried verbatim" not in src and '"""' not in src[:4]
    ]
    assert learner and all("\nassert " not in "\n" + src for src in learner)
