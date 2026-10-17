# ruff: noqa: E501  -- assertion messages quote notebook text on one line
"""Regression tests for the tutorial-triage review of 2026-10-15, row 5 (T5-1..T5-8), fixed 2026-10-17.

CI's lightweight install only: the selection rule and the fresh-test draw run on synthetic records, notebook cells are
executed from the committed JSON with stand-ins, and the provenance test reads the stamped revision with git.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from bert_masked_lm_pipeline import samples

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "bert_masked_lm_colab.ipynb"
sys.path.insert(0, str(ROOT / "tools"))
import build_notebook  # noqa: E402
from notebook_template import TEMPLATE  # noqa: E402


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def _markdown(notebook: dict) -> str:
    return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")


# --- T5-1: the pre-registered selection rule and the fresh test --------------------------------------------


def test_selection_prefers_the_smaller_config_within_tau():
    picked = samples.select_configuration({"B": 10.89, "A": 10.87, "C": 10.80}, 12.89)
    assert picked["v_star"] == 10.80 and picked["within_tau"] == ["B", "A", "C"] and picked["picked"] == "B"
    assert picked["kept"] == "B" and picked["selection_split"] == "validation" and picked["tau"] == 0.10
    assert samples.select_configuration({"B": 11.0, "A": 10.88, "C": 10.80}, 12.89)["picked"] == "A"
    assert samples.select_configuration({"B": 11.5, "A": 11.2, "C": 10.80}, 12.89)["picked"] == "C"


def test_selection_keeps_the_frozen_model_unless_the_pick_beats_it_by_tau():
    assert samples.select_configuration({"B": 12.85, "A": 12.9, "C": 12.95}, 12.89)["kept"] == "frozen"
    assert samples.select_configuration({"B": 12.78, "A": 12.9, "C": 12.95}, 12.89)["kept"] == "B"
    with pytest.raises(ValueError, match="expected validation results"):
        samples.select_configuration({"A": 10.0}, 12.0)


def test_grid_matches_the_preregistration():
    assert samples.SELECTION_GRID == (("B", 2, 5e-5), ("A", 4, 5e-5), ("C", 4, 1e-4)) and samples.SELECTION_TAU == 0.10
    prereg = (ROOT / "docs" / "reselection-preregistration-2026-10-17.md").read_text(encoding="utf-8")
    for needle in ("| **B** | 2 | 5e-5 |", "| **A** | 4 | 5e-5 |", "| **C** | 4 | 1e-4 |", "**τ = 0.10**", "random.Random(20261017)", "**B → A → C**"):
        assert needle in prereg, needle


def _corpus(n_dev: int = 40) -> dict:
    def rec(split, i, text=None):
        return {"id": f"{split}-p{i}", "text": text or f"{split} abstract number {i} " + "word " * 60, "paper_id": f"{split}{i}"}

    dev = [rec("dev", i) for i in range(n_dev)]
    dev.append({**rec("dev", 999), "text": "train abstract number 0 " + "word " * 60})  # a cross-member duplicate text
    return {"train": [rec("train", i) for i in range(5)], "dev": dev, "test": [rec("test", i) for i in range(5)]}


def test_fresh_test_is_paper_and_text_disjoint_and_pinned():
    corpus = _corpus()
    used = {"train": corpus["train"], "validation": corpus["dev"][:5], "test": corpus["test"], "clozes": corpus["dev"][5:8]}
    fresh = samples.build_fresh_test(corpus, used, seed=1, size=10, expected_pool=32, expected_sha256=None)
    used_texts = {r["text"].lower() for part in used.values() for r in part}
    assert len(fresh) == 10 and not ({r["text"].lower() for r in fresh} & used_texts)
    assert not ({r["paper_id"] for r in fresh} & {r["paper_id"] for part in used.values() for r in part})
    assert "dev999" not in {r["paper_id"] for r in fresh}  # the text shared with train is excluded
    digest = samples.paper_ids_digest(fresh)
    assert samples.build_fresh_test(corpus, used, seed=1, size=10, expected_pool=32, expected_sha256=digest) == fresh
    with pytest.raises(ValueError, match="does not reproduce the pinned paper ids"):
        samples.build_fresh_test(corpus, used, seed=2, size=10, expected_pool=32, expected_sha256=digest)
    with pytest.raises(ValueError, match="never-scored dev pool has 32 abstracts, expected 559"):
        samples.build_fresh_test(corpus, used, seed=1, size=10)


def test_pinned_fresh_test_ids_file_matches_the_digest():
    ids = (ROOT / "docs" / "fresh-test-paper-ids-2026-10-17.txt").read_text(encoding="utf-8").split()
    assert len(ids) == samples.FRESH_TEST_SIZE == 100 and len(set(ids)) == 100
    assert hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest() == samples.FRESH_TEST_PAPER_IDS_SHA256


def test_notebook_scores_the_fresh_test_once_and_labels_the_exposed_test(notebook):
    code = "\n".join(_code(notebook))
    assert code.count("adapted_test = pipe.evaluate(test_records") == 1
    assert "score_test = run_role == 'pre-registered' and not kept_frozen" in code
    assert "scored once" not in code.lower() and "scored once" not in _markdown(notebook).lower()
    assert "parity_records = val_records[:10]" in code
    markdown = _markdown(notebook)
    assert "adapted test perplexity, local Windows CPU" not in markdown
    assert "The test split was never used for training or epoch selection" not in markdown
    assert "`TRAINABLE_LAYERS` and `LEARNING_RATE` are selected on the **validation** split" in markdown
    assert markdown.count("exposed development test") >= 3


def test_exploratory_run_does_not_score_the_fresh_test(notebook, tmp_path, monkeypatch):
    source = next(s for s in _code(notebook) if "adapted_test = pipe.evaluate(test_records" in s)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    scored = []
    m = {"perplexity": 11.0, "bits_per_token": 1.0, "top1_accuracy": 0.5, "top5_accuracy": 0.7, "n_masked": 9}
    ns = {
        "json": json, "RUN_STATE": {"refused": None}, "Path": Path, "time": __import__("time"), "differs": ["TRAINABLE_LAYERS"],
        "paired_bootstrap_delta": None,
        "pipe": types.SimpleNamespace(evaluate=lambda records, **kw: scored.append(records[0]["id"]) or dict(m)),
        "test_records": [{"id": "fresh-test-0000"}], "val_records": [{"id": "validation-0000"}], "exposed_test_records": [{"id": "test-0000"}],
        "unigram": m, "frozen_test": m, "frozen_exposed": m, "MASK_RATE": 0.15, "SEED": 0, "MODEL_ID": "m", "MODEL_REVISION": "r",
        "MODEL_KEY": "k", "data_source": "sample", "dataset_manifests": {"test": {"digest": "d"}}, "disjoint": {}, "USE_BYOD": False,
        "paper_ids_digest": lambda r: "d", "selection": {"picked": "A", "kept": "A"}, "grid_rows": [], "run_role": "exploratory",
        "adapt_result": {"n_trainable": 1, "best_epoch": 1, "history": [{"val": dict(m, perplexity=12.0)}]}, "adapt_seconds": 1.0,
        "TRAINABLE_LAYERS": 1, "EPOCHS": 2, "LEARNING_RATE": 5e-5,
    }
    exec(compile(source, "<section 8>", "exec"), ns)
    assert scored == ["validation-0000"] and ns["adapted_test"] is None and ns["verdict"].startswith("exploratory")
    assert ns["run_history"][-1]["test_adapted"] is None and ns["run_history"][-1]["val_delta"] == -1.0
    assert ns["test_look"] == "not scored (exploratory)" and ns["report_path"].endswith("_evaluation_report_exploratory_1.json")
    assert not (tmp_path / "outputs" / "bert_masked_lm_evaluation_report.json").exists()
    assert not (tmp_path / "outputs" / "bert_masked_lm_preregistered_evaluations.json").exists()


# --- Score-once rule (SCORE_ONCE_RULE.md item 7): every pre-registered setting gates the fresh test ------------


def _training_cell_role(notebook, **changes):
    """Run Section 7's training cell with stand-ins and return (run_role, differs)."""
    source = next(s for s in _code(notebook) if "TRAINABLE_LAYERS_OVERRIDE = 0  # @param" in s)
    override = changes.pop("TRAINABLE_LAYERS_OVERRIDE", 0)
    source = source.replace("TRAINABLE_LAYERS_OVERRIDE = 0  # @param", f"TRAINABLE_LAYERS_OVERRIDE = {override}  # @param", 1)
    history = [{"val": {"perplexity": 12.9}}, {"val": {"perplexity": 11.0}}]
    ns = {
        "RUN_STATE": {"refused": None}, "SELECTION_GRID": samples.SELECTION_GRID, "time": __import__("time"),
        "selection": {"picked": changes.pop("picked", "A"), "kept": "A"}, "grid_rows": [], "report": lambda entry: None,
        "PREREGISTERED": {"data source": "SciTLDR-A sample", "SPLIT_SEED": 42, "EPOCHS": 2, "BATCH_SIZE": 8, "MASK_RATE": 0.15, "SEED": 0, "config": "A"},
        "USE_BYOD": False, "SPLIT_SEED": 42, "EPOCHS": 2, "BATCH_SIZE": 8, "MASK_RATE": 0.15, "SEED": 0,
        "reset_to_pretrained": lambda: None, "train_records": [], "val_records": [],
        "pipe": types.SimpleNamespace(adapt=lambda *a, **kw: {"history": history, "objective": "mlm", "n_trainable": 1, "n_total": 2, "n_train_tokens": 3, "best_epoch": 1, "selection": "val"}),
    }
    ns.update(changes)
    exec(compile(source, "<section 7 training>", "exec"), ns)
    return ns["run_role"], ns["differs"]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"EPOCHS": 4}, ["EPOCHS"]),
        ({"BATCH_SIZE": 4}, ["BATCH_SIZE"]),
        ({"SEED": 1}, ["SEED"]),
        ({"MASK_RATE": 0.2}, ["MASK_RATE"]),
        ({"SPLIT_SEED": 7}, ["SPLIT_SEED"]),
        ({"USE_BYOD": True}, ["data source"]),
        ({"picked": "C"}, ["config", "LEARNING_RATE"]),
        ({"TRAINABLE_LAYERS_OVERRIDE": 1}, ["TRAINABLE_LAYERS"]),
    ],
)
def test_any_changed_setting_makes_the_run_exploratory(notebook, changes, expected):
    role, differs = _training_cell_role(notebook, **changes)
    assert role == "exploratory" and differs == expected


def _with_recorded(source, recorded):
    """Replace the committed RECORDED_EVALUATION constant (None simulates the state before any hosted record)."""
    source, n = re.subn(r"^( *)RECORDED_EVALUATION = .*$", lambda m: f"{m.group(1)}RECORDED_EVALUATION = {recorded!r}", source, count=1, flags=re.M)
    assert n == 1
    return source


def test_recorded_evaluation_points_at_the_hosted_t4_run(notebook):
    source = next(s for s in _code(notebook) if "adapted_test = pipe.evaluate(test_records" in s)
    line = re.search(r"^ *RECORDED_EVALUATION = (.*)$", source, flags=re.M).group(1)
    recorded = __import__("ast").literal_eval(line)
    assert recorded["commit"] == "65e6a28" and recorded["notebook_blob"] == "6d4e7d99" and recorded["config"] == "A"
    assert recorded["fresh_test_perplexity"] == {"frozen": 14.3927, "adapted": 11.9792, "delta": -2.4135, "delta_ci95": [-2.999, -1.902]}
    assert recorded["exposed_dev_test_perplexity"] == {"frozen": 15.0932, "adapted": 12.6567}
    assert "'reproduction_of'" in source


def test_a_later_identical_hosted_run_is_labelled_a_reproduction(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    scored: list[str] = []
    source = next(s for s in _code(notebook) if "adapted_test = pipe.evaluate(test_records" in s)
    recorded = __import__("ast").literal_eval(re.search(r"^ *RECORDED_EVALUATION = (.*)$", source, flags=re.M).group(1))
    ns = _section8(notebook, tmp_path, scored, recorded=recorded)
    assert ns["test_look"] == "reproduction" and "fresh-test-0000" in scored
    assert "'reproduction_of': 'hosted Colab T4, Colab account 2, 2026-10-17 (docs/release-verification.md)'" in capsys.readouterr().out
    report = json.loads((tmp_path / "outputs" / "bert_masked_lm_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["test_look"] == "reproduction"


def test_pre_registered_settings_score_and_a_repeat_is_a_reproduction(notebook, tmp_path, monkeypatch):
    assert _training_cell_role(notebook) == ("pre-registered", [])
    source = _with_recorded(next(s for s in _code(notebook) if "adapted_test = pipe.evaluate(test_records" in s), None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    scored = []
    m = {"perplexity": 11.0, "bits_per_token": 1.0, "top1_accuracy": 0.5, "top5_accuracy": 0.7, "n_masked": 9, "per_record": [[2.0, 1]]}
    ns = {
        "json": json, "RUN_STATE": {"refused": None}, "Path": Path, "time": __import__("time"), "differs": [],
        "paired_bootstrap_delta": samples.__dict__.get("paired_bootstrap_delta") or __import__("bert_masked_lm_pipeline").paired_bootstrap_delta,
        "pipe": types.SimpleNamespace(evaluate=lambda records, **kw: scored.append(records[0]["id"]) or dict(m)),
        "test_records": [{"id": "fresh-test-0000"}], "val_records": [{"id": "validation-0000"}], "exposed_test_records": [{"id": "test-0000"}],
        "unigram": m, "frozen_test": dict(m, perplexity=14.0), "frozen_exposed": m, "MASK_RATE": 0.15, "SEED": 0, "MODEL_ID": "m",
        "MODEL_REVISION": "r", "MODEL_KEY": "k", "data_source": "sample", "dataset_manifests": {"test": {"digest": "d"}}, "disjoint": {},
        "USE_BYOD": False, "paper_ids_digest": lambda r: "d", "selection": {"picked": "A", "kept": "A"}, "grid_rows": [],
        "run_role": "pre-registered", "adapt_result": {"n_trainable": 1, "best_epoch": 1, "history": [{"val": dict(m, perplexity=12.0)}]},
        "adapt_seconds": 1.0, "TRAINABLE_LAYERS": 4, "EPOCHS": 2, "LEARNING_RATE": 5e-5,
    }
    exec(compile(source, "<section 8>", "exec"), ns)
    assert ns["test_look"] == "pre-registered evaluation" and "fresh-test-0000" in scored
    exec(compile(source, "<section 8>", "exec"), ns)
    assert ns["test_look"] == "reproduction" and scored.count("fresh-test-0000") == 2
    log = json.loads((tmp_path / "outputs" / "bert_masked_lm_preregistered_evaluations.json").read_text(encoding="utf-8"))
    assert [e["test_look"] for e in log] == ["pre-registered evaluation", "reproduction"]
    report = json.loads((tmp_path / "outputs" / "bert_masked_lm_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["test_look"] == "reproduction" and report["settings_differing_from_preregistration"] == []


def test_paired_bootstrap_and_near_duplicate_warning():
    from bert_masked_lm_pipeline import near_duplicate_warnings, paired_bootstrap_delta

    frozen = {"per_record": [[2.0 + 0.1 * i, 1] for i in range(20)]}
    adapted = {"per_record": [[1.8 + 0.1 * i, 1] for i in range(20)]}
    ci = paired_bootstrap_delta(frozen, adapted, n_boot=200, seed=0)
    assert ci["ci_low"] <= ci["delta"] <= ci["ci_high"] < 0
    assert paired_bootstrap_delta(frozen, adapted, n_boot=200, seed=0) == ci
    with pytest.raises(ValueError, match="same masked positions"):
        paired_bootstrap_delta(frozen, {"per_record": [[1.0, 2]] * 20})
    base = "the model learns graph structure from molecules and reports strong results on several public benchmarks today"
    warn = near_duplicate_warnings({"train": [{"id": "t", "text": base}], "test": [{"id": "x", "text": base + " again"}, {"id": "y", "text": "an unrelated abstract about speech"}]})
    assert len(warn) == 1 and warn[0].startswith("test:x")


def test_exploratory_adapter_is_not_labelled_kept(notebook):
    section9 = next(s for s in _code(notebook) if "artifact_dir = Path(" in s)
    assert "'no: exploratory run' if run_role == 'exploratory'" in section9 and "'kept': kept_label" in section9
    assert "artifact_dir = Path(f'outputs/bert_masked_lm_adapter{run_tag}')" in section9


def _section8(notebook, tmp_path, scored, *, run_role="pre-registered", differs=(), recorded=None):
    source = next(s for s in _code(notebook) if "adapted_test = pipe.evaluate(test_records" in s)
    source = _with_recorded(source, recorded)
    m = {"perplexity": 11.0, "bits_per_token": 1.0, "top1_accuracy": 0.5, "top5_accuracy": 0.7, "n_masked": 9, "per_record": [[2.0, 1]]}
    from bert_masked_lm_pipeline import paired_bootstrap_delta

    ns = {
        "json": json, "RUN_STATE": {"refused": None}, "Path": Path, "time": __import__("time"), "differs": list(differs),
        "paired_bootstrap_delta": paired_bootstrap_delta,
        "pipe": types.SimpleNamespace(evaluate=lambda records, **kw: scored.append(records[0]["id"]) or dict(m)),
        "test_records": [{"id": "fresh-test-0000"}], "val_records": [{"id": "validation-0000"}], "exposed_test_records": [{"id": "test-0000"}],
        "unigram": m, "frozen_test": dict(m, perplexity=14.0), "frozen_exposed": m, "MASK_RATE": 0.15, "SEED": 0, "MODEL_ID": "m",
        "MODEL_REVISION": "r", "MODEL_KEY": "k", "data_source": "sample", "dataset_manifests": {"test": {"digest": "d"}}, "disjoint": {},
        "USE_BYOD": False, "paper_ids_digest": lambda r: "d", "selection": {"picked": "A", "kept": "A"}, "grid_rows": [],
        "run_role": run_role, "adapt_result": {"n_trainable": 1, "best_epoch": 1, "history": [{"val": dict(m, perplexity=12.0)}]},
        "adapt_seconds": 1.0, "TRAINABLE_LAYERS": 4, "EPOCHS": 2, "LEARNING_RATE": 5e-5,
    }
    exec(compile(source, "<section 8>", "exec"), ns)
    return ns


def test_a_recorded_hosted_evaluation_makes_a_fresh_runtime_a_reproduction(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    scored: list[str] = []
    ns = _section8(notebook, tmp_path, scored, recorded={"revision": "hosted"})
    assert ns["test_look"] == "reproduction" and "fresh-test-0000" in scored


def test_exploratory_reports_never_overwrite_earlier_ones_or_the_preregistered_file(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "bert_masked_lm_evaluation_report.json").write_text("{\"test_look\": \"pre-registered evaluation\"}", encoding="utf-8")
    (tmp_path / "outputs" / "bert_masked_lm_evaluation_report_exploratory_1.json").write_text("{}", encoding="utf-8")
    scored: list[str] = []
    ns = _section8(notebook, tmp_path, scored, run_role="exploratory", differs=["EPOCHS"])
    assert ns["test_look"] == "not scored (exploratory)" and scored == ["validation-0000"]
    assert ns["report_path"].endswith("_evaluation_report_exploratory_2.json")
    assert (tmp_path / "outputs" / "bert_masked_lm_evaluation_report_exploratory_1.json").read_text(encoding="utf-8") == "{}"
    assert "pre-registered evaluation" in (tmp_path / "outputs" / "bert_masked_lm_evaluation_report.json").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("changes", "fresh"),
    [({}, True), ({"SPLIT_SEED": 7}, False), ({"USE_BYOD": True}, False), ({"SEED": 1}, False), ({"MASK_RATE": 0.2}, False)],
)
def test_section_6_reads_the_fresh_test_only_under_preregistered_settings(notebook, changes, fresh):
    source = next(s for s in _code(notebook) if "section6_preregistered = " in s)
    for name in ("SEED", "MASK_RATE"):
        if name in changes:
            default = "0" if name == "SEED" else "0.15"
            source = source.replace(f"{name} = {default}  # @param", f"{name} = {changes.pop(name)}  # @param", 1)
    read: list[str] = []
    m = {"perplexity": 11.0, "bits_per_token": 1.0, "top1_accuracy": 0.5, "top5_accuracy": 0.7, "n_masked": 9, "n_records": 1, "verdict": "measured", "adapted": False, "definitions": {}}

    def evaluate(records, **kw):
        read.append(records[0]["id"])
        return dict(m)

    ns = {
        "RUN_STATE": {"refused": None}, "time": __import__("time"), "reset_to_pretrained": lambda: None,
        "USE_BYOD": False, "SPLIT_SEED": 42, "train_records": [], "test_records": [{"id": "fresh-test-0000"}],
        "val_records": [{"id": "validation-0000"}], "exposed_test_records": [{"id": "test-0000"}],
        "pipe": types.SimpleNamespace(evaluate=evaluate, unigram_baseline=lambda train, records, **kw: dict(m, baseline="floor")),
    }
    ns.update(changes)
    exec(compile(source, "<section 6>", "exec"), ns)
    assert ns["section6_preregistered"] is fresh
    assert ("fresh-test-0000" in read) is fresh and ("test-0000" in read) is fresh


# --- T5-2 / T5-8: limitation and wording ----------------------------------------------------------------


def test_pretraining_overlap_and_reload_tolerance_are_stated(notebook):
    markdown = _markdown(notebook)
    assert "BooksCorpus" in markdown and "a small pretraining overlap cannot be ruled out" in markdown
    assert "Reload parity was exact" not in markdown and "within 1e-6" in markdown


# --- T5-4: spec 2.5, kind, use notice --------------------------------------------------------------------


def test_spec_kind_and_use_notice(notebook):
    dimer = notebook["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL" == TEMPLATE["kind"]
    first_code = next(i for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code")
    before = "\n".join("".join(c["source"]) for c in notebook["cells"][:first_code])
    assert "> **⚠️ Notice.** For research, training and evaluation only." in before
    assert "accept no liability for\n> its use." in before and "[!WARNING]" not in before


# --- T5-5: DAT20 refusals and EXE5 ---------------------------------------------------------------------


def test_cells_after_section_4_skip_when_byod_was_refused(notebook, capsys):
    guarded = [s for s in _code(notebook) if "if RUN_STATE['refused']:\n    print({'skipped'" in s]
    assert len(guarded) == 6
    ns = {"RUN_STATE": {"refused": "mine.jsonl: too small"}}
    for source in guarded:
        exec(compile(source.replace("# @param", "#"), "<guarded>", "exec"), ns)
    out = capsys.readouterr().out
    assert out.count("'skipped'") == 6 and "Traceback" not in out


def test_isolated_env_variable_is_documented(notebook):
    assert "DIMER_ISOLATED_ENV" in _markdown(notebook)


# --- T5-6: the provenance stamp names a revision that holds the carried modules --------------------------


def test_stamped_revision_holds_the_carried_modules(notebook):
    meta = notebook["metadata"]["dimer"]["generated_from"]
    rev = meta["revision"]
    try:
        present = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{rev}^{{commit}}"], capture_output=True).returncode == 0
    except OSError:
        present = False
    if not present:
        pytest.skip(f"revision {rev[:12]} is not in this checkout (CI fetches full history)")
    joined = b""
    for rel in meta["modules"]:
        committed = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{rel}"], capture_output=True, check=True).stdout
        assert committed.replace(b"\r\n", b"\n") == (ROOT / rel).read_bytes().replace(b"\r\n", b"\n"), rel
        joined += committed.replace(b"\r\n", b"\n")
    ctx = build_notebook.load_context(ROOT, TEMPLATE, rev)
    order = [f"{ctx['pkg_rel']}/{m}" for m in ctx["modules"]]
    assert order == meta["modules"] and hashlib.sha256(joined).hexdigest() == meta["module_sha256"]


def test_generator_refuses_a_dirty_carried_module(tmp_path):
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git not available")
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    module = repo / "src" / "m.py"
    module.write_bytes(b"X = 1\n")
    env = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), *env, "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), *env, "commit", "-q", "-m", "init"], check=True)
    head = build_notebook._head_revision(repo, ["src/m.py"])
    assert len(head) == 40
    module.write_bytes(b"X = 2\n")
    with pytest.raises(SystemExit, match="differs from HEAD"):
        build_notebook._head_revision(repo, ["src/m.py"])
