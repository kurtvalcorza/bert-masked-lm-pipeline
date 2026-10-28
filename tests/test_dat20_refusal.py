"""DAT20: a refused BYOD input in Section 4 shows exactly one line, ``Refused: <file>: <reason>``, with no traceback,
and the cell raises so Run all stops there. The router relays such a refusal as that one line; a genuine failure keeps
its traceback. The Section 4 cell is executed from the committed notebook with a tokenizer-only pipeline."""
# ruff: noqa: E501

from __future__ import annotations

import json

import pytest

from test_bert_masked_lm_colab_review_fixes import (
    _cell,
    _code_cells,
    _doc,
    _run_section4,
    _write_jsonl,
    notebook,  # noqa: F401  (module-scoped fixture)
)

REFUSAL_CELL = "bert_masked_lm-17"


def _line(ns: dict) -> str:
    refusal = ns.get("_refusal")
    assert refusal is not None, "the cell did not raise a refusal"
    assert type(refusal).__name__ == "BYODRefusal" and refusal.dimer_refusal is True
    rendered = refusal._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0] and "Traceback" not in rendered[0]
    return rendered[0]


def test_the_scripted_refusal_input_is_one_line_naming_the_file(notebook, tmp_path, monkeypatch, capsys):  # noqa: F811
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime-host\n", encoding="utf-8")
    ns = _run_section4(notebook, tmp_path, monkeypatch, byod_path=str(hostname))
    assert _line(ns).startswith("Refused: hostname: BYOD corpora must be .csv, .json, .jsonl or .txt. ")
    assert "Traceback" not in capsys.readouterr().out


def test_a_directory_is_refused_by_name(notebook, tmp_path, monkeypatch):  # noqa: F811
    folder = tmp_path / "etc"
    folder.mkdir()
    ns = _run_section4(notebook, tmp_path, monkeypatch, byod_path=str(folder))
    assert _line(ns).startswith("Refused: etc: BYOD_PATH = ")


def test_a_refusal_invalidates_earlier_results_in_the_cell(notebook, tmp_path, monkeypatch):  # noqa: F811
    ok = _run_section4(notebook, tmp_path, monkeypatch, byod_path=str(_write_jsonl(tmp_path / "ok.jsonl", [_doc(i) for i in range(20)])))
    assert "_refusal" not in ok and "train_records" in ok
    small = _write_jsonl(tmp_path / "small.jsonl", [_doc(i) for i in range(11)])
    ns = _run_section4(notebook, tmp_path, monkeypatch, byod_path=str(small))
    assert _line(ns).startswith("Refused: small.jsonl: the train split has 7 records")
    assert ns["RUN_STATE"]["refused"].startswith("small.jsonl: ")
    status = json.loads((tmp_path / "outputs" / "bert_masked_lm_run_status.json").read_text(encoding="utf-8"))
    assert status["status"].startswith("invalidated") and "earlier run" in status["files_in_outputs"]
    assert not {"splits", "train_records", "test_records", "clozes"} & set(ns)


def test_a_genuine_loader_failure_keeps_its_type(notebook, tmp_path, monkeypatch):  # noqa: F811
    import test_bert_masked_lm_colab_review_fixes as helpers

    real = helpers._carried_namespace

    def broken():
        ns = real()

        def load_byod_dataset(path):
            raise RuntimeError("internal bug")

        ns["load_byod_dataset"] = load_byod_dataset
        return ns

    monkeypatch.setattr(helpers, "_carried_namespace", broken)
    path = _write_jsonl(tmp_path / "ok.jsonl", [_doc(i) for i in range(20)])
    with pytest.raises(RuntimeError, match="internal bug"):
        _run_section4(notebook, tmp_path, monkeypatch, byod_path=str(path))


def test_the_refusal_cell_id_and_the_router_channel(notebook):  # noqa: F811
    cell = next(c for c in notebook["cells"] if c.get("id") == REFUSAL_CELL)
    source = "".join(cell["source"])
    assert source == _cell(notebook, "USE_BYOD = False  # @param") and "BYOD_PATH" in source
    assert "raise BYODRefusal(BYOD_LABEL, " in source and "'byod_refused'" not in source
    router = _code_cells(notebook)[1]
    assert 'if getattr(exc, "dimer_refusal", False):' in router
    assert 'elif kind == "refusal":' in router and "class IsolatedCellRefusal(Exception):" in router
    assert 'return [f"Refused: {self}"]' in router
