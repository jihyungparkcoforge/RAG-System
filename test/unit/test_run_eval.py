import json
from unittest.mock import MagicMock

import pytest

import run_eval
from fakes import make_pipeline


def retrieval_row(case_id):
    metrics = {"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr": 1.0}
    return {
        "case_id": case_id, "stage": "retrieval", "k": 5,
        "recall_at_k": 1.0, "candidate_recall": 1.0,
        "pre_rerank": metrics, "post_rerank": metrics,
        "gold_ranks": {f"{case_id}:1": {"pre_rerank": 2, "post_rerank": 1}},
        "gold_retrieved_by": {f"{case_id}:1": ["vector", "keyword"]},
        "gold_matched_keywords": {f"{case_id}:1": ["hotel", "limit"]},
    }


def generation_row(case, run_judge):
    return {
        "case_id": case["id"], "stage": "generation",
        "missing_facts": [], "invalid_citations": [], "missing_evidence_citations": ["2"],
        "decision_judge": {"probability_correct": 0.8, "verdict": "correct"} if run_judge else None,
    }


@pytest.fixture
def fake_harness(monkeypatch):
    """Replace model loading and per-case evaluation; record what main() does."""
    state = {"client": MagicMock(), "pipeline": make_pipeline(), "generated": []}
    monkeypatch.setattr(run_eval, "load_dotenv", lambda path: None)
    monkeypatch.setattr(run_eval, "build_eval_pipeline",
                        lambda db_dir: (state["pipeline"], state["client"]))
    monkeypatch.setattr(run_eval, "evaluate_retrieval", lambda p, case, k: retrieval_row(case["id"]))

    def attach(pipeline):
        pipeline.llm_client = MagicMock()
        return pipeline

    def generate(pipeline, case, run_judge):
        state["generated"].append((case["id"], run_judge))
        return generation_row(case, run_judge)

    monkeypatch.setattr(run_eval, "attach_llm_client", attach)
    monkeypatch.setattr(run_eval, "evaluate_generation", generate)
    return state


def run_main(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["run_eval.py", *args])
    run_eval.main()


def test_retrieval_only_run_prints_tables_and_cleans_up(monkeypatch, capsys, fake_harness):
    run_main(monkeypatch, "--case", "hotel_conflict")
    out = capsys.readouterr().out
    assert "=== Retrieval" in out
    assert "hotel_conflict" in out and "2→1" in out and "keyword,vector" in out and "hotel,limit" in out
    assert "=== Generation" not in out
    assert "mean recall@5: 1.000 (1 cases)" in out
    assert fake_harness["generated"] == []
    fake_harness["client"].delete_collection.assert_called_once_with(run_eval.EVAL_COLLECTION)


def test_generation_with_judge_writes_report(monkeypatch, capsys, tmp_path, fake_harness):
    for name in run_eval.JUDGE_ENV:
        monkeypatch.setenv(name, "x")
    report_path = tmp_path / "out" / "report.json"

    run_main(monkeypatch, "--case", "hotel_conflict", "--case", "meal_conflict",
             "--generation", "--judge", "--report", str(report_path))

    out = capsys.readouterr().out
    assert fake_harness["generated"] == [("hotel_conflict", True), ("meal_conflict", True)]
    assert "=== Generation (gold context) ===" in out
    assert "0.800" in out
    assert "generation regex+citation checks passed: 0/2" in out
    assert "judge verdicts: {'correct': 2}" in out
    fake_harness["pipeline"].llm_client.close.assert_called_once()
    report = json.loads(report_path.read_text())
    assert report["retrieval_cases_evaluated"] == 2
    assert report["judge_verdicts"] == {"correct": 2}


def test_generation_without_judge_shows_dashes(monkeypatch, capsys, fake_harness):
    run_main(monkeypatch, "--case", "hotel_conflict", "--generation")
    out = capsys.readouterr().out
    assert fake_harness["generated"] == [("hotel_conflict", False)]
    assert "judge verdicts" not in out


def test_unknown_case_exits(monkeypatch, fake_harness):
    with pytest.raises(SystemExit, match="Unknown case id: nope"):
        run_main(monkeypatch, "--case", "nope")


def test_judge_without_configuration_exits(monkeypatch, fake_harness):
    for name in run_eval.JUDGE_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="Missing judge configuration"):
        run_main(monkeypatch, "--generation", "--judge")


def test_judge_requires_generation_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--judge"])
    with pytest.raises(SystemExit):
        run_eval.parse_args()


def test_fmt_ranks_marks_missing_ranks():
    row = {"gold_ranks": {"a": {"pre_rerank": None, "post_rerank": 3}}}
    assert run_eval.fmt_ranks(row) == "-→3"
