import json
from types import SimpleNamespace

import conftest


def retrieval_row(recall):
    metrics = {
        "recall_at_1": recall,
        "recall_at_3": recall,
        "recall_at_5": recall,
        "mrr": recall,
    }
    return {
        "stage": "retrieval",
        "k": 5,
        "recall_at_k": recall,
        "pre_rerank": metrics,
        "post_rerank": metrics,
    }


def fake_config(rows, **options):
    return SimpleNamespace(
        evaluation_rows=rows, getoption=lambda name: options.get(name)
    )


def test_generation_tests_are_skipped_without_flag():
    marked = []

    class Item:
        def __init__(self, keywords):
            self.keywords = keywords

        def add_marker(self, marker):
            marked.append((self, marker))

    generation, retrieval = Item({"generation": 1}), Item({"retrieval": 1})
    conftest.pytest_collection_modifyitems(fake_config([]), [generation, retrieval])
    assert [item for item, _ in marked] == [generation]

    marked.clear()
    conftest.pytest_collection_modifyitems(
        fake_config([], **{"--run-generation": True}), [generation]
    )
    assert marked == []


def test_session_finish_writes_report_only_when_requested(tmp_path):
    rows = [retrieval_row(1.0), retrieval_row(0.0)]
    path = tmp_path / "nested" / "report.json"

    conftest.pytest_sessionfinish(SimpleNamespace(config=fake_config(rows)), 0)
    assert not path.exists()

    conftest.pytest_sessionfinish(
        SimpleNamespace(config=fake_config(rows, **{"--eval-report": str(path)})),
        1,
    )
    report = json.loads(path.read_text())
    assert report["pytest_exit_status"] == 1
    assert report["mean_recall_at_k"] == 0.5


def test_terminal_summary_prints_retrieval_metrics():
    lines = []
    reporter = SimpleNamespace(
        config=fake_config([retrieval_row(1.0), {"stage": "generation"}]),
        write_sep=lambda sep, text: lines.append(text),
        write_line=lambda text: lines.append(text),
    )
    conftest.pytest_terminal_summary(reporter)
    assert lines[0] == "Retrieval: mean recall at k=[5] = 1.000 (1 cases)"
    assert lines[1].startswith("pre_rerank: recall_at_1=1.000")


def test_terminal_summary_is_silent_without_retrieval_rows():
    lines = []
    reporter = SimpleNamespace(
        config=fake_config([]),
        write_sep=lambda *a: lines.append(a),
        write_line=lambda *a: lines.append(a),
    )
    conftest.pytest_terminal_summary(reporter)
    assert lines == []


def test_run_judge_is_off_by_default(run_judge):
    assert run_judge is False
