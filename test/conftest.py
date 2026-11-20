"""Real-model evaluation fixtures with an isolated database and opt-in API calls."""

import json
from pathlib import Path

import pytest
from dotenv import load_dotenv
from evaluation import (
    EVAL_COLLECTION,
    JUDGE_ENV,
    ROOT,
    attach_llm_client,
    build_eval_pipeline,
    build_report,
    missing_env,
    rank_summary,
)


def pytest_addoption(parser):
    parser.addoption(
        "--run-generation",
        action="store_true",
        default=False,
        help="Call Microsoft Foundry for the generation evaluation (may incur costs).",
    )
    parser.addoption(
        "--run-judge",
        action="store_true",
        default=False,
        help="Also call the Microsoft-Decision-1.1 judge (requires --run-generation).",
    )
    parser.addoption(
        "--eval-report",
        default=None,
        help="Write evaluation results and generated answers to this JSON path.",
    )


def pytest_configure(config):
    # Foundry settings for --run-generation / --run-judge.
    load_dotenv(ROOT / ".env")
    config.addinivalue_line(
        "markers", "retrieval: real embedding/retrieval/reranking evaluation"
    )
    config.addinivalue_line(
        "markers", "generation: opt-in real Microsoft Foundry generation"
    )
    config.evaluation_rows = []


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-generation"):
        for item in items:
            if "generation" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason="Use --run-generation for real API calls")
                )


@pytest.fixture(scope="session")
def rag(tmp_path_factory):
    pipeline, client = build_eval_pipeline(tmp_path_factory.mktemp("rag_eval"))
    yield pipeline
    client.delete_collection(EVAL_COLLECTION)


@pytest.fixture(scope="session")
def generation_rag(rag):
    try:
        attach_llm_client(rag)
    except RuntimeError as exc:
        pytest.fail(str(exc))
    yield rag
    rag.llm_client.close()


@pytest.fixture(scope="session")
def run_judge(request):
    if not request.config.getoption("--run-judge"):
        return False
    missing = missing_env(JUDGE_ENV)
    if missing:
        pytest.fail("Missing judge configuration: " + ", ".join(missing))
    return True


@pytest.fixture
def record_evaluation(request):
    def record(row):
        request.config.evaluation_rows.append(row)
        print(json.dumps(row, indent=2))

    return record


def pytest_sessionfinish(session, exitstatus):
    destination = session.config.getoption("--eval-report")
    if destination:
        report = build_report(
            session.config.evaluation_rows, pytest_exit_status=int(exitstatus)
        )
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def pytest_terminal_summary(terminalreporter):
    rows = [
        row
        for row in terminalreporter.config.evaluation_rows
        if row["stage"] == "retrieval"
    ]
    if rows:
        mean = sum(row["recall_at_k"] for row in rows) / len(rows)
        k_values = sorted({row["k"] for row in rows})
        terminalreporter.write_sep(
            "=",
            f"Retrieval: mean recall at k={k_values} = {mean:.3f} ({len(rows)} cases)",
        )
        summary = rank_summary(rows)
        assert summary is not None
        for stage, metrics in summary.items():
            text = ", ".join(f"{name}={value:.3f}" for name, value in metrics.items())
            terminalreporter.write_line(f"{stage}: {text}")
