import pytest
from evaluation import evaluate_retrieval, load_cases


@pytest.mark.retrieval
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["id"])
def test_retrieval_recall(rag, case, record_evaluation):
    result = evaluate_retrieval(rag, case, k=5)
    record_evaluation(result)
    # An explicit target, not a claim that the current implementation achieves it.
    assert result["recall_at_k"] == 1.0, (
        f"Missing gold evidence in top {result['k']}: {result['missing_ids']}; "
        f"candidate recall={result['candidate_recall']:.2f}"
    )
