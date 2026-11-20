import pytest
from evaluation import evaluate_generation, load_cases


@pytest.mark.generation
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["id"])
def test_generated_answer(generation_rag, run_judge, case, record_evaluation):
    result = evaluate_generation(generation_rag, case, run_judge=run_judge)
    record_evaluation(result)
    problems = {
        key: result[key]
        for key in ("missing_facts", "invalid_citations", "missing_evidence_citations")
        if result[key]
    }
    assert not problems, f"Answer evaluation failed: {problems}"
