# RAG evaluation tests

Run from the project root. This suite contains 10 evaluation questions, each with a manually specified expected answer, evidence chunk IDs, evidence quotes, and answer checks in `evaluation_cases.json`.

## Install and run

`uv sync` installs the application and the `dev` dependency group (pytest, pytest-cov, ruff) into `.venv`.

```bash
uv sync
.venv/bin/python -m pytest test -q --eval-report test/results-retrieval.json
```

Default execution runs real ingestion, embedding, hybrid retrieval, and reranking. The 10 generation cases are skipped. No Foundry API requests are made. A temporary Chroma database is populated from `docs`; the existing `chroma_data` directory is not used.

To prevent model downloads when both models are already cached:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest test -q --eval-report test/results-retrieval.json
```

Run real generation separately (10 API calls, which may incur costs):

```bash
.venv/bin/python -m pytest test -m generation --run-generation -q --eval-report test/results-generation.json
```

The LLM decision judge is off by default. Add `--run-judge` to call it; this also requires `FOUNDRY_ENDPOINT`, `FOUNDRY_DECISION_DEPLOYMENT`, and `FOUNDRY_API_KEY`, and fails before any test if they are missing. It calls `Microsoft-Decision-1.1` through `<resource-endpoint>/providers/microsoft/v1/systemone` ([docs](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision)); deploy that model separately and copy the resource endpoint from its deployment page (not the `/openai/v1` base URL used for generation). Judge verdicts are counted in the report but are not asserted.

Generation uses the application's `.env` loading and requires `FOUNDRY_BASE_URL`, `FOUNDRY_API_KEY`, and `FOUNDRY_DEPLOYMENT_NAME`. Missing configuration fails explicitly. Credentials are not included in reports. Reports do contain questions, source IDs, and generated answers.

Add `-s` to display every evaluation record. Use `-k hotel_conflict` to run one question, or `-m retrieval` to select only retrieval tests.

## What is measured

- **Candidate recall:** fraction of the manually specified gold chunks present in the output of the production `retrieve()` method.
- **Recall@k:** fraction of gold chunks present in the first k results after production `rerank()`. The current pytest cases use k=5. Each question targets 1.0; failing questions are useful evidence of retrieval gaps. The JSON report includes per-question results and their unweighted mean. For a question with one gold chunk this is equivalent to hit@k; conflict questions require both sources.
- **Rank metrics before vs. after reranking:** recall@1/3/5 and MRR (1/rank of the first gold chunk) for both the `retrieve()` order (vector hits by distance, then keyword-only hits) and the `rerank()` order, plus each gold chunk's rank and whether it came from vector or keyword search. Recall@5 alone cannot show the reranker's effect, because the vector search returns only 5 results; identical pre/post numbers mean reranking did not move any gold chunk.
- **Generated answer checks:** call production `generate()` with fixed gold evidence, then check required facts using regex and check that expected evidence labels are cited without nonexistent labels.

Generation is evaluated with gold evidence to separate generation quality from retrieval failures. This is **not an end-to-end evaluation** of `ask()`. Also, `ask()` currently sends all ranked candidates, whereas the retrieval metric deliberately measures only the top k.

Regex checks are a basic, auditable keyword rubric, not a semantic judge. They can miss valid paraphrases or pass an answer that mentions the expected numbers but assigns them incorrectly. Citation checks verify label presence, not whether each claim is supported. Review the saved answers for factual accuracy, source attribution, contradiction handling, and unsupported claims before submission. A passing automated check alone does not establish answer correctness.

## Dataset and reproducibility

Questions cover the current equipment allowance, eligibility, submission deadline, approvals, receipt requirements, ownership/refunds, historical allowance, Finance review time, and conflicting hotel/meal limits. Both conflicting sources have explicit gold IDs; neither source is silently treated as authoritative.

Gold chunk IDs are frozen against the current documents and the application's LangChain recursive splitter with an 800-character limit and a target overlap of 50 characters (actual overlap depends on boundaries). The limit was raised from 500 after `test/sweep_chunking.py` compared 300/500/800/1200; gold IDs were re-mapped by locating each evidence quote in the 800/50 chunks of the source text, not from retrieval results. Evidence quotes are checked before evaluation to detect shifted boundaries. If documents or chunking change, review and update the gold evidence manually; do not replace expected IDs with whichever results the retriever happens to return. Header chunks supply document dates in generation context but are not part of retrieval recall targets. The dataset is small and curated, not exhaustive relevance labeling.

Tests use CPU inference and a fresh database, avoiding contamination from previous demo records. This improves repeatability, but does not guarantee bit-for-bit determinism across hardware, model revisions, dependency versions, or approximate index construction. Keep the same model cache and dependency lockfile when comparing runs. Generation remains nondeterministic even with fixed context; no exact-string answer assertion is used.

The fixture bypasses `RAGPipeline.__init__` because it eagerly creates an API client. It injects real models and an isolated collection, then uses the original `ingest`, `retrieve`, `rerank`, and `generate` methods. Those methods are not mocked. This does not test the constructor or production database setup.

The embedding model is shared with the application through `EMBEDDING_MODEL_NAME`: `BAAI/bge-m3` (8192-token input limit, 1024-dimensional dense vectors). The application uses `policy_chunks_bge_m3_v1`, a separate collection from the old MiniLM vectors. Re-ingest documents when migrating; old vectors cannot be reused. The original MiniLM collection is retained. The reranker is shared through `RERANKER_MODEL_NAME`: `cross-encoder/ettin-reranker-68m-v1` (English, Apache 2.0, ~270MB). On these 10 cases it gave post-rerank recall@1 0.60 / MRR 0.850, versus 0.50 / 0.783 for `BAAI/bge-reranker-v2-m3` and 0.50 / 0.800 for `ettin-reranker-32m-v1`; with so few cases these differences are one or two gold positions. The previous `cross-encoder/ms-marco-MiniLM-L6-v2` returned NaN scores for every pair with the installed torch/transformers versions, so reranking silently kept the input order; results produced with it reflect no reranking.

The production `retrieve()` currently uses the fixed keywords `monitor` and `equipment` for every question. These tests preserve that behavior; they do not substitute an improved search implementation. They also do not assert that hybrid search must outperform vector-only search: that assignment deliverable needs a separate measured comparison using the same corpus and result limit.

## Running without pytest

`test/run_eval.py` runs the same evaluation and prints per-case tables plus a summary (recall@k, candidate recall, post-rerank recall@1/3/5, MRR before→after rerank, gold chunk ranks, generation checks, judge verdicts):

```bash
python test/run_eval.py                          # retrieval only, no API calls
python test/run_eval.py --generation --judge     # + generation and decision judge (20 API calls)
python test/run_eval.py --case hotel_conflict --k 3 --report test/results-all.json
```

It uses the same isolated temporary database and helpers as pytest (`build_eval_pipeline`, `evaluate_retrieval`, `evaluate_generation`, `build_report` in `test/evaluation.py`), but does not assert pass/fail.
