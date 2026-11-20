# RAG-System

A retrieval-augmented question-answering assistant for company policy documents.
It indexes the `.txt`/`.md` files in `docs/` into a local Chroma database and
answers questions with source citations (`[S1]`, `[S2]`, ...) using a model
deployed in Microsoft Foundry.

## How it works

```
question ─┬─ vector search (BAAI/bge-m3, top 5) ──┐
          └─ keyword match (spaCy keywords) ──────┴─ merge ─ rerank ─ generate ─ answer [S1]
```

1. **Ingest** (`RAGPipeline.ingest`): documents are split on paragraph boundaries
   (`chunk_text_recursive`, 800 characters, 50 overlap), embedded, and upserted
   with IDs like `policy_v2.txt:0003`. Each chunk stores its document name, chunk
   index, title, and file dates. Re-ingesting a file removes chunks that no longer exist.
2. **Retrieve** (`retrieve`): candidates are the nearest chunks by embedding plus
   chunks containing a keyword extracted from the question (entities, noun phrases,
   and policy IDs such as `FIN-118`, matched case-insensitively). Each candidate
   records whether it came from vector search, keyword match, or both.
3. **Rerank** (`rerank`): a cross-encoder (`cross-encoder/ettin-reranker-68m-v1`)
   scores each (question, chunk) pair and orders the candidates.
4. **Generate** (`generate`): the ranked chunks are sent to the LLM as labelled
   sources with their document name and created date. The model must answer only
   from the sources, cite them, and report conflicts between policy versions.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create `.env` from the template and fill in the Foundry settings:

```bash
cp .env.example .env
```

`FOUNDRY_BASE_URL`, `FOUNDRY_API_KEY`, and `FOUNDRY_DEPLOYMENT_NAME` are required.
`FOUNDRY_ENDPOINT` and `FOUNDRY_DECISION_DEPLOYMENT` are only needed for the LLM
judge in the evaluation.

The embedding and reranker models are downloaded from Hugging Face on first use.

## Usage

```bash
uv run python main.py --ingest   # build ./chroma_data from docs/
uv run python main.py            # ask a question
```

### With Docker

Only Docker and the `.env` file above are needed.

```bash
docker compose build
docker compose run --rm rag --ingest   # build the vector database from docs/
docker compose run --rm rag            # ask a question
```

The first `--ingest` downloads the embedding and reranker models (about 2.5 GB)
into the `model-cache` volume; later runs reuse them. The database is kept in the
`chroma-data` volume, and `docs/` is mounted from the host, so after editing a
document, run `--ingest` again without rebuilding. `docker compose down -v`
removes both volumes.

Inside the container (Linux), files have no creation time, so chunks get no
`created_at` metadata and the prompt shows `Created: Not specified`; dates written
in the documents themselves are unaffected.

## Project layout

```
main.py                    CLI entry point
src/rag_pipeline.py        RAGPipeline: ingest, retrieve, rerank, generate
src/chunking.py            fixed-size and paragraph-aware chunking
src/keyword_extraction.py  spaCy keyword and policy-ID extraction
docs/                      source policy documents
test/unit/                 fast unit tests (no models, network, or API calls)
test/                      retrieval/generation evaluation on 10 labelled questions
```

## Tests and evaluation

```bash
uv run pytest test/unit                # unit tests
uv run python test/run_eval.py         # retrieval evaluation (real models, no API calls)
uv run python test/run_eval.py --generation --judge   # + answer generation and LLM judge
```

CI runs the full pytest suite with coverage and the retrieval evaluation on every
push and pull request. See [test/README.md](test/README.md) for the evaluation
dataset, metrics, options, and the chunk-size sweep (`test/sweep_chunking.py`).
