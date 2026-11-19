"""Lightweight stand-ins for models, Chroma collections, and the LLM client."""

from unittest.mock import MagicMock

import numpy as np

from src.RAGPipeline import RAGPipeline


class FakeEmbeddingModel:
    """Deterministic 3-dimensional embeddings; records every encode() call."""

    def __init__(self, *args, **kwargs):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True):
        self.calls.append(list(texts))
        return np.array([[float(len(text)), 1.0, 0.5] for text in texts])


class FakeReranker:
    """Returns preset scores in input order."""

    def __init__(self, scores=None, *args, **kwargs):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        scores = self.scores if self.scores is not None else [0.0] * len(pairs)
        return np.array(scores[: len(pairs)], dtype=float)


class FakeCollection:
    """Minimal collection with canned get()/query() results; records the arguments."""

    def __init__(self, get_result=None, query_result=None, records=None):
        self.get_result = get_result or {"ids": [], "documents": [], "metadatas": []}
        self.query_result = query_result or {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]],
        }
        self.records = records or {}
        self.get_calls = []
        self.query_calls = []

    def get(self, ids=None, **kwargs):
        self.get_calls.append({"ids": ids, **kwargs})
        if ids is not None:
            found = [chunk_id for chunk_id in ids if chunk_id in self.records]
            return {
                "ids": found,
                "documents": [self.records[chunk_id]["document"] for chunk_id in found],
                "metadatas": [self.records[chunk_id].get("metadata", {}) for chunk_id in found],
            }
        return self.get_result

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return self.query_result


def make_llm_client(output_text="answer [S1]"):
    client = MagicMock()
    client.responses.create.return_value.output_text = output_text
    return client


def make_pipeline(**parts):
    """RAGPipeline without __init__ (no models, DB, or API client), with given attributes."""
    pipeline = RAGPipeline.__new__(RAGPipeline)
    for name, value in parts.items():
        setattr(pipeline, name, value)
    return pipeline
