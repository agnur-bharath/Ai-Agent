import os
import json
import builtins
import pytest

import numpy as np

from agent_tools import EmbeddingStore


class DummyResp:
    def __init__(self, embedding):
        self.data = [type("x", (), {"embedding": embedding})]


def test_embedding_store_add_and_query(monkeypatch, tmp_path):
    # Mock OpenAI embedding call to return deterministic vectors
    def fake_embed(model, input):
        # produce a vector based on length to be deterministic
        v = np.ones(8) * (len(input) % 5 + 1)
        return DummyResp(v.tolist())

    monkeypatch.setattr("openai.Embedding.create", fake_embed)

    store_path = tmp_path / "store.json"
    store = EmbeddingStore(path=str(store_path), model="test-model")
    store.add_document("doc1", "para one.\n\npara two is here.")
    store.save()

    # reload and query
    store2 = EmbeddingStore(path=str(store_path), model="test-model")
    store2.load()
    results = store2.query("question about para")
    assert len(results) == 1 or len(results) == 3
    # ensure each result has expected keys
    for r in results:
        assert "id" in r and "text" in r and "score" in r
