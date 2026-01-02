import os
import json
import builtins
import pytest

import numpy as np
import openai

from agent_tools import EmbeddingStore


class DummyResp:
    def __init__(self, embedding):
        self.data = [type("x", (), {"embedding": embedding})]


def test_embedding_store_add_and_query(monkeypatch, tmp_path):
    # Mock OpenAI embedding call to return deterministic vectors
    def fake_embed(*args, **kwargs):
        input_text = kwargs.get("input") or (args[1] if len(args) > 1 else "")
        v = np.ones(8) * (len(input_text) % 5 + 1)
        return DummyResp(v.tolist())

    monkeypatch.setattr(openai.Embedding, "create", fake_embed)

    store_path = tmp_path / "store.db"
    store = EmbeddingStore(path=str(store_path), model="test-model")
    store.add_document("doc1", "para one.\n\npara two is here.")

    # query
    results = store.query("question about para")
    assert isinstance(results, list)
    for r in results:
        assert "id" in r and "text" in r and "score" in r


def test_cache_set_get(tmp_path):
    db_path = tmp_path / "store.db"
    store = EmbeddingStore(path=str(db_path), model="test-model")
    key = "abc123"
    assert store.get_cache(key) is None
    store.set_cache(key, "response text")
    assert store.get_cache(key) == "response text"
