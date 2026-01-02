import os
import glob
import json
import sqlite3
import time
import math

import openai
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

from utils import retry_openai


class EmbeddingStore:
    """SQLite-backed embedding store with optional Faiss index and a simple cache for chat responses.

    Tables:
      docs(id TEXT PRIMARY KEY, chunk_index INTEGER, text TEXT, embedding TEXT)
      cache(key TEXT PRIMARY KEY, response TEXT, created_at REAL)
    """

    def __init__(self, path="store.db", model="text-embedding-3-small"):
        self.path = path
        self.model = model
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY, chunk_index INTEGER, text TEXT, embedding TEXT)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_docs_id ON docs(id)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, response TEXT, created_at REAL)"
        )
        self._conn.commit()
        self._faiss_index = None
        self._faiss_dim = None

    def ingest_dir(self, dir_path):
        md_files = glob.glob(os.path.join(dir_path, "*.md"))
        for fp in md_files:
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()
            self.add_document(os.path.basename(fp), text)
        # build Faiss index if available
        if _HAS_FAISS:
            self._build_faiss()

    def add_document(self, doc_id, text):
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        cur = self._conn.cursor()
        for i, p in enumerate(parts):
            eid = f"{doc_id}::{i}"
            emb = self._embed(p)
            emb_json = json.dumps(emb.tolist())
            cur.execute(
                "INSERT OR REPLACE INTO docs(id, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
                (eid, i, p, emb_json),
            )
        self._conn.commit()

    def _embed(self, text):
        def _call():
            return openai.Embedding.create(model=self.model, input=text)

        resp = retry_openai(_call)
        return np.array(resp.data[0].embedding, dtype=float)

    def _build_faiss(self):
        if not _HAS_FAISS:
            return
        cur = self._conn.cursor()
        rows = cur.execute("SELECT id, embedding FROM docs ORDER BY id").fetchall()
        if not rows:
            return
        embeddings = np.array([json.loads(r[1]) for r in rows], dtype='float32')
        dim = embeddings.shape[1]
        self._faiss_dim = dim
        index = faiss.IndexFlatIP(dim)
        # normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings_norm = embeddings / norms
        index.add(embeddings_norm)
        self._faiss_index = (index, [r[0] for r in rows])

    def query(self, text, top_k=3):
        q = self._embed(text)
        cur = self._conn.cursor()
        if _HAS_FAISS:
            if not self._faiss_index:
                self._build_faiss()
            if self._faiss_index:
                index, ids = self._faiss_index
                q32 = np.array(q, dtype='float32')
                q_norm = q32 / (np.linalg.norm(q32) if np.linalg.norm(q32) != 0 else 1.0)
                sims, idx = index.search(np.expand_dims(q_norm, axis=0), top_k)
                results = []
                for score, i in zip(sims[0], idx[0]):
                    if i < 0:
                        continue
                    doc_id = ids[int(i)]
                    row = cur.execute("SELECT text FROM docs WHERE id = ?", (doc_id,)).fetchone()
                    text = row[0] if row else ""
                    results.append({"id": doc_id, "text": text, "score": float(score)})
                return results
        # fallback: brute-force using DB embeddings
        rows = cur.execute("SELECT id, text, embedding FROM docs").fetchall()
        if not rows:
            return []
        embeddings = np.array([json.loads(r[2]) for r in rows], dtype=float)
        emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        q_norm = q / np.linalg.norm(q)
        sims = emb_norm.dot(q_norm)
        idxs = np.argsort(-sims)[:top_k]
        results = []
        for i in idxs:
            results.append({"id": rows[int(i)][0], "text": rows[int(i)][1], "score": float(sims[int(i)])})
        return results

    # Simple cache stored in SQLite
    def get_cache(self, key):
        cur = self._conn.cursor()
        row = cur.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_cache(self, key, response):
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO cache(key, response, created_at) VALUES (?, ?, ?)",
            (key, response, time.time()),
        )
        self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


