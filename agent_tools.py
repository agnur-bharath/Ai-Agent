import os
import glob
import json
import openai
import numpy as np


class EmbeddingStore:
    """A tiny, dependency-light embedding store that uses OpenAI embeddings.

    It stores small text chunks with their embeddings in a JSON file (`store.json` by default).
    Query uses cosine similarity computed with NumPy.
    """

    def __init__(self, path="store.json", model="text-embedding-3-small"):
        self.path = path
        self.model = model
        self.items = []  # list of dicts: {id, text, embedding}
        self.embeddings = None

    def ingest_dir(self, dir_path):
        md_files = glob.glob(os.path.join(dir_path, "*.md"))
        for fp in md_files:
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()
            self.add_document(os.path.basename(fp), text)

    def add_document(self, doc_id, text):
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, p in enumerate(parts):
            eid = f"{doc_id}::{i}"
            emb = self._embed(p)
            self.items.append({"id": eid, "text": p, "embedding": emb.tolist()})
        self._rebuild_embeddings()

    def _embed(self, text):
        resp = openai.Embedding.create(model=self.model, input=text)
        return np.array(resp.data[0].embedding, dtype=float)

    def _rebuild_embeddings(self):
        if self.items:
            self.embeddings = np.array([item["embedding"] for item in self.items])
        else:
            self.embeddings = None

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.items = json.load(f)
            self._rebuild_embeddings()

    def query(self, text, top_k=3):
        q = self._embed(text)
        if self.embeddings is None or len(self.items) == 0:
            return []
        # cosine similarity
        emb = self.embeddings
        emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        q_norm = q / np.linalg.norm(q)
        sims = emb_norm.dot(q_norm)
        idx = np.argsort(-sims)[:top_k]
        return [{"id": self.items[int(i)]["id"], "text": self.items[int(i)]["text"], "score": float(sims[int(i)])} for i in idx]
