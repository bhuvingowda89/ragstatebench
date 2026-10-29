"""CPU dense retrieval with explicit, content-addressed embedding caches."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .models import RetrievalHit


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, model_version: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install ragstatebench[dense] for dense retrieval") from exc
        self.model = SentenceTransformer(
            model_name, revision=model_version, device="cpu",
            local_files_only=Path(model_name).is_dir(),
        )

    def encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                                 convert_to_numpy=True).tolist()


class DenseRetriever:
    def __init__(self, chunks, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 *, model_version: str = "unspecified", encoder=None,
                 cache_dir: str | Path | None = None, batch_size: int = 32):
        self.chunks = list(chunks)
        self.model_name = model_name
        self.model_version = model_version
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        if encoder is None and model_version == "unspecified":
            raise ValueError("Pin embedding model revision for reproducible dense retrieval")
        self.encoder = encoder if encoder is not None else SentenceTransformerEncoder(model_name, model_version)
        identity = [(c.chunk_id, c.text) for c in self.chunks]
        key = hashlib.sha256(json.dumps([model_name, model_version,
                                         getattr(self.encoder, "cache_identity", None), identity],
                                     ensure_ascii=False).encode()).hexdigest()
        self.cache_path = Path(cache_dir) / f"{key}.json" if cache_dir is not None else None
        if self.cache_path and self.cache_path.exists():
            self.vectors = json.loads(self.cache_path.read_text())["vectors"]
        else:
            self.vectors = []
            encode_documents = getattr(self.encoder, "encode_documents", self.encoder.encode)
            for i in range(0, len(self.chunks), batch_size):
                self.vectors.extend(encode_documents(
                    [c.text for c in self.chunks[i:i + batch_size]], batch_size=batch_size))
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"model_name": model_name,
                                           "model_version": model_version,
                                           "vectors": self.vectors}))
                tmp.replace(self.cache_path)
        if len(self.vectors) != len(self.chunks):
            raise ValueError("Encoder returned the wrong number of embeddings")

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        if top_k <= 0 or not self.chunks:
            return []
        encode_query = getattr(self.encoder, "encode_query", self.encoder.encode)
        q = encode_query([query], batch_size=1)[0]
        def cosine(v):
            if len(q) != len(v):
                raise ValueError("Query and document embedding dimensions differ")
            denom = math.sqrt(sum(x*x for x in q) * sum(x*x for x in v))
            return sum(a*b for a, b in zip(q, v)) / denom if denom else 0.0
        hits = [RetrievalHit(c, cosine(v)) for c, v in zip(self.chunks, self.vectors)]
        return sorted(hits, key=lambda h: (-h.score, h.chunk.chunk_id))[:top_k]
