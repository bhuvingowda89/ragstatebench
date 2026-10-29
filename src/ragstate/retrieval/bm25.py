from __future__ import annotations
import math, re
from collections import Counter, defaultdict
from .models import RetrievalHit

TOKEN_RE = re.compile(r"[A-Za-z0-9_$%-]+")

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]

class BM25Retriever:
    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c.text) for c in self.chunks]
        self.doc_lens = [len(x) for x in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        self.df = defaultdict(int)
        for toks in self.doc_tokens:
            for term in set(toks):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if n else 0.0

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        qterms = tokenize(query)
        hits = []
        for chunk, toks, dl in zip(self.chunks, self.doc_tokens, self.doc_lens):
            tf = Counter(toks)
            score = 0.0
            for term in qterms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += self._idf(term) * (f * (self.k1 + 1)) / denom
            if score > 0:
                hits.append(RetrievalHit(chunk=chunk, score=score))
        hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
        return hits[:top_k]
