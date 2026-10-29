"""Zero-shot, local-only cross-encoder reranking of frozen candidate chunks."""
from __future__ import annotations

import math
from pathlib import Path

from .models import RetrievalHit


class LocalCrossEncoder:
    def __init__(self,snapshot_path,revision):
        path=Path(snapshot_path)
        if not path.is_dir() or path.name!=revision:
            raise ValueError('Pinned local reranker snapshot is unavailable')
        from sentence_transformers import CrossEncoder
        self.model=CrossEncoder(str(path),revision=revision,device='cpu',
                                local_files_only=True)
        self.snapshot_path=str(path)
        self.revision=revision

    def score_pairs(self,pairs):
        scores=self.model.predict(pairs,batch_size=16,show_progress_bar=False,
                                  convert_to_numpy=True,device='cpu')
        if len(scores)!=len(pairs):raise ValueError('Reranker score count differs from candidates')
        return [float(score) for score in scores]


def rerank_candidates(query,candidates,scorer,top_k=10):
    """Send only query and candidate text to the scorer; preserve provenance."""
    if top_k<=0:return []
    pairs=[(query,hit.chunk.text) for hit in candidates]
    scores=scorer.score_pairs(pairs)
    if len(scores)!=len(candidates):raise ValueError('Reranker score count mismatch')
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError('Non-finite reranker score')
    ranked=sorted(enumerate(zip(candidates,scores)),
                  key=lambda item:(-float(item[1][1]),item[0],item[1][0].chunk.chunk_id))
    return [RetrievalHit(hit.chunk,float(score),hit.components)
            for _,(hit,score) in ranked[:top_k]]
