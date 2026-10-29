"""Focused offline checks for the frozen second-stage ranking boundary."""

from ragstate.retrieval.models import Chunk, RetrievalHit
from ragstate.retrieval.reranker import rerank_candidates


class RecordingScorer:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def score_pairs(self, pairs):
        self.pairs = pairs
        return self.scores


def _hits():
    return [
        RetrievalHit(Chunk(f"c{i}", f"doc{i}", f"logical{i}", 2,
                           "tenant", 3, f"Passage {i}"),
                     0.1 / i, (("bm25", i, 1.0),))
        for i in range(1, 4)
    ]


def test_reranker_only_receives_query_and_candidate_text_and_keeps_provenance():
    hits = _hits()
    scorer = RecordingScorer([0.2, 0.9, 0.3])
    ranked = rerank_candidates("natural query", hits, scorer, 2)
    assert scorer.pairs == [("natural query", "Passage 1"),
                            ("natural query", "Passage 2"),
                            ("natural query", "Passage 3")]
    assert [hit.chunk.chunk_id for hit in ranked] == ["c2", "c3"]
    assert ranked[0].chunk is hits[1].chunk
    assert ranked[0].components == hits[1].components


def test_ties_preserve_frozen_candidate_order_and_repeated_ranking():
    hits = _hits()
    scorer = RecordingScorer([0.5, 0.5, 0.5])
    first = rerank_candidates("q", hits, scorer)
    second = rerank_candidates("q", hits, scorer)
    assert [h.chunk.chunk_id for h in first] == ["c1", "c2", "c3"]
    assert first == second


def test_invalid_scores_do_not_create_rankings():
    import pytest

    with pytest.raises(ValueError, match="score count"):
        rerank_candidates("q", _hits(), RecordingScorer([1.0]))
    with pytest.raises(ValueError, match="Non-finite"):
        rerank_candidates("q", _hits(), RecordingScorer([1.0, float("nan"), 0.0]))


def test_top_k_is_a_subset_of_the_frozen_candidate_pool():
    hits = _hits()
    scorer = RecordingScorer([0.3, 0.9, 0.2])
    ranked = rerank_candidates("q", hits, scorer, 1)
    assert [h.chunk.chunk_id for h in ranked] == ["c2"]
    assert set(h.chunk.chunk_id for h in ranked) <= set(h.chunk.chunk_id for h in hits)
    assert len(scorer.pairs) == len(hits)


def test_local_cross_encoder_rejects_missing_snapshot_without_loading(tmp_path):
    import pytest

    from ragstate.retrieval.reranker import LocalCrossEncoder

    with pytest.raises(ValueError, match="snapshot is unavailable"):
        LocalCrossEncoder(tmp_path / "missing", "missing")
