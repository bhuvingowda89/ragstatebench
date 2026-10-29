"""Frozen PA2 ranking built exclusively from materialized replica planes."""
from __future__ import annotations

import os
import time

from ragstate.campaign.phase7a import PromptedEncoder
from .architecture import retrieval_bundle
from .dense import SentenceTransformerEncoder
from .parent_aware import parent_chunk_rrf


class MemoizingEncoder:
    """Reuse text embeddings across replica index rebuilds, never relevance."""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.document_vectors = {}
        self.query_vectors = {}
        self.cache_identity = wrapped.cache_identity

    def encode_documents(self, texts, batch_size):
        missing = list(dict.fromkeys(text for text in texts if text not in self.document_vectors))
        if missing:
            vectors = self.wrapped.encode_documents(missing, batch_size)
            self.document_vectors.update(zip(missing, vectors))
        return [self.document_vectors[text] for text in texts]

    def encode_query(self, texts, batch_size):
        missing = list(dict.fromkeys(text for text in texts if text not in self.query_vectors))
        if missing:
            vectors = self.wrapped.encode_query(missing, batch_size)
            self.query_vectors.update(zip(missing, vectors))
        return [self.query_vectors[text] for text in texts]

    def encode(self, texts, batch_size):
        return self.encode_query(texts, batch_size)


class PA2ReplicaRetriever:
    """PA2's two independent indices receive only replica materializations."""

    def __init__(self, config, encoder=None):
        self.config = config
        if encoder is None:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            import torch
            torch.set_num_threads(1)
            encoder = PromptedEncoder(SentenceTransformerEncoder(
                config['embedding_snapshot'], config['embedding_revision']),
                config['embedding_query_prefix'], config['embedding_passage_prefix'])
        self.encoder = MemoizingEncoder(encoder)
        self.key = None
        self.chunk_index = None
        self.parent_index = None
        self.last_parent_latency_ms = 0.0
        self.last_index_build_ms = 0.0

    def retrieve(self, query_text, chunk_replica, parent_replica):
        key = (chunk_replica.state_hash(), parent_replica.state_hash())
        if self.key != key:
            started = time.perf_counter()
            self.chunk_index = retrieval_bundle(
                chunk_replica.ordered_chunks(), self.encoder,
                self.config['embedding_model'], self.config['embedding_revision'],
                self.config['hybrid'])['hybrid']
            self.parent_index = retrieval_bundle(
                parent_replica.ordered_parent_chunks(), self.encoder,
                self.config['embedding_model'], self.config['embedding_revision'],
                self.config['hybrid'])['hybrid']
            self.last_index_build_ms = (time.perf_counter() - started) * 1000
            self.key = key
        candidates = self.chunk_index.retrieve(query_text, self.config['candidate_k'])
        started = time.perf_counter()
        parents = self.parent_index.retrieve(query_text,
                                             len(parent_replica.documents))
        self.last_parent_latency_ms = (time.perf_counter() - started) * 1000
        ranks = {hit.chunk.logical_document_id: rank
                 for rank, hit in enumerate(parents, 1)}
        # A chunk with no replica-visible parent cannot be fused. This only
        # affects divergent states; clean PA2 ordering remains frozen.
        candidates = [hit for hit in candidates
                      if hit.chunk.logical_document_id in ranks]
        return parent_chunk_rrf(candidates, ranks,
                                self.config['parent_chunk_rrf_k'])[:self.config['top_k']]
