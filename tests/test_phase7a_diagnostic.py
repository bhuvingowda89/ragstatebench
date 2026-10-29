"""Small offline checks for the model-specific encoding boundary."""

from ragstate.campaign.phase7a import PromptedEncoder
from ragstate.retrieval.chunking import chunk_document
from ragstate.retrieval.dense import DenseRetriever
from ragstate.world.models import DocumentVersion


class RecordingEncoder:
    def __init__(self):
        self.inputs = []

    def encode(self, texts, batch_size):
        self.inputs.append(list(texts))
        return [[1.0, 0.0] if "first" in text else [0.0, 1.0] for text in texts]


def test_model_prefixes_reach_distinct_document_and_query_paths():
    base = RecordingEncoder()
    prompted = PromptedEncoder(base, "query: ", "passage: ")
    chunks = chunk_document(DocumentVersion("first-v1", "first", "t", 1, 0,
                                            content="first record"))
    retriever = DenseRetriever(chunks, model_name="local-test", model_version="fixed",
                               encoder=prompted)
    assert retriever.retrieve("first", 1)[0].chunk.document_id == "first-v1"
    assert base.inputs == [["passage: first record"], ["query: first"]]
    assert prompted.cache_identity == ("query: ", "passage: ")
