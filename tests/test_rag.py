from document_processor import split_text
from rag import RAGEngine


def test_split_text_returns_overlapping_chunks():
    text = "A" * 1900
    chunks = split_text(text, chunk_size=800, overlap=100)
    assert len(chunks) == 3
    assert chunks[0][-100:] == chunks[1][:100]


def test_split_text_handles_empty_text():
    assert split_text("   ") == []


class FakeEmbeddingModel:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    def search(self, query_embedding, top_k=4):
        return [{
            "text": "The policy allows 15 days of annual leave.",
            "source": "policy.pdf",
            "page": 2,
            "similarity": 0.85
        }]


class FakeLLM:
    def generate(self, prompt):
        assert "15 days of annual leave" in prompt
        return "The policy allows 15 days of annual leave (policy.pdf, page 2)."


def test_rag_engine_uses_retrieved_context():
    engine = RAGEngine(FakeEmbeddingModel(), FakeVectorStore(), FakeLLM())
    result = engine.answer("How many annual leave days are allowed?")
    assert "15 days" in result["answer"]
    assert result["sources"][0]["source"] == "policy.pdf"


def test_rag_engine_does_not_call_llm_when_no_relevant_context():
    class EmptyStore:
        def search(self, query_embedding, top_k=4):
            return []

    class ShouldNotRunLLM:
        def generate(self, prompt):
            raise AssertionError("LLM should not be called without context")

    engine = RAGEngine(FakeEmbeddingModel(), EmptyStore(), ShouldNotRunLLM())
    result = engine.answer("Question outside the documents")
    assert result["sources"] == []
    assert "could not find relevant information" in result["answer"]
