from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Converts text into numeric vectors using a small local model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return vectors.tolist()
