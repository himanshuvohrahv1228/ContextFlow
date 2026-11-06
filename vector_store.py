import hashlib
import chromadb


class VectorStore:
    """Stores document chunks and searches them by vector similarity."""

    def __init__(self, persist_directory: str = "data/chroma"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="contextflow_documents",
            metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _make_id(item):
        raw = f"{item['source']}|{item['page']}|{item['chunk']}|{item['text']}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def add_documents(self, documents, embeddings):
        if not documents:
            return
        if len(documents) != len(embeddings):
            raise ValueError("Each document must have exactly one embedding")

        self.collection.upsert(
            ids=[self._make_id(item) for item in documents],
            documents=[item["text"] for item in documents],
            embeddings=embeddings,
            metadatas=[
                {
                    "source": item["source"],
                    "page": int(item["page"]),
                    "chunk": int(item["chunk"]),
                }
                for item in documents
            ]
        )

    def search(self, query_embedding, top_k: int = 4):
        if self.collection.count() == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        matches = []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            matches.append({
                "text": text,
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "?"),
                "similarity": max(0.0, 1.0 - float(distance)),
            })
        return matches

    def count(self):
        return self.collection.count()
