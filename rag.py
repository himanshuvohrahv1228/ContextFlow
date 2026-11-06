
import requests


class OllamaClient:
    """Connects ContextFlow to the local Ollama model."""

    def __init__(
        self,
        model="qwen2.5:3b",
        base_url="http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 256
                }
            },
            timeout=180
        )

        response.raise_for_status()

        answer = response.json().get("response", "").strip()

        if not answer:
            raise RuntimeError("Ollama returned an empty response.")

        return answer


class RAGEngine:
    """Retrieves document chunks and generates grounded answers."""

    def __init__(
        self,
        embedding_model,
        vector_store,
        llm_client,
        top_k=8
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k

    def answer(self, question):
        question = (question or "").strip()

        if not question:
            raise ValueError("Please enter a question.")

        # Step 1: Convert the question into an embedding.
        query_vector = self.embedding_model.embed([question])[0]

        # Step 2: Retrieve the closest document chunks.
        matches = self.vector_store.search(
            query_vector,
            top_k=self.top_k
        )

        print("\n--- RETRIEVED DOCUMENT CHUNKS ---")

        for item in matches:
            print("Similarity:", round(item["similarity"], 3))
            print("Source:", item["source"])
            print("Page:", item["page"])
            print("Text:", item["text"])
            print("-------------------------------")

        # Do not reject results using a fixed similarity threshold.
        if not matches:
            return {
                "answer": (
                    "No document chunks were found. "
                    "Please process your PDF again."
                ),
                "sources": []
            }

        # Step 3: Prepare context for the LLM.
        context_parts = []
        sources = []

        for item in matches:
            context_parts.append(
                f"""
Source: {item['source']}
Page: {item['page']}
Content:
{item['text']}
"""
            )

            sources.append({
                "source": item["source"],
                "page": item["page"],
                "preview": item["text"][:220]
            })

        context = "\n".join(context_parts)

        # Step 4: Ask the LLM using retrieved context only.
        prompt = f"""
You are ContextFlow, a document question-answering assistant.

Answer the user's question using the document context below.

Rules:
1. Use only information present in the context.
2. Do not invent or assume facts.
3. For questions asking to list items, include every matching item found in the context.
4. Preserve exact project titles, names, tools and terminology.
5. If the context does not contain the answer, say so clearly.
6. Give a direct and concise answer.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

        answer = self.llm_client.generate(prompt)

        return {
            "answer": answer,
            "sources": sources
        }
