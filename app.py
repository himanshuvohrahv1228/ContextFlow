import streamlit as st
from pathlib import Path

from document_processor import process_pdf
from embeddings import EmbeddingModel
from vector_store import VectorStore
from rag import RAGEngine, OllamaClient
from chat_history import ChatHistory

st.set_page_config(page_title="ContextFlow", page_icon="📚", layout="wide")

st.title("📚 ContextFlow")
st.caption("AI-Powered Document Intelligence Assistant")
st.write("Upload PDFs, ask questions, and get answers grounded in your documents with page references.")

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_embedding_model():
    return EmbeddingModel()

@st.cache_resource
def load_vector_store():
    return VectorStore(persist_directory=str(CHROMA_DIR))

@st.cache_resource
def load_ollama_client():
    return OllamaClient()

@st.cache_resource
def load_chat_history():
    return ChatHistory(str(DATA_DIR / "chat_history.db"))

def clear_chat():
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("1. Add documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload more than one PDF."
    )

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Reading PDFs and creating searchable chunks..."):
            try:
                embedding_model = load_embedding_model()
                vector_store = load_vector_store()
                total_chunks = 0
                for uploaded in uploaded_files:
                    safe_name = Path(uploaded.name).name
                    destination = UPLOAD_DIR / safe_name
                    destination.write_bytes(uploaded.getvalue())
                    chunks = process_pdf(str(destination))
                    if chunks:
                        vectors = embedding_model.embed([item["text"] for item in chunks])
                        vector_store.add_documents(chunks, vectors)
                        total_chunks += len(chunks)
                if total_chunks:
                    st.success(f"Processed {len(uploaded_files)} file(s) and indexed {total_chunks} text chunks.")
                    st.write("Total chunks stored in ChromaDB:", vector_store.count())
                else:
                    st.warning("No readable text was found. If this is a scanned PDF, OCR is needed.")
            except Exception as exc:
                st.error(f"Could not process the PDFs: {exc}")

    st.divider()
    st.header("2. Local AI setup")
    st.write("ContextFlow uses Ollama running on your computer.")
    st.code("ollama pull qwen2.5:3b\nollama serve", language="bash")
    st.caption("Keep Ollama running while asking questions.")
    if st.button("Clear chat"):
        clear_chat()
        st.rerun()

    st.divider()
    st.caption("Your PDFs, embeddings and chat database are stored locally in the project folder.")

try:
    embedding_model = load_embedding_model()
    vector_store = load_vector_store()
    ollama_client = load_ollama_client()
    history_db = load_chat_history()
    engine = RAGEngine(embedding_model, vector_store, ollama_client)
except Exception as exc:
    st.error(f"Startup issue: {exc}")
    st.info("Install the requirements, download the embedding model on first run, and start Ollama.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your uploaded PDFs...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents and preparing an answer..."):
            try:
                result = engine.answer(question)
                answer = result["answer"]
                st.markdown(answer)
                if result["sources"]:
                    with st.expander("Sources used"):
                        for source in result["sources"]:
                            st.write(f"**{source['source']} — page {source['page']}**")
                            st.caption(source["preview"])
                st.session_state.messages.append({"role": "assistant", "content": answer})
                history_db.save(question, answer)
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
                st.info("Check that Ollama is running and that qwen2.5:3b is downloaded.")
