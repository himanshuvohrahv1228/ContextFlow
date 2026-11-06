# ContextFlow – AI-Powered Document Intelligence Assistant

ContextFlow is a beginner-friendly Retrieval-Augmented Generation (RAG) project. It lets a user upload PDF documents, search their content using semantic similarity, and ask a local language model questions with source/page references.

## Features

- Upload one or multiple PDF files
- Extract PDF text page by page
- Split text into overlapping chunks
- Create local text embeddings with Sentence Transformers
- Store and search vectors with ChromaDB
- Retrieve relevant passages before generating an answer
- Generate answers with a local Ollama model
- Show source PDF names and page numbers
- Save question-answer history in SQLite
- Unit tests for chunking and the RAG flow

## Technology

Python, Streamlit, PyPDF, Sentence Transformers, ChromaDB, Ollama, SQLite, pytest.

## Requirements

- Python 3.10 or newer
- Ollama installed
- Internet connection for the first download of the embedding model and Ollama model
- Recommended: 8 GB RAM or more. A 3B model may be slow on CPU-only laptops.

## Setup on Windows (PowerShell)

Open PowerShell in the ContextFlow project folder.

### 1. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install Python packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install and start Ollama

Install Ollama from https://ollama.com/download/windows

Then, in PowerShell:

```powershell
ollama pull qwen2.5:3b
ollama serve
```

Keep the Ollama terminal open. If Ollama is already running in the background, `ollama serve` may say the port is in use; that is okay.

### 4. Run ContextFlow

Open a second PowerShell terminal, activate the virtual environment, go to the project folder, and run:

```powershell
streamlit run app.py
```

The app opens in your browser, usually at http://localhost:8501.

On the first run, Sentence Transformers downloads `all-MiniLM-L6-v2`. This is a one-time model download. After it is cached, embeddings run locally.

## Run tests

```powershell
python -m pytest -q
```

The tests use fake embedding, vector-store and LLM objects, so they do not require Ollama or a downloaded embedding model.

## How it works

1. **Upload:** Streamlit receives PDF files.
2. **Extract:** PyPDF extracts text one page at a time.
3. **Chunk:** Text is split into 800-character chunks with 150-character overlap.
4. **Embed:** Sentence Transformers converts each chunk into a numeric vector.
5. **Store:** ChromaDB saves vectors and metadata (file name and page number).
6. **Retrieve:** A question is embedded and ChromaDB finds the four closest chunks.
7. **Generate:** ContextFlow sends only the retrieved passages and the question to Ollama.
8. **Cite:** The UI shows the source file and page for retrieved passages.

## Important limitations

- Scanned/image-only PDFs need OCR; this version extracts selectable text.
- Answers depend on the quality of the uploaded documents and local language model.
- Similarity is a retrieval signal, not a guarantee that a passage answers the question.
- Documents and chat history stay in the local project folder. Do not upload confidential documents unless you are allowed to process them.

## Project structure

```text
ContextFlow/
├── app.py
├── document_processor.py
├── embeddings.py
├── vector_store.py
├── rag.py
├── chat_history.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── uploads/
└── tests/
    └── test_rag.py
```
