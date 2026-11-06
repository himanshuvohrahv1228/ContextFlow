
from pathlib import Path
from pypdf import PdfReader


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150):
    """Split text while preserving line and paragraph boundaries."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid overlap value")

    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]

    if not lines:
        return []

    chunks = []
    current_chunk = []

    for line in lines:
        candidate = "\n".join(current_chunk + [line])

        if len(candidate) <= chunk_size:
            current_chunk.append(line)
        else:
            if current_chunk:
                chunks.append("\n".join(current_chunk))

            # Keep previous lines as context for the next chunk.
            overlap_lines = []
            overlap_length = 0

            for previous_line in reversed(current_chunk):
                if overlap_length + len(previous_line) > overlap:
                    break
                overlap_lines.insert(0, previous_line)
                overlap_length += len(previous_line)

            current_chunk = overlap_lines + [line]

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def process_pdf(pdf_path: str, chunk_size: int = 1000, overlap: int = 150):
    """Extract text page by page and attach source metadata."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))
    output = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        page_chunks = split_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            output.append({
                "text": chunk,
                "source": path.name,
                "page": page_number,
                "chunk": chunk_number
            })

    return output
