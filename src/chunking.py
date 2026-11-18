from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=500, overlap=50):
    """fixed chunking"""
    if not 0 <= overlap < chunk_size:
        raise ValueError("0 <= overlap < chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_text_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split on paragraph boundaries first; sizes and target overlap are characters."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("Expected 0 <= overlap < chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return splitter.split_text(text)
