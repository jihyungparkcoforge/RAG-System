def chunk_text(text, chunk_size=500, overlap=50):
    """fixed chunking"""
    if not 0 <= overlap < chunk_size:
        raise ValueError("0 <= overlap < chunk_size여야 합니다.")

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
