import pytest

from src.chunking import chunk_text, chunk_text_recursive


def test_fixed_chunking_uses_overlapping_windows():
    assert chunk_text("abcdefghij", chunk_size=4, overlap=1) == ["abcd", "defg", "ghij"]


def test_fixed_chunking_returns_short_text_as_one_chunk():
    assert chunk_text("abc", chunk_size=10, overlap=2) == ["abc"]


def test_fixed_chunking_skips_whitespace_only_windows():
    assert chunk_text("ab    ", chunk_size=2, overlap=0) == ["ab"]


@pytest.mark.parametrize("chunker", [chunk_text, chunk_text_recursive])
@pytest.mark.parametrize("chunk_size, overlap", [(10, 10), (10, 20), (10, -1)])
def test_invalid_overlap_is_rejected(chunker, chunk_size, overlap):
    with pytest.raises(ValueError):
        chunker("some text", chunk_size=chunk_size, overlap=overlap)


def test_recursive_chunking_splits_on_paragraphs_first():
    assert chunk_text_recursive("para one.\n\npara two.", chunk_size=12, overlap=0) == [
        "para one.", "para two.",
    ]


def test_recursive_chunking_normalizes_windows_newlines():
    chunks = chunk_text_recursive("first\r\n\r\nsecond", chunk_size=20, overlap=0)
    assert chunks == ["first\n\nsecond"]


def test_recursive_chunking_respects_max_size_with_default_settings():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = chunk_text_recursive(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 800 for chunk in chunks)


def test_recursive_chunking_of_empty_text_is_empty():
    assert chunk_text_recursive("") == []
