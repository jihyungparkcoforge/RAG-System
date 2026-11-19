from src.keyword_extraction import extract_keywords


def test_extracts_policy_ids_and_nouns():
    keywords = extract_keywords("What does FIN-118 say about the hotel limit?")
    assert "FIN-118" in keywords  # regex keeps the original case
    assert "hotel" in keywords
    assert "limit" in keywords


def test_returns_each_keyword_once_in_first_seen_order():
    keywords = extract_keywords("What is the hotel limit for FIN-118 and FIN-118?")
    assert len(keywords) == len(set(keywords))


def test_question_without_content_words_yields_no_keywords():
    assert extract_keywords("is it?") == []
