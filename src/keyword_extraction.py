import re
from functools import cache

import spacy  # pyright: ignore[reportMissingImports]


@cache
def _nlp():
    """Load the spaCy model on first use rather than at import time."""
    return spacy.load("en_core_web_sm")


def extract_keywords(question: str) -> list[str]:
    """Entities, noun phrases, noun/number lemmas, and policy IDs, deduplicated."""
    doc = _nlp()(question)
    keywords = []

    for ent in doc.ents:
        keywords.append(ent.text.lower())

    for chunk in doc.noun_chunks:
        phrase = " ".join(
            token.text.lower()
            for token in chunk
            if not token.is_stop and not token.is_punct
        )
        if phrase:
            keywords.append(phrase)

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "NUM"}:
            keywords.append(token.lemma_.lower())

    policy_ids = re.findall(r"\b[A-Z]{2,5}-\d+\b", question)
    keywords.extend(policy_ids)

    return list(dict.fromkeys(keywords))
