"""Named entity extraction filtered to a fixed set of useful entity types."""
from src.processors.base import NLPProcessor
import spacy

_ALLOWED_TYPES = frozenset({"PERSON", "ORG", "GPE", "DATE", "PRODUCT"})


class EntityExtractor(NLPProcessor):
    name = "entities"

    def __init__(self):
        # ponytail: defer — duplicate spaCy load across extractive.py/entities.py
        # → shared lazy loader → add if profiling shows it costs meaningfully
        # more than the ~1-2s it costs once per process today.
        self._nlp = spacy.load("en_core_web_sm")

    def process(self, segments: list[dict]) -> list[dict]:
        results = []
        for segment in segments:
            doc = self._nlp(segment["text"])
            seen: set[tuple[str, str]] = set()
            for ent in doc.ents:
                if ent.label_ not in _ALLOWED_TYPES:
                    continue
                key = (ent.text, ent.label_)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "segment_id": segment["segment_id"],
                        "entity_text": ent.text,
                        "entity_type": ent.label_,
                    }
                )
        return results
