"""Extractive summarization via pytextrank sentence-level ranking."""
from src.processors.base import NLPProcessor
import spacy
import pytextrank  # noqa: F401  (registers the "textrank" spaCy pipe factory)


class ExtractiveSummarizer(NLPProcessor):
    name = "extractive_summary"

    def __init__(self):
        # ponytail: defer — duplicate spaCy load across extractive.py/entities.py
        # → shared lazy loader → add if profiling shows it costs meaningfully
        # more than the ~1-2s it costs once per process today.
        self._nlp = spacy.load("en_core_web_sm")
        self._nlp.add_pipe("textrank")

    def process(self, segments: list[dict]) -> list[dict]:
        results = []
        for segment in segments:
            doc = self._nlp(segment["text"])
            sent_list = list(doc.sents)
            sent_dist = doc._.textrank.calc_sent_dist(limit_phrases=10)
            sentence_scores = [
                (sent_list[s.sent_id].text, -s.distance) for s in sent_dist
            ]
            if sentence_scores:
                summary = max(sentence_scores, key=lambda pair: pair[1])[0]
            else:
                summary = ""
            results.append(
                {
                    "segment_id": segment["segment_id"],
                    "summary": summary,
                    "sentence_scores": sentence_scores,
                }
            )
        return results
