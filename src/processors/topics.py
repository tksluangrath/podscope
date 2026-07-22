"""Topic segmentation via sentence-embedding cosine distance between adjacent segments."""
import statistics

from src.processors.base import NLPProcessor
from sentence_transformers import SentenceTransformer, util


class TopicSegmenter(NLPProcessor):
    name = "topic_segmenter"

    def __init__(self, std_multiplier: float = 1.0):
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        # ponytail: adaptive per-video threshold (mean + k*stdev of this
        # video's own adjacent-segment distances) replaces a fixed 0.35
        # cutoff. Whisper segments are short conversational utterances --
        # measured on a real 70-min video, their adjacent-segment distances
        # had a *median* of 0.80, so a fixed absolute 0.35 threshold flagged
        # ~97% of segments as new topics (1726 topics for 1783 segments,
        # i.e. barely any grouping at all). Self-calibrating to each video's
        # own pacing instead of one global constant fixes that; on the same
        # video this drops to ~136 topics.
        self.std_multiplier = std_multiplier

    def process(self, segments: list[dict]) -> list[dict]:
        if not segments:
            return []
        embeddings = self._model.encode([s["text"] for s in segments])
        distances = [
            1.0 - float(util.cos_sim(embeddings[i], embeddings[i - 1])[0][0])
            for i in range(1, len(embeddings))
        ]
        # Too few segments to have a meaningful spread -- never split.
        threshold = (
            statistics.mean(distances) + self.std_multiplier * statistics.stdev(distances)
            if len(distances) >= 2
            else float("inf")
        )

        results = [{"segment_id": segments[0]["segment_id"], "topic_label": "topic_0"}]
        topic_idx = 0
        for i, distance in enumerate(distances, start=1):
            if distance > threshold:
                topic_idx += 1
            results.append(
                {"segment_id": segments[i]["segment_id"], "topic_label": f"topic_{topic_idx}"}
            )
        return results
