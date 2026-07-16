"""Topic segmentation via sentence-embedding cosine distance between adjacent segments."""
from src.processors.base import NLPProcessor
from sentence_transformers import SentenceTransformer, util


class TopicSegmenter(NLPProcessor):
    name = "topic_segmenter"

    def __init__(self, threshold: float = 0.35):
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

    def process(self, segments: list[dict]) -> list[dict]:
        embeddings = self._model.encode([s["text"] for s in segments])
        results = [{"segment_id": segments[0]["segment_id"], "topic_label": "topic_0"}]
        topic_idx = 0
        for i in range(1, len(segments)):
            distance = 1.0 - float(util.cos_sim(embeddings[i], embeddings[i - 1])[0][0])
            if distance > self.threshold:
                topic_idx += 1
            results.append(
                {"segment_id": segments[i]["segment_id"], "topic_label": f"topic_{topic_idx}"}
            )
        return results
