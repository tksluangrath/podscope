"""Summary quality metrics: compression ratio, semantic similarity, and TextRank score lookup."""

from sentence_transformers import SentenceTransformer, util

_MODEL: SentenceTransformer | None = None  # ponytail: lazy-loaded so importing this module doesn't trigger a model download


def compression_ratio(original: str, summary: str) -> float | None:
    if not original.strip():
        return None
    return len(summary.split()) / len(original.split())


def semantic_similarity(text_a: str, text_b: str, model: SentenceTransformer | None = None) -> float:
    global _MODEL
    if model is None:
        if _MODEL is None:
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        model = _MODEL
    embeddings = model.encode([text_a, text_b])
    return float(util.cos_sim(embeddings[0], embeddings[1]).item())


def textrank_score(sentence_scores: list[tuple[str, float]], target_sentence: str) -> float | None:
    target = target_sentence.strip()
    scores = dict(sentence_scores)
    return scores.get(target) if target else None


def compute_all(segment_text: str, ext_summary: str, abs_summary: str, sentence_scores: list[tuple[str, float]]) -> dict:
    return {
        "compression_ratio": compression_ratio(segment_text, abs_summary),
        "semantic_similarity": semantic_similarity(ext_summary, abs_summary),
        "textrank_score": textrank_score(sentence_scores, ext_summary),
    }
