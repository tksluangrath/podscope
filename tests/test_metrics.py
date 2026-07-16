import pytest

from src import metrics
from src.metrics import compression_ratio, semantic_similarity, textrank_score, compute_all


def test_compression_ratio_normal():
    assert compression_ratio("one two three four five", "one two") == pytest.approx(0.4)


def test_compression_ratio_empty_original():
    assert compression_ratio("", "summary") is None


def test_compression_ratio_whitespace_only_original():
    assert compression_ratio("   ", "summary") is None


def test_compression_ratio_single_word():
    assert compression_ratio("word", "word") == pytest.approx(1.0)


def test_compression_ratio_full_overlap():
    assert compression_ratio("a b c", "a b c") == pytest.approx(1.0)


def test_semantic_similarity_identical():
    text = "The quick brown fox jumps over the lazy dog."
    score = semantic_similarity(text, text)
    assert score > 0.99


def test_semantic_similarity_unrelated():
    score = semantic_similarity(
        "The stock market crashed today.",
        "I enjoy eating pizza for lunch.",
    )
    assert score < 0.5


def test_semantic_similarity_partial_overlap():
    score = semantic_similarity(
        "The cat sat on the mat.",
        "A cat was sitting on a rug.",
    )
    assert 0.5 <= score <= 0.99


def test_textrank_score_found():
    scores = [("The cat sat.", 0.5), ("It was raining.", 0.2)]
    assert textrank_score(scores, "It was raining.") == pytest.approx(0.2)


def test_textrank_score_not_found():
    scores = [("The cat sat.", 0.5)]
    assert textrank_score(scores, "Not present.") is None


def test_textrank_score_empty_scores():
    assert textrank_score([], "The cat sat.") is None


def test_textrank_score_empty_target():
    scores = [("The cat sat.", 0.5)]
    assert textrank_score(scores, "") is None


def test_textrank_score_strips_whitespace():
    scores = [("The cat sat.", 0.5)]
    assert textrank_score(scores, " The cat sat. ") == pytest.approx(0.5)


def test_compute_all_returns_all_keys():
    result = compute_all(
        "segment text here",
        "ext summary here",
        "abs summary here",
        [("ext summary here", 0.3)],
    )
    assert set(result.keys()) == {"compression_ratio", "semantic_similarity", "textrank_score"}


def test_compute_all_routes_arguments_correctly(monkeypatch):
    calls = {}

    def fake_compression_ratio(a, b):
        calls["compression_ratio"] = (a, b)
        return 1.23

    def fake_semantic_similarity(a, b):
        calls["semantic_similarity"] = (a, b)
        return 4.56

    def fake_textrank_score(a, b):
        calls["textrank_score"] = (a, b)
        return 7.89

    monkeypatch.setattr(metrics, "compression_ratio", fake_compression_ratio)
    monkeypatch.setattr(metrics, "semantic_similarity", fake_semantic_similarity)
    monkeypatch.setattr(metrics, "textrank_score", fake_textrank_score)

    segment_text = "SEGMENT_SENTINEL"
    ext_summary = "EXT_SENTINEL"
    abs_summary = "ABS_SENTINEL"
    sentence_scores = [("SCORES_SENTINEL", 1.0)]

    metrics.compute_all(segment_text, ext_summary, abs_summary, sentence_scores)

    assert calls["semantic_similarity"] == (ext_summary, abs_summary)
    assert calls["compression_ratio"] == (segment_text, abs_summary)
    assert calls["textrank_score"] == (sentence_scores, ext_summary)
