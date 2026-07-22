"""Correctness regression gate for CI (GitHub-hosted runners, no Ollama/GPU).

Runs the fixed sample_segments.json fixture through the real extractive,
entity, and topic processors plus the real metrics computation -- the parts
of the pipeline that don't depend on a live Ollama server -- and checks the
result against a checked-in golden output. This is what should fail if a
perf change (model swap, batching, etc.) accidentally changes segment
counts, entity counts, or metric computation, not just summary wording.

Abstractive summarization is stubbed out (not skipped silently -- see
FakeAbstractive below) since no test in this suite may depend on a live
Ollama connection (tests/test_registry.py's existing rule). That's fine
here: this test is about the deterministic code paths (merge, metrics),
not about evaluating any particular LLM's summary quality.
"""
import json
from pathlib import Path

import pytest

from src.processors.entities import EntityExtractor
from src.processors.extractive import ExtractiveSummarizer
from src.processors.topics import TopicSegmenter
from src.run import _merge_nlp

FIXTURES = Path(__file__).parent / "fixtures"


class FakeAbstractive:
    name = "abstractive_summary"

    def process(self, segments):
        return [{"segment_id": s["segment_id"], "summary": " ".join(s["text"].split()[:6])} for s in segments]


def _run_pipeline_stages(segments):
    nlp_results = {
        "extractive_summary": ExtractiveSummarizer().process(segments),
        "entities": EntityExtractor().process(segments),
        "topic_segmenter": TopicSegmenter().process(segments),
        "abstractive_summary": FakeAbstractive().process(segments),
    }
    enriched = _merge_nlp(segments, nlp_results)
    return enriched, nlp_results["entities"]


def test_correctness_matches_golden_output():
    segments = json.loads((FIXTURES / "sample_segments.json").read_text())
    golden = json.loads((FIXTURES / "golden_correctness.json").read_text())

    enriched, entities = _run_pipeline_stages(segments)

    assert len(enriched) == golden["segment_count"]
    assert len(entities) == golden["entity_count"]
    assert [s["topic_label"] for s in enriched] == golden["topic_labels"]

    for got, want in zip((s["compression_ratio"] for s in enriched), golden["compression_ratios"]):
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want)

    for got, want in zip((s["semantic_similarity"] for s in enriched), golden["semantic_similarities"]):
        if want is None:
            assert got is None
        else:
            # A looser tolerance than compression_ratio: semantic_similarity
            # comes from a real embedding model, so exact floats can drift a
            # hair across sentence-transformers versions even with fixed
            # input -- the regression this test guards against is a large
            # jump (broken merge, wrong segment matched), not float noise.
            assert got == pytest.approx(want, abs=0.02)
