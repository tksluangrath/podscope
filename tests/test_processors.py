import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.processors.extractive import ExtractiveSummarizer
from src.processors.entities import EntityExtractor
from src.processors.topics import TopicSegmenter
from src.processors.abstractive import AbstractiveSummarizer

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_segments.json"


def _load_fixture_segments():
    return json.loads(FIXTURE_PATH.read_text())


def _find_single_sentence_segment(segments):
    """First segment with non-empty text and no internal sentence separator."""
    for s in segments:
        if s["text"] != "" and sum(s["text"].count(sep) for sep in (". ", "? ", "! ")) == 0:
            return s
    raise AssertionError("fixture has no single-sentence segment")


def _find_empty_text_segment(segments):
    for s in segments:
        if s["text"] == "":
            return s
    raise AssertionError("fixture has no empty-text segment")


class TestExtractiveSummarizer:
    @pytest.fixture(scope="class")
    def summarizer(self):
        return ExtractiveSummarizer()

    @pytest.fixture(scope="class")
    def fixture_segments(self):
        return _load_fixture_segments()

    def test_process_returns_one_result_per_segment(self, summarizer, fixture_segments):
        results = summarizer.process(fixture_segments)
        assert len(results) == len(fixture_segments)

    def test_sentence_scores_are_sentence_level_not_phrase_level(self, summarizer, fixture_segments):
        results = summarizer.process(fixture_segments)
        for result in results:
            sentence_texts = [sent for sent, _score in result["sentence_scores"]]
            assert result["summary"] in sentence_texts or (
                result["summary"] == "" and sentence_texts == []
            )

    def test_single_sentence_segment(self, summarizer, fixture_segments):
        target = _find_single_sentence_segment(fixture_segments)
        result = summarizer.process([target])[0]
        assert result["summary"] == target["text"] and len(result["sentence_scores"]) == 1

    def test_empty_text_segment_does_not_raise(self, summarizer, fixture_segments):
        target = _find_empty_text_segment(fixture_segments)
        results = summarizer.process(fixture_segments)
        result = next(r for r in results if r["segment_id"] == target["segment_id"])
        assert result["summary"] == "" and result["sentence_scores"] == []


class TestEntityExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        return EntityExtractor()

    @pytest.fixture(scope="class")
    def fixture_segments(self):
        return _load_fixture_segments()

    @pytest.fixture(scope="class")
    def results(self, extractor, fixture_segments):
        return extractor.process(fixture_segments)

    def test_process_returns_only_allowed_types(self, results):
        allowed = {"PERSON", "ORG", "GPE", "DATE", "PRODUCT"}
        assert all(row["entity_type"] in allowed for row in results)

    def test_dedup_within_segment(self, results):
        # segment_id 8's text mentions "John Kim" twice: "...John Kim from the
        # audience sent in a question about whether Maria Chen and John Kim
        # will collaborate on an open dataset next year." Verified directly
        # against tests/fixtures/sample_segments.json.
        matches = [
            row for row in results
            if row["segment_id"] == 8
            and row["entity_text"] == "John Kim"
            and row["entity_type"] == "PERSON"
        ]
        assert len(matches) == 1

    def test_segment_with_no_entities_returns_empty_list_not_error(self, results):
        # segment_id 3, text "So let's just dive right in." has no named entities.
        rows_for_segment = [row for row in results if row["segment_id"] == 3]
        assert rows_for_segment == []


class TestTopicSegmenter:
    @pytest.fixture(scope="class")
    def segmenter(self):
        return TopicSegmenter()

    @pytest.fixture(scope="class")
    def fixture_segments(self):
        return _load_fixture_segments()

    def test_process_returns_one_result_per_segment(self, segmenter, fixture_segments):
        results = segmenter.process(fixture_segments)
        assert len(results) == len(fixture_segments)

    def test_single_segment_never_crosses_threshold(self, segmenter, fixture_segments):
        results = segmenter.process(fixture_segments[:1])
        assert results[0]["topic_label"] == "topic_0"

    def test_low_threshold_forces_boundary_on_dissimilar_segments(self, fixture_segments):
        # segment_id 1 (AI/ML thread: "...future of large language models.")
        # vs segment_id 6 (sourdough thread: "...sourdough starters...").
        by_id = {s["segment_id"]: s for s in fixture_segments}
        pair = [by_id[1], by_id[6]]
        results = TopicSegmenter(threshold=0.01).process(pair)
        assert results[0]["topic_label"] != results[1]["topic_label"]

    def test_high_threshold_keeps_similar_segments_together(self, fixture_segments):
        # segment_id 6 and segment_id 7 are both part of the sourdough thread.
        by_id = {s["segment_id"]: s for s in fixture_segments}
        pair = [by_id[6], by_id[7]]
        results = TopicSegmenter(threshold=0.99).process(pair)
        assert results[0]["topic_label"] == results[1]["topic_label"]


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return SimpleNamespace(content=self.responses[len(self.calls) - 1])


class TestAbstractiveSummarizer:
    @pytest.fixture(scope="class")
    def fixture_segments(self):
        return _load_fixture_segments()

    def test_process_calls_invoke_once_per_segment_not_batch(self, fixture_segments):
        fake = FakeLLM([f"summary {i}" for i in range(len(fixture_segments))])
        summarizer = AbstractiveSummarizer(llm=fake)
        summarizer.process(fixture_segments)
        assert len(fake.calls) == len(fixture_segments)

    def test_process_preserves_segment_order_and_ids(self, fixture_segments):
        fake = FakeLLM([f"summary {i}" for i in range(len(fixture_segments))])
        summarizer = AbstractiveSummarizer(llm=fake)
        results = summarizer.process(fixture_segments)
        assert [r["segment_id"] for r in results] == [s["segment_id"] for s in fixture_segments]

    def test_process_returns_only_segment_id_and_summary_keys(self, fixture_segments):
        fake = FakeLLM([f"summary {i}" for i in range(len(fixture_segments))])
        summarizer = AbstractiveSummarizer(llm=fake)
        results = summarizer.process(fixture_segments)
        assert all(set(r.keys()) == {"segment_id", "summary"} for r in results)

    def test_process_summary_is_llm_response_content_not_hardcoded(self, fixture_segments):
        fake = FakeLLM([f"summary {i}" for i in range(len(fixture_segments))])
        summarizer = AbstractiveSummarizer(llm=fake)
        results = summarizer.process(fixture_segments)
        assert [r["summary"] for r in results] == [f"summary {i}" for i in range(len(fixture_segments))]

    def test_process_does_not_mutate_input_segments(self, fixture_segments):
        fake = FakeLLM([f"summary {i}" for i in range(len(fixture_segments))])
        summarizer = AbstractiveSummarizer(llm=fake)
        before = deepcopy(fixture_segments)
        summarizer.process(fixture_segments)
        assert fixture_segments == before

    def test_constructing_without_injected_llm_builds_real_chatollama(self):
        from langchain_ollama import ChatOllama

        summarizer = AbstractiveSummarizer(model="llama3")
        assert type(summarizer.llm) is ChatOllama
        assert summarizer.llm.model == "llama3"
