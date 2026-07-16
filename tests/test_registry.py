import json
from pathlib import Path

from src.processors.base import NLPProcessor
from src.processors.registry import run_all, PROCESSORS


class FakeProcessor(NLPProcessor):
    name = "fake_a"

    def process(self, segments):
        return [{"segment_id": s["segment_id"], "value": 1} for s in segments]


class FakeVariableProcessor(NLPProcessor):
    name = "fake_entities"

    def process(self, segments):
        return [{"segment_id": segments[0]["segment_id"], "entity": "x"},
                {"segment_id": segments[0]["segment_id"], "entity": "y"}]


def test_run_all_keys_by_processor_name():
    segments = [{"segment_id": 1, "text": "hello"}, {"segment_id": 2, "text": "world"}]
    result = run_all(segments, processors=[FakeProcessor()])
    assert result == {"fake_a": [{"segment_id": 1, "value": 1}, {"segment_id": 2, "value": 1}]}


def test_run_all_handles_variable_length_output():
    segments = [{"segment_id": 1, "text": "a"}, {"segment_id": 2, "text": "b"}, {"segment_id": 3, "text": "c"}]
    result = run_all(segments, processors=[FakeProcessor(), FakeVariableProcessor()])
    assert set(result.keys()) == {"fake_a", "fake_entities"}
    assert len(result["fake_a"]) == 3
    assert len(result["fake_entities"]) == 2


def test_run_all_defaults_to_module_level_processors(monkeypatch):
    # Verifies the defaulting behavior only (no `processors=` override falls
    # back to registry.PROCESSORS) via monkeypatched fakes — must not
    # dispatch through the real PROCESSORS list, since that includes
    # AbstractiveSummarizer, which calls a live Ollama server. No test in
    # this suite may depend on a live Ollama connection.
    import src.processors.registry as registry_module

    monkeypatch.setattr(registry_module, "PROCESSORS", [FakeProcessor()])
    segments = [{"segment_id": 1, "text": "hello"}, {"segment_id": 2, "text": "world"}]
    result = registry_module.run_all(segments)
    assert result == {"fake_a": [{"segment_id": 1, "value": 1}, {"segment_id": 2, "value": 1}]}


def test_run_all_does_not_mutate_input_segments():
    segments = [{"segment_id": 1, "text": "hello"}, {"segment_id": 2, "text": "world"}]
    original = [dict(s) for s in segments]
    run_all(segments, processors=[FakeProcessor()])
    assert segments == original


def test_sample_segments_fixture_meets_requirements():
    path = Path(__file__).parent / "fixtures" / "sample_segments.json"
    segments = json.loads(path.read_text())
    assert 5 <= len(segments) <= 10
    assert all(isinstance(s["segment_id"], int) and isinstance(s["text"], str) for s in segments)
    assert any(s["text"] == "" for s in segments)
    assert any(
        sum(s["text"].count(sep) for sep in (". ", "? ", "! ")) > 0 for s in segments
    )
    assert any(
        s["text"] != "" and sum(s["text"].count(sep) for sep in (". ", "? ", "! ")) == 0
        for s in segments
    )
