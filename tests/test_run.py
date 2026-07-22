from copy import deepcopy
from unittest.mock import MagicMock

import src.run as run_module
from src.run import _assign_entity_ids, _merge_nlp, _remap_segment_ids, process_playlist


def _make_nlp_results():
    return {
        "extractive_summary": [
            {
                "segment_id": 1,
                "summary": "The cat sat on the mat.",
                "sentence_scores": [("The cat sat on the mat.", 0.9)],
            }
        ],
        "topic_segmenter": [{"segment_id": 1, "topic_label": "topic_0"}],
        "abstractive_summary": [{"segment_id": 1, "summary": "A cat is sitting."}],
        "entities": [],
    }


def test_merge_nlp_populates_all_expected_keys():
    segments = [{"segment_id": 1, "text": "The cat sat on the mat."}]
    nlp_results = _make_nlp_results()

    result = _merge_nlp(segments, nlp_results)

    assert set(result[0].keys()) >= {
        "segment_id",
        "text",
        "topic_label",
        "ext_summary",
        "abs_summary",
        "textrank_score",
        "compression_ratio",
        "semantic_similarity",
    }


def test_merge_nlp_calls_compute_all_with_correct_arguments(monkeypatch):
    fake_compute_all = MagicMock(
        return_value={
            "compression_ratio": 0.5,
            "semantic_similarity": 0.8,
            "textrank_score": 0.9,
        }
    )
    monkeypatch.setattr("src.metrics.compute_all", fake_compute_all)

    segments = [{"segment_id": 1, "text": "The cat sat on the mat."}]
    nlp_results = _make_nlp_results()

    _merge_nlp(segments, nlp_results)

    fake_compute_all.assert_called_once_with(
        "The cat sat on the mat.",
        "The cat sat on the mat.",
        "A cat is sitting.",
        [("The cat sat on the mat.", 0.9)],
    )


def test_merge_nlp_computes_compression_ratio_against_full_topic_text(monkeypatch):
    # Two segments share topic_0 and one abstractive summary (as
    # AbstractiveSummarizer now produces per topic-group, not per segment).
    # compression_ratio must be computed against BOTH segments' text joined
    # -- the text actually fed to the LLM -- not either segment's own
    # fragment alone, or it comes out inflated above 1.0.
    fake_compute_all = MagicMock(return_value={
        "compression_ratio": 0.5, "semantic_similarity": 0.8, "textrank_score": 0.9,
    })
    monkeypatch.setattr("src.metrics.compute_all", fake_compute_all)

    segments = [
        {"segment_id": 1, "text": "The cat sat."},
        {"segment_id": 2, "text": "on the mat."},
    ]
    nlp_results = {
        "extractive_summary": [
            {"segment_id": 1, "summary": "The cat sat.", "sentence_scores": []},
            {"segment_id": 2, "summary": "on the mat.", "sentence_scores": []},
        ],
        "topic_segmenter": [
            {"segment_id": 1, "topic_label": "topic_0"},
            {"segment_id": 2, "topic_label": "topic_0"},
        ],
        "abstractive_summary": [
            {"segment_id": 1, "summary": "A cat sat on a mat."},
            {"segment_id": 2, "summary": "A cat sat on a mat."},
        ],
        "entities": [],
    }

    _merge_nlp(segments, nlp_results)

    for call in fake_compute_all.call_args_list:
        assert call.args[0] == "The cat sat.\non the mat."


def test_remap_segment_ids_applies_correct_offset():
    segments = [
        {"segment_id": 0, "text": "a"},
        {"segment_id": 1, "text": "b"},
        {"segment_id": 2, "text": "c"},
    ]

    result = _remap_segment_ids(segments, offset=100)

    assert [s["segment_id"] for s in result] == [100, 101, 102]


def test_remap_segment_ids_does_not_mutate_input():
    segments = [
        {"segment_id": 0, "text": "a"},
        {"segment_id": 1, "text": "b"},
    ]
    original = deepcopy(segments)

    _remap_segment_ids(segments, offset=100)

    assert segments == original


def test_assign_entity_ids_applies_independent_offset():
    entities = [
        {"segment_id": 100, "entity_text": "Alice", "entity_type": "PERSON"},
        {"segment_id": 101, "entity_text": "Bob", "entity_type": "PERSON"},
    ]

    result = _assign_entity_ids(entities, offset=50)

    assert [e["entity_id"] for e in result] == [50, 51]
    assert [e["segment_id"] for e in result] == [100, 101]


def test_merge_nlp_preserves_already_global_entity_segment_ids():
    segments = [{"segment_id": 1, "text": "The cat sat on the mat."}]
    nlp_results = _make_nlp_results()
    nlp_results["entities"] = [
        {"segment_id": 101, "entity_text": "Alice", "entity_type": "PERSON"}
    ]
    entities_before = deepcopy(nlp_results["entities"])

    _merge_nlp(segments, nlp_results)

    assert nlp_results["entities"] == entities_before


def test_process_playlist_continues_past_single_episode_failure(monkeypatch):
    monkeypatch.setattr(run_module.ingest, "expand_playlist", lambda url: ["url1", "url2", "url3"])
    monkeypatch.setattr(run_module.ingest, "peek_video_id", lambda url: url.replace("url", "id"))

    calls = []

    def fake_process_video(url, spark, model_size):
        calls.append(url)
        if url == "url2":
            raise RuntimeError("boom")

    monkeypatch.setattr(run_module, "process_video", fake_process_video)

    results = process_playlist("playlist_url", spark=MagicMock(), model_size="tiny")

    assert calls == ["url1", "url2", "url3"]  # continued past the failure
    assert results == [("url1", None), ("url2", "boom"), ("url3", None)]


def test_process_playlist_prints_episode_done_sentinel_for_every_entry(monkeypatch, capsys):
    monkeypatch.setattr(run_module.ingest, "expand_playlist", lambda url: ["url1", "url2"])
    monkeypatch.setattr(run_module.ingest, "peek_video_id", lambda url: url.replace("url", "id"))

    def fake_process_video(url, spark, model_size):
        if url == "url2":
            raise RuntimeError("boom")

    monkeypatch.setattr(run_module, "process_video", fake_process_video)

    process_playlist("playlist_url", spark=MagicMock(), model_size="tiny")

    out = capsys.readouterr().out
    assert "PODSCOPE_EPISODE_DONE 1/2 id1 ok" in out
    assert "PODSCOPE_EPISODE_DONE 2/2 id2 failed" in out


def test_process_playlist_falls_back_to_unknown_when_peek_fails(monkeypatch, capsys):
    monkeypatch.setattr(run_module.ingest, "expand_playlist", lambda url: ["url1"])
    monkeypatch.setattr(run_module.ingest, "peek_video_id", lambda url: None)
    monkeypatch.setattr(run_module, "process_video", lambda url, spark, model_size: None)

    process_playlist("playlist_url", spark=MagicMock(), model_size="tiny")

    out = capsys.readouterr().out
    assert "PODSCOPE_EPISODE_DONE 1/1 unknown ok" in out
