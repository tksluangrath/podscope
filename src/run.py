"""Orchestration layer: wires ingest -> transcribe -> registry -> metrics -> db
into a single-video pipeline and a CLI over it.

`_remap_segment_ids`, `_assign_entity_ids`, and `_merge_nlp` are pure
functions with no I/O, so they're unit tested directly (tests/test_run.py).
`_preflight_check`, `process_video`, `process_batch`, and `main` glue those
pure functions to real Spark/Ollama/yt-dlp/whisper calls -- they're exercised
only by a manual end-to-end run, not pytest, since testing them would mean
mocking every I/O boundary for no real coverage gain over the pure-function
tests plus the E2E smoke run.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from datetime import datetime, timezone

import spacy

from src import db, ingest, metrics, transcribe
from src.processors import registry
from src.processors.abstractive import AbstractiveSummarizer
from src.processors.topics import TopicSegmenter

# ponytail: honor OLLAMA_HOST (set by docker-compose.yml to reach the host's
# Ollama from inside the container); defaults to localhost for bare-host runs.
OLLAMA_URL = f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}/api/tags"


def _remap_segment_ids(segments: list[dict], offset: int) -> list[dict]:
    """New dicts with segment_id += offset. Does not mutate input."""
    return [{**s, "segment_id": s["segment_id"] + offset} for s in segments]


def _assign_entity_ids(entities: list[dict], offset: int) -> list[dict]:
    """New dicts with entity_id = local enumeration index + offset.
    Does not touch segment_id -- that's already global by the time
    entities reach this function.
    """
    return [{**e, "entity_id": i + offset} for i, e in enumerate(entities)]


def _merge_nlp(segments: list[dict], nlp_results: dict[str, list[dict]]) -> list[dict]:
    """Join each processor's per-segment output onto the base segment dict,
    then call metrics.compute_all() per segment. Entities are not merged
    here -- they stay a separate list handled by the caller.
    """
    extractive_by_id = {r["segment_id"]: r for r in nlp_results["extractive_summary"]}
    topics_by_id = {r["segment_id"]: r for r in nlp_results["topic_segmenter"]}
    abstractive_by_id = {r["segment_id"]: r for r in nlp_results["abstractive_summary"]}

    enriched = []
    for seg in segments:
        sid = seg["segment_id"]
        ext = extractive_by_id.get(sid, {})
        topic = topics_by_id.get(sid, {})
        abstr = abstractive_by_id.get(sid, {})

        ext_summary = ext.get("summary", "")
        abs_summary = abstr.get("summary", "")
        sentence_scores = ext.get("sentence_scores", [])

        computed = metrics.compute_all(seg["text"], ext_summary, abs_summary, sentence_scores)

        enriched.append({
            **seg,
            "topic_label": topic.get("topic_label"),
            "ext_summary": ext_summary,
            "abs_summary": abs_summary,
            **computed,
        })
    return enriched


def _preflight_check() -> None:
    """Verify Ollama and en_core_web_sm are available before any expensive
    work (download/transcribe) starts.
    """
    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=5)
    except Exception as e:
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_URL} -- run `ollama serve`"
        ) from e

    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeError(
            "en_core_web_sm not installed -- run `python -m spacy download en_core_web_sm`"
        )


def process_video(url: str, spark, model_size: str) -> None:
    """Full single-video pipeline. Step order (4 before 5) matters: remapping
    segment_id to a global value before registry.run_all() means every
    processor's output -- including entities' segment_id -- already carries
    the global ID, with no second remap needed later.
    """
    audio_path, title, video_id = ingest.download_audio(url)

    if db.video_exists(spark, video_id):
        print(f"Video {video_id} already processed, skipping.")
        return

    segments = transcribe.transcribe(audio_path, model_size)

    offset = db.read_max_id(spark, "segments", "segment_id") + 1
    segments = _remap_segment_ids(segments, offset)

    nlp_results = registry.run_all(segments)

    enriched_segments = _merge_nlp(segments, nlp_results)
    enriched_segments = [{**s, "video_id": video_id} for s in enriched_segments]

    entity_offset = db.read_max_id(spark, "entities", "entity_id") + 1
    entities = _assign_entity_ids(nlp_results["entities"], entity_offset)
    entities = [{**e, "video_id": video_id} for e in entities]

    db.write_segments(spark, enriched_segments)
    db.write_entities(spark, entities)

    db.write_video(spark, video_id, url, title, datetime.now(timezone.utc))


def process_batch(urls_file: str, spark, model_size: str) -> list[tuple[str, str | None]]:
    """Process every URL in urls_file, continuing past per-URL failures.
    Returns list of (url, error_message_or_None).
    """
    urls = [line.strip() for line in open(urls_file) if line.strip()]

    results: list[tuple[str, str | None]] = []
    for url in urls:
        try:
            process_video(url, spark, model_size)
            results.append((url, None))
        except Exception as e:
            print(f"Failed to process {url}: {e}")
            results.append((url, str(e)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Podscope ingestion pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", type=str, help="Single video URL")
    group.add_argument("--urls-file", type=str, help="Newline-separated file of URLs")
    parser.add_argument("--warehouse", type=str, default="data/iceberg")
    parser.add_argument("--model-size", type=str, default="base")
    parser.add_argument("--topic-threshold", type=float, default=0.35)
    parser.add_argument("--llm-model", type=str, default="llama3")
    args = parser.parse_args()

    _preflight_check()

    spark = db.build_spark(args.warehouse)
    try:
        if args.topic_threshold != 0.35:
            for p in registry.PROCESSORS:
                if isinstance(p, TopicSegmenter):
                    p.threshold = args.topic_threshold

        if args.llm_model != "llama3":
            registry.PROCESSORS = [
                AbstractiveSummarizer(model=args.llm_model) if isinstance(p, AbstractiveSummarizer) else p
                for p in registry.PROCESSORS
            ]

        if args.url:
            process_video(args.url, spark, args.model_size)
        else:
            results = process_batch(args.urls_file, spark, args.model_size)
            failures = [(u, e) for u, e in results if e is not None]
            print(f"Processed {len(results)} URLs: {len(results) - len(failures)} succeeded, {len(failures)} failed.")
            for url, error in failures:
                print(f"  FAILED {url}: {error}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
