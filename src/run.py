"""Orchestration layer: wires ingest -> transcribe -> registry -> metrics -> db
into a single-video pipeline and a CLI over it.

`_remap_segment_ids`, `_assign_entity_ids`, and `_merge_nlp` are pure
functions with no I/O, so they're unit tested directly (tests/test_run.py).
`_preflight_check`, `process_video`, `process_batch`, `process_playlist`, and
`main` glue those pure functions to real Spark/Ollama/yt-dlp/whisper calls --
they're exercised only by a manual end-to-end run, not pytest, since testing
them fully would mean mocking every I/O boundary for no real coverage gain
over the pure-function tests plus the E2E smoke run. `process_playlist`'s own
per-episode continue-past-failure loop and sentinel printing are worth
testing in isolation though, with `expand_playlist`/`process_video` mocked
out -- see tests/test_run.py.
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

    # compression_ratio needs the text actually fed to the LLM, not any one
    # segment's own fragment. abstractive_summary produces one summary per
    # topic group (see AbstractiveSummarizer), so comparing its length
    # against a single ~2-4s segment's word count made compression_ratio
    # come out *above* 1.0 (a "compression" that's really an expansion)
    # whenever a topic spanned more than a couple segments.
    topic_texts: dict[str | None, list[str]] = {}
    for seg in segments:
        label = topics_by_id.get(seg["segment_id"], {}).get("topic_label")
        if seg["text"]:
            topic_texts.setdefault(label, []).append(seg["text"])
    topic_original_text = {label: "\n".join(texts) for label, texts in topic_texts.items()}

    enriched = []
    for seg in segments:
        sid = seg["segment_id"]
        ext = extractive_by_id.get(sid, {})
        topic = topics_by_id.get(sid, {})
        abstr = abstractive_by_id.get(sid, {})

        ext_summary = ext.get("summary", "")
        abs_summary = abstr.get("summary", "")
        sentence_scores = ext.get("sentence_scores", [])

        topic_label = topic.get("topic_label")
        original_text = topic_original_text.get(topic_label, seg["text"])

        computed = metrics.compute_all(original_text, ext_summary, abs_summary, sentence_scores)

        enriched.append({
            **seg,
            "topic_label": topic_label,
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
    # ponytail: cheap no-network check before paying for a full download --
    # matters for batch restarts, where every already-done video would
    # otherwise be re-downloaded just to be thrown away by the check below.
    peeked_id = ingest.peek_video_id(url)
    if peeked_id is not None and db.video_exists(spark, peeked_id):
        print(f"Video {peeked_id} already processed, skipping.")
        return

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


def process_playlist(url: str, spark, model_size: str) -> list[tuple[str, str | None]]:
    """Expand url into its episodes (src.ingest.expand_playlist) and process
    each one, continuing past per-episode failures exactly like
    process_batch does for a pre-built URL list -- same failure policy, not
    a second one invented for playlists.

    Prints a PODSCOPE_EPISODE_DONE sentinel after every episode, success or
    failure, for src/tui.py's stdout-line parser to pick up (same mechanism
    it already uses for "[download]" and "already processed" -- see
    tui.py's run_pipeline).
    """
    urls = ingest.expand_playlist(url)
    total = len(urls)

    results: list[tuple[str, str | None]] = []
    for i, video_url in enumerate(urls, start=1):
        video_id = ingest.peek_video_id(video_url) or "unknown"
        try:
            process_video(video_url, spark, model_size)
            results.append((video_url, None))
            print(f"PODSCOPE_EPISODE_DONE {i}/{total} {video_id} ok")
        except Exception as e:
            print(f"Failed to process {video_url}: {e}")
            results.append((video_url, str(e)))
            print(f"PODSCOPE_EPISODE_DONE {i}/{total} {video_id} failed")
    return results


def _print_batch_summary(results: list[tuple[str, str | None]]) -> None:
    failures = [(u, e) for u, e in results if e is not None]
    print(f"Processed {len(results)} URLs: {len(results) - len(failures)} succeeded, {len(failures)} failed.")
    for url, error in failures:
        print(f"  FAILED {url}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Podscope ingestion pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", type=str, help="Single video URL")
    group.add_argument("--urls-file", type=str, help="Newline-separated file of URLs")
    parser.add_argument("--warehouse", type=str, default="data/iceberg")
    # ponytail: "tiny" over "base" is a measured, named accuracy tradeoff
    # (more transcription errors, especially uncommon words/accents) traded
    # for closing the 5-minute single-video wall-clock budget -- see
    # BENCHMARKS.md for the before/after numbers this was picked from.
    parser.add_argument("--model-size", type=str, default="tiny")
    parser.add_argument("--topic-std-multiplier", type=float, default=1.0)
    parser.add_argument("--llm-model", type=str, default="llama3")
    args = parser.parse_args()

    _preflight_check()

    spark = db.build_spark(args.warehouse)
    try:
        if args.topic_std_multiplier != 1.0:
            for p in registry.PROCESSORS:
                if isinstance(p, TopicSegmenter):
                    p.std_multiplier = args.topic_std_multiplier

        if args.llm_model != "llama3":
            registry.PROCESSORS = [
                AbstractiveSummarizer(model=args.llm_model) if isinstance(p, AbstractiveSummarizer) else p
                for p in registry.PROCESSORS
            ]

        if args.url:
            if ingest.is_playlist_url(args.url):
                _print_batch_summary(process_playlist(args.url, spark, args.model_size))
            else:
                process_video(args.url, spark, args.model_size)
        else:
            _print_batch_summary(process_batch(args.urls_file, spark, args.model_size))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
