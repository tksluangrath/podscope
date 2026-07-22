"""Stage-by-stage timing + memory benchmark for the single-video pipeline.

Runs the same fixed sample video every time (see SAMPLE_VIDEO_URL) so
before/after numbers are comparable, and resets that video's rows out of
Iceberg first so each run is a genuine cold run, not a "skip -- already
processed" no-op.

Usage:
    python scripts/benchmark.py            # print table, exit non-zero on budget miss
    python scripts/benchmark.py --json out.json   # also dump machine-readable results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from src import db, ingest, transcribe
from src.processors import registry
from src.processors.abstractive import OLLAMA_RESIDENT_CEILING_GB, ollama_resident_gb
from src.run import _assign_entity_ids, _merge_nlp, _remap_segment_ids

# ponytail: one fixed video for every benchmark run, ever -- comparing two
# runs of *different* videos would just be measuring video length, not the
# pipeline. Rice University speech, 18:15 -- inside the 15-25 min "typical
# episode" range and already cached locally so download timing reflects a
# real (not synthetic) yt-dlp call.
SAMPLE_VIDEO_URL = "https://www.youtube.com/watch?v=WZyRbnpGyzQ"
SAMPLE_VIDEO_ID = "WZyRbnpGyzQ"

WALL_CLOCK_BUDGET_SECONDS = 5 * 60


def _reset_video(spark, video_id: str) -> None:
    """Delete video_id's rows from all three tables so this run is a real
    cold run instead of hitting the already-processed skip path."""
    for table in ("videos", "segments", "entities"):
        spark.sql(f"DELETE FROM local.db.{table} WHERE video_id = '{video_id}'")


def _sample_ollama_peak(model: str, stop_event, peak: list[float]) -> None:
    while not stop_event.is_set():
        peak[0] = max(peak[0], ollama_resident_gb(model))
        time.sleep(0.5)


def run_benchmark(model_size: str = "tiny", llm_model: str = "llama3.2:1b") -> dict:
    import threading

    stages: dict[str, float] = {}
    t_setup0 = time.monotonic()
    spark = db.build_spark("data/iceberg")
    _reset_video(spark, SAMPLE_VIDEO_ID)
    stages["setup (spark init)"] = time.monotonic() - t_setup0

    t0 = time.monotonic()
    audio_path, title, video_id = ingest.download_audio(SAMPLE_VIDEO_URL)
    stages["download audio"] = time.monotonic() - t0

    t0 = time.monotonic()
    segments = transcribe.transcribe(audio_path, model_size)
    stages["transcribing"] = time.monotonic() - t0

    offset = db.read_max_id(spark, "segments", "segment_id") + 1
    segments = _remap_segment_ids(segments, offset)

    non_llm = [p for p in registry.PROCESSORS if p.name != "abstractive_summary"]
    llm_procs = [p for p in registry.PROCESSORS if p.name == "abstractive_summary"]

    t0 = time.monotonic()
    nlp_results = registry.run_all(segments, non_llm)
    stages["nlp processors (extractive, entities, topics)"] = time.monotonic() - t0

    stop_event = threading.Event()
    peak = [0.0]
    sampler = threading.Thread(target=_sample_ollama_peak, args=(llm_model, stop_event, peak), daemon=True)
    sampler.start()
    t0 = time.monotonic()
    nlp_results.update(registry.run_all(segments, llm_procs))
    stages["abstractive summarization (ollama)"] = time.monotonic() - t0
    stop_event.set()
    sampler.join(timeout=2)

    enriched_segments = _merge_nlp(segments, nlp_results)
    enriched_segments = [{**s, "video_id": video_id} for s in enriched_segments]
    entity_offset = db.read_max_id(spark, "entities", "entity_id") + 1
    entities = _assign_entity_ids(nlp_results["entities"], entity_offset)
    entities = [{**e, "video_id": video_id} for e in entities]

    t0 = time.monotonic()
    db.write_segments(spark, enriched_segments)
    db.write_entities(spark, entities)
    db.write_video(spark, video_id, SAMPLE_VIDEO_URL, title, datetime.now(timezone.utc))
    stages["writing to iceberg"] = time.monotonic() - t0

    spark.stop()

    return {
        "sample_video_id": video_id,
        "segment_count": len(enriched_segments),
        "entity_count": len(entities),
        "stages": stages,
        "total_wall_clock": sum(stages.values()),
        "ollama_peak_resident_gb": peak[0],
        "model_size": model_size,
        "llm_model": llm_model,
    }


def print_table(result: dict, baseline: dict | None = None) -> None:
    print(f"\nSample video: {result['sample_video_id']}  "
          f"(model_size={result['model_size']}, llm_model={result['llm_model']})")
    print(f"{'Stage':<45} {'Wall-clock':>12} {'Before':>12}")
    print("-" * 71)
    for stage, secs in result["stages"].items():
        before = f"{baseline['stages'][stage]:.1f}s" if baseline and stage in baseline["stages"] else "--"
        print(f"{stage:<45} {secs:>10.1f}s {before:>12}")
    print("-" * 71)
    before_total = f"{baseline['total_wall_clock']:.1f}s" if baseline else "--"
    print(f"{'TOTAL':<45} {result['total_wall_clock']:>10.1f}s {before_total:>12}")
    before_mem = f"{baseline['ollama_peak_resident_gb']:.2f} GB" if baseline else "--"
    print(f"\nOllama peak resident: {result['ollama_peak_resident_gb']:.2f} GB "
          f"(before: {before_mem}, ceiling: {OLLAMA_RESIDENT_CEILING_GB} GB)")
    print(f"Segments: {result['segment_count']}  Entities: {result['entity_count']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Podscope single-video pipeline benchmark")
    parser.add_argument("--model-size", default="tiny")
    parser.add_argument("--llm-model", default="llama3.2:1b")
    parser.add_argument("--baseline", help="path to a previous run's --json output, for comparison")
    parser.add_argument("--json", help="write machine-readable results here")
    args = parser.parse_args()

    baseline = None
    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)

    result = run_benchmark(model_size=args.model_size, llm_model=args.llm_model)
    print_table(result, baseline)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)

    ok = True
    if result["total_wall_clock"] > WALL_CLOCK_BUDGET_SECONDS:
        print(f"\nFAIL: total wall-clock {result['total_wall_clock']:.1f}s exceeds "
              f"{WALL_CLOCK_BUDGET_SECONDS}s budget")
        ok = False
    if result["ollama_peak_resident_gb"] > OLLAMA_RESIDENT_CEILING_GB:
        print(f"\nFAIL: Ollama peak resident {result['ollama_peak_resident_gb']:.2f} GB exceeds "
              f"{OLLAMA_RESIDENT_CEILING_GB} GB ceiling")
        ok = False
    if ok:
        print("\nPASS: within 5-minute budget and memory ceiling")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
