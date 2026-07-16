"""Standalone cross-video entity co-occurrence analysis. Run manually/
separately from src/run.py, after multiple videos have been processed --
NOT part of the main ingest pipeline. Reads the entities table, computes
which entity PAIRS co-occur within the same segment across the corpus, and
overwrites the co_occurrences table. No automated tests for this file by
design -- verified via a manual runbook against the real warehouse (see
project planning docs), matching this project's decision to keep Spark
test infrastructure to one file (tests/test_db.py) only.
"""
from __future__ import annotations

import argparse

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from src.db import build_spark, read_entities, write_co_occurrences


def compute_co_occurrences(entities_df: DataFrame) -> DataFrame:
    """Self-join entities on segment_id to find (entity_a, entity_b) pairs
    co-occurring in the same segment, entity_a strictly < entity_b
    alphabetically. Dedups on (segment_id, entity_text) BEFORE the join,
    dropping entity_type entirely -- this is what prevents a self-pair when
    the same entity_text is tagged under two different entity_types in one
    segment (e.g. "Pixar" as both ORG and PERSON). segment_id is globally
    unique across videos per the schema, so joining on segment_id alone is
    sufficient -- no separate video_id equality condition needed.

    Returns the FULL, UNFILTERED aggregation -- video_count (countDistinct
    video_id) and segment_count (count) per pair. --min-videos filtering
    happens only in main(), never here, so this function stays
    independently callable regardless of CLI concerns."""
    dedup = entities_df.select("segment_id", "video_id", "entity_text").distinct()

    left, right = dedup.alias("a"), dedup.alias("b")
    pairs = left.join(
        right,
        (F.col("a.segment_id") == F.col("b.segment_id"))
        & (F.col("a.entity_text") < F.col("b.entity_text")),
    ).select(
        F.col("a.entity_text").alias("entity_a"),
        F.col("b.entity_text").alias("entity_b"),
        F.col("a.video_id").alias("video_id"),
        F.col("a.segment_id").alias("segment_id"),
    )

    return pairs.groupBy("entity_a", "entity_b").agg(
        F.countDistinct("video_id").alias("video_count"),
        F.count("segment_id").alias("segment_count"),
    )


def main() -> None:
    """CLI: --min-videos (default 2, filters what gets WRITTEN to
    co_occurrences), --top-n (default 30, filters ONLY what gets PRINTED to
    stdout -- never affects what's persisted), --warehouse (default
    data/iceberg). Overwrites co_occurrences each run (via
    write_co_occurrences's overwrite semantics) -- never appends, since
    counts are a function of the ENTIRE current entities table, recomputed
    from scratch."""
    parser = argparse.ArgumentParser(description="Cross-video entity co-occurrence analysis")
    parser.add_argument("--min-videos", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--warehouse", type=str, default="data/iceberg")
    args = parser.parse_args()

    spark = build_spark(args.warehouse)
    try:
        result_df = compute_co_occurrences(read_entities(spark))
        filtered_df = result_df.filter(F.col("video_count") >= args.min_videos)

        pairs = [row.asDict() for row in filtered_df.collect()]
        write_co_occurrences(spark, pairs)

        top = filtered_df.orderBy(F.col("segment_count").desc()).limit(args.top_n).collect()
        for row in top:
            print(f"{row['entity_a']} <-> {row['entity_b']}: "
                  f"{row['video_count']} videos, {row['segment_count']} segments")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
