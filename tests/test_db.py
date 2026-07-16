# ponytail: these tests are order-dependent BY DESIGN. They share ONE
# module-scoped SparkSession + ONE temp Iceberg warehouse for the whole file
# (building a SparkSession with spark.jars.packages resolves a jar over the
# network/Ivy cache every time -- doing that once per test instead of once
# per file would multiply that cost 9x and flake). Later tests rely on state
# earlier tests leave behind (e.g. read_max_id returning -1 on an empty
# table only works if that test runs BEFORE anything writes to that table).
#
# DO NOT reorder these tests. DO NOT run this file with pytest-randomly or
# any other test-order-randomizing plugin. DO NOT run this file with
# pytest-xdist / parallelized -- it will break.
from __future__ import annotations

import tempfile
from datetime import datetime, timezone

import pytest

from src.db import (
    build_spark,
    read_co_occurrences,
    read_entities,
    read_max_id,
    read_segments,
    video_exists,
    write_co_occurrences,
    write_entities,
    write_segments,
    write_video,
)


@pytest.fixture(scope="module")
def spark():
    with tempfile.TemporaryDirectory() as warehouse:
        s = build_spark(warehouse)
        yield s
        s.stop()


def _segment(
    segment_id: int,
    video_id: str = "vid-001",
    start_time: float = 0.0,
    end_time: float = 1.0,
    text: str = "hello world",
    topic_label: str = "",
    ext_summary: str = "",
    abs_summary: str = "",
    textrank_score: float | None = None,
    compression_ratio: float | None = None,
    semantic_similarity: float | None = None,
) -> dict:
    return {
        "segment_id": segment_id,
        "video_id": video_id,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
        "topic_label": topic_label,
        "ext_summary": ext_summary,
        "abs_summary": abs_summary,
        "textrank_score": textrank_score,
        "compression_ratio": compression_ratio,
        "semantic_similarity": semantic_similarity,
    }


def test_build_spark_creates_all_four_tables(spark):
    rows = spark.sql("SHOW TABLES IN local.db").collect()
    table_names = {row.tableName for row in rows}
    assert {"videos", "segments", "entities", "co_occurrences"} <= table_names

    columns = {row.col_name for row in spark.sql("DESCRIBE local.db.segments").collect()}
    expected_columns = {
        "segment_id",
        "video_id",
        "start_time",
        "end_time",
        "text",
        "topic_label",
        "ext_summary",
        "abs_summary",
        "textrank_score",
        "compression_ratio",
        "semantic_similarity",
    }
    assert expected_columns <= columns


def test_video_exists_false_for_unwritten_id(spark):
    assert video_exists(spark, "never-written-id") is False


def test_write_video_then_video_exists_true(spark):
    write_video(
        spark,
        "vid-001",
        "https://youtube.com/watch?v=vid-001",
        "Test Video",
        datetime.now(timezone.utc),
    )
    assert video_exists(spark, "vid-001") is True


def test_read_max_id_returns_negative_one_on_empty_table(spark):
    assert read_max_id(spark, "segments", "segment_id") == -1


def test_read_max_id_returns_correct_max_after_write_segments(spark):
    write_segments(
        spark,
        [_segment(5), _segment(10), _segment(3)],
    )
    assert read_max_id(spark, "segments", "segment_id") == 10


def test_write_segments_then_read_segments_roundtrips_with_none_metrics(spark):
    write_segments(
        spark,
        [
            _segment(
                20,
                video_id="vid-none-metrics",
                start_time=12.5,
                end_time=15.0,
                text="none metrics segment",
                textrank_score=None,
                compression_ratio=None,
                semantic_similarity=None,
            )
        ],
    )

    df = read_segments(spark, video_id="vid-none-metrics")
    pdf = df.toPandas()
    assert len(pdf) == 1
    row = pdf.iloc[0]

    assert row["text"] == "none metrics segment"
    assert row["start_time"] == pytest.approx(12.5)
    assert row["end_time"] == pytest.approx(15.0)
    assert pd_is_null(row["textrank_score"])
    assert pd_is_null(row["compression_ratio"])
    assert pd_is_null(row["semantic_similarity"])


def pd_is_null(value) -> bool:
    import pandas as pd

    return pd.isna(value)


def test_write_entities_handles_empty_list(spark):
    write_entities(spark, [])


def test_write_co_occurrences_overwrites_not_duplicates(spark):
    write_co_occurrences(
        spark,
        [{"entity_a": "Alice", "entity_b": "Bob", "video_count": 1, "segment_count": 2}],
    )
    write_co_occurrences(
        spark,
        [{"entity_a": "Carol", "entity_b": "Dave", "video_count": 3, "segment_count": 5}],
    )

    pdf = read_co_occurrences(spark).toPandas()
    pairs = set(zip(pdf["entity_a"], pdf["entity_b"]))
    assert pairs == {("Carol", "Dave")}


def test_read_segments_filters_by_video_id(spark):
    write_segments(spark, [_segment(30, video_id="vid-other")])

    pdf = read_segments(spark, video_id="vid-other").toPandas()
    assert set(pdf["video_id"]) == {"vid-other"}
