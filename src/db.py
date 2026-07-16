"""The only Spark/Iceberg boundary in the codebase.

All reads and writes to the local Iceberg warehouse (catalog `local`,
namespace `db`) go through this module.
"""
from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

VIDEOS_SCHEMA = StructType([
    StructField("video_id", StringType(), nullable=False),
    StructField("url", StringType(), nullable=False),
    StructField("title", StringType(), nullable=True),
    StructField("processed_at", TimestampType(), nullable=False),
])

SEGMENTS_SCHEMA = StructType([
    StructField("segment_id", IntegerType(), nullable=False),
    StructField("video_id", StringType(), nullable=False),
    StructField("start_time", FloatType(), nullable=False),
    StructField("end_time", FloatType(), nullable=False),
    StructField("text", StringType(), nullable=True),
    StructField("topic_label", StringType(), nullable=True),
    StructField("ext_summary", StringType(), nullable=True),
    StructField("abs_summary", StringType(), nullable=True),
    StructField("textrank_score", FloatType(), nullable=True),
    StructField("compression_ratio", FloatType(), nullable=True),
    StructField("semantic_similarity", FloatType(), nullable=True),
])

ENTITIES_SCHEMA = StructType([
    StructField("entity_id", IntegerType(), nullable=False),
    StructField("segment_id", IntegerType(), nullable=False),
    StructField("video_id", StringType(), nullable=False),
    StructField("entity_text", StringType(), nullable=False),
    StructField("entity_type", StringType(), nullable=False),
])

CO_OCCURRENCES_SCHEMA = StructType([
    StructField("entity_a", StringType(), nullable=False),
    StructField("entity_b", StringType(), nullable=False),
    StructField("video_count", IntegerType(), nullable=False),
    StructField("segment_count", IntegerType(), nullable=False),
])


def build_spark(warehouse_path: str) -> SparkSession:
    """Construct a SparkSession configured with a local Iceberg HadoopCatalog
    at warehouse_path, ZSTD parquet compression, then ensure the local.db
    namespace and all 4 tables exist (CREATE TABLE IF NOT EXISTS against the
    explicit StructTypes). Caller is responsible for calling spark.stop().
    """
    spark = (
        SparkSession.builder.appName("podscope")
        .config(
            "spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.7.1",
        )
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.local.type", "hadoop")
        .config("spark.sql.catalog.local.warehouse", warehouse_path)
        .config("spark.sql.parquet.compression.codec", "zstd")
        .getOrCreate()
    )

    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.db")

    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS local.db.videos (
            video_id STRING,
            url STRING,
            title STRING,
            processed_at TIMESTAMP
        ) USING iceberg
        """
    )

    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS local.db.segments (
            segment_id INT,
            video_id STRING,
            start_time FLOAT,
            end_time FLOAT,
            text STRING,
            topic_label STRING,
            ext_summary STRING,
            abs_summary STRING,
            textrank_score FLOAT,
            compression_ratio FLOAT,
            semantic_similarity FLOAT
        ) USING iceberg
        """
    )

    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS local.db.entities (
            entity_id INT,
            segment_id INT,
            video_id STRING,
            entity_text STRING,
            entity_type STRING
        ) USING iceberg
        """
    )

    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS local.db.co_occurrences (
            entity_a STRING,
            entity_b STRING,
            video_count INT,
            segment_count INT
        ) USING iceberg
        """
    )

    return spark


def video_exists(spark: SparkSession, video_id: str) -> bool:
    """True if video_id is already present in the videos table."""
    return (
        spark.table("local.db.videos")
        .filter(f"video_id = '{video_id}'")
        .limit(1)
        .count()
        > 0
    )


def write_video(
    spark: SparkSession, video_id: str, url: str, title: str, processed_at: datetime
) -> None:
    """Append one row to videos, using VIDEOS_SCHEMA explicitly for createDataFrame."""
    df = spark.createDataFrame(
        [(video_id, url, title, processed_at)], schema=VIDEOS_SCHEMA
    )
    df.writeTo("local.db.videos").append()


def read_max_id(spark: SparkSession, table: str, id_column: str) -> int:
    """Return the current max value of id_column in local.db.<table>, or -1
    if the table has zero rows. table is the bare table name (e.g. "segments"),
    not the full catalog path -- this function qualifies it internally.
    """
    row = spark.sql(f"SELECT MAX({id_column}) AS m FROM local.db.{table}").collect()[0]
    return -1 if row["m"] is None else int(row["m"])


def write_segments(spark: SparkSession, segments: list[dict]) -> None:
    """Append enriched segment rows, using SEGMENTS_SCHEMA explicitly for
    createDataFrame -- must correctly handle rows where textrank_score/
    compression_ratio/semantic_similarity are None.
    """
    df = spark.createDataFrame(segments, schema=SEGMENTS_SCHEMA)
    df.writeTo("local.db.segments").append()


def write_entities(spark: SparkSession, entities: list[dict]) -> None:
    """Append entity rows using ENTITIES_SCHEMA explicitly -- must handle
    an empty list without error (spark.createDataFrame([], schema) is valid
    when a schema is passed explicitly).
    """
    df = spark.createDataFrame(entities, schema=ENTITIES_SCHEMA)
    df.writeTo("local.db.entities").append()


def write_co_occurrences(spark: SparkSession, pairs: list[dict]) -> None:
    """OVERWRITE the co_occurrences table (not append) using
    CO_OCCURRENCES_SCHEMA explicitly. Co-occurrence counts are a function of
    the entire current entities table, recomputed from scratch each run --
    appending would duplicate every existing pair on a second run.

    Uses createOrReplace(), not overwritePartitions(): on an unpartitioned
    table, overwritePartitions()'s dynamic-overwrite semantics only replace
    partitions present in the incoming data, so writing an empty pairs list
    (a legitimate result -- no entity pair yet co-occurs across enough
    videos) is a silent no-op that leaves stale rows behind. createOrReplace()
    always drops and recreates the table from the given DataFrame, correctly
    overwriting to empty when pairs is empty.
    """
    df = spark.createDataFrame(pairs, schema=CO_OCCURRENCES_SCHEMA)
    df.writeTo("local.db.co_occurrences").createOrReplace()


def read_segments(spark: SparkSession, video_id: str | None = None) -> DataFrame:
    """Read segments, optionally filtered to one video_id."""
    df = spark.table("local.db.segments")
    if video_id is not None:
        df = df.filter(df.video_id == video_id)
    return df


def read_entities(spark: SparkSession, video_id: str | None = None) -> DataFrame:
    """Read entities, optionally filtered to one video_id."""
    df = spark.table("local.db.entities")
    if video_id is not None:
        df = df.filter(df.video_id == video_id)
    return df


def read_co_occurrences(spark: SparkSession) -> DataFrame:
    """Read the full co_occurrences table."""
    return spark.table("local.db.co_occurrences")
