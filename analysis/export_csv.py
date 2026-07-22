"""Export the Iceberg warehouse to flat CSV files for Power BI (or any tool
that reads CSV/folder sources) to import. Run manually/separately from
src/run.py, any time you want a fresh snapshot -- NOT part of the main
ingest pipeline. Overwrites each output file from scratch every run. No
automated tests for this file by design, matching this project's other
analysis/ scripts -- verified via a manual runbook against the real
warehouse (see project planning docs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.db import build_spark, read_co_occurrences, read_entities, read_segments

TABLES = ("videos", "segments", "entities", "co_occurrences")


def export_table(spark, table: str, out_dir: Path) -> int:
    if table == "videos":
        df = spark.table("local.db.videos")
    elif table == "segments":
        df = read_segments(spark)
    elif table == "entities":
        df = read_entities(spark)
    elif table == "co_occurrences":
        df = read_co_occurrences(spark)
    else:
        raise ValueError(f"unknown table: {table}")

    pdf = df.toPandas()
    pdf.to_csv(out_dir / f"{table}.csv", index=False)
    return len(pdf)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the Iceberg warehouse to CSV files for Power BI"
    )
    parser.add_argument("--warehouse", type=str, default="data/iceberg")
    parser.add_argument("--out-dir", type=str, default="data/exports")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = build_spark(args.warehouse)
    try:
        for table in TABLES:
            n = export_table(spark, table, out_dir)
            print(f"{table}: {n} rows -> {out_dir / f'{table}.csv'}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
