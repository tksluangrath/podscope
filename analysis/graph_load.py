"""Load entity co-occurrence data into Neo4j for graph visualization. Run
manually/separately from src/run.py and analysis/co_occurrence.py, after
co_occurrences has been computed -- reads the same Iceberg tables and
mirrors them into Neo4j as (:Entity)-[:CO_OCCURS_WITH]->(:Entity), purely
for visualization in Neo4j Browser. No automated tests for this file by
design, same as co_occurrence.py -- verified via a manual runbook against
a real warehouse + a running Neo4j container.
"""
from __future__ import annotations

import argparse

from neo4j import Driver, GraphDatabase
from pyspark.sql import DataFrame

from src.db import build_spark, read_co_occurrences, read_entities


def load_entities(driver: Driver, entities_df: DataFrame) -> int:
    """MERGE one :Entity node per distinct entity_text. entity_text is the
    join key co_occurrences already uses, so it's the node identity here
    too. dropDuplicates keeps an arbitrary entity_type when the same text
    was tagged under more than one type across segments -- fine for a
    visualization label, not used for any downstream matching.
    """
    rows = entities_df.select("entity_text", "entity_type").dropDuplicates(["entity_text"]).collect()
    rows = [{"text": r["entity_text"], "type": r["entity_type"]} for r in rows]
    with driver.session() as session:
        session.run(
            "UNWIND $rows AS row "
            "MERGE (e:Entity {text: row.text}) "
            "SET e.type = row.type",
            rows=rows,
        )
    return len(rows)


def load_co_occurrences(driver: Driver, pairs_df: DataFrame) -> int:
    """MERGE one CO_OCCURS_WITH relationship per (entity_a, entity_b) pair.
    Requires load_entities to have run first so both endpoints exist.
    """
    rows = [row.asDict() for row in pairs_df.collect()]
    with driver.session() as session:
        session.run(
            "UNWIND $rows AS row "
            "MATCH (a:Entity {text: row.entity_a}), (b:Entity {text: row.entity_b}) "
            "MERGE (a)-[r:CO_OCCURS_WITH]->(b) "
            "SET r.video_count = row.video_count, r.segment_count = row.segment_count",
            rows=rows,
        )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the entity co-occurrence graph into Neo4j")
    parser.add_argument("--warehouse", type=str, default="data/iceberg")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="podscope-local")
    args = parser.parse_args()

    spark = build_spark(args.warehouse)
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        entity_count = load_entities(driver, read_entities(spark))
        pair_count = load_co_occurrences(driver, read_co_occurrences(spark))
        print(f"Loaded {entity_count} entities and {pair_count} co-occurrence relationships into Neo4j.")
    finally:
        driver.close()
        spark.stop()


if __name__ == "__main__":
    main()
