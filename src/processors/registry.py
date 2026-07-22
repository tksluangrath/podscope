"""Central dispatch point: run every registered NLPProcessor over a batch of segments, keyed by processor name."""

from src.processors.base import NLPProcessor
from src.processors.extractive import ExtractiveSummarizer
from src.processors.entities import EntityExtractor
from src.processors.topics import TopicSegmenter
from src.processors.abstractive import AbstractiveSummarizer

PROCESSORS: list[NLPProcessor] = [
    ExtractiveSummarizer(),
    EntityExtractor(),
    TopicSegmenter(),
    AbstractiveSummarizer(),
]


def run_all(
    segments: list[dict],
    processors: list[NLPProcessor] | None = None,
) -> dict[str, list[dict]]:
    """Run every processor in `processors` (defaults to the module-level
    PROCESSORS) over segments, keyed by processor.name. The injectable
    parameter exists so this section's tests can drive fake NLPProcessor
    stubs without importing section-03/04 modules that don't exist yet —
    the production call site (run_all(segments)) is unaffected.
    Do not add a plugin-discovery/config-driven registry — a plain list is
    the whole requirement at this project's scale (ponytail-audit finding).

    One exception to "every processor gets the same raw segments": once
    topic_segmenter has run, its topic_label is merged onto the segments
    passed to abstractive_summary, so that processor can group segments by
    topic instead of summarizing each raw ASR fragment (often a few words,
    no surrounding context) in isolation. Only fires when both names are
    present in `processors` -- fake-processor-only tests are unaffected.
    """
    procs = processors if processors is not None else PROCESSORS
    results: dict[str, list[dict]] = {}
    topics_by_id: dict[int, str] | None = None
    for p in procs:
        input_segments = segments
        if p.name == "abstractive_summary" and topics_by_id is not None:
            input_segments = [
                {**s, "topic_label": topics_by_id.get(s["segment_id"])} for s in segments
            ]
        results[p.name] = p.process(input_segments)
        if p.name == "topic_segmenter":
            topics_by_id = {r["segment_id"]: r["topic_label"] for r in results[p.name]}
    return results
