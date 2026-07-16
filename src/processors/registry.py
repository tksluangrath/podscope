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
    the whole requirement at this project's scale (ponytail-audit finding)."""
    procs = processors if processors is not None else PROCESSORS
    return {p.name: p.process(segments) for p in procs}
