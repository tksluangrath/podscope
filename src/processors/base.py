"""Shared interface every NLP processor implements, so registry.run_all() can dispatch across them uniformly."""

from abc import ABC, abstractmethod


class NLPProcessor(ABC):
    name: str

    @abstractmethod
    def process(self, segments: list[dict]) -> list[dict]:
        """Input: list of segment dicts (at minimum segment_id, text).
        Output: list of dicts keyed by the same segment_id, with
        processor-specific fields added. Must not mutate the input list
        or its dicts in place — return new dicts."""
