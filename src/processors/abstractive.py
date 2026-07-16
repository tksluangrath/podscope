"""Abstractive summarization via a local Ollama chat model."""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.processors.base import NLPProcessor

_SYSTEM_PROMPT = "Summarize this transcript segment in 1-2 concise sentences."


class AbstractiveSummarizer(NLPProcessor):
    name = "abstractive_summary"

    def __init__(self, llm: BaseChatModel | None = None, model: str = "llama3.2:latest"):
        # ponytail: generous 120s timeout — CPU-bound local generation can
        # easily exceed typical HTTP client defaults.
        self.llm = llm if llm is not None else ChatOllama(model=model, timeout=120)

    def process(self, segments: list[dict]) -> list[dict]:
        results = []
        for segment in segments:
            prompt = f"{_SYSTEM_PROMPT}\n\n{segment['text']}"
            response = self.llm.invoke(prompt)
            results.append({"segment_id": segment["segment_id"], "summary": response.content})
        return results
