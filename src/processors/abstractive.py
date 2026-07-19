"""Abstractive summarization via a local Ollama chat model."""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.processors.base import NLPProcessor

_SYSTEM_PROMPT = (
    "Summarize this transcript segment in 1-2 concise sentences. "
    "The segment is a raw speech-to-text fragment and may be short or start "
    "mid-sentence -- summarize whatever content it has. Output only the "
    "summary itself, with no preamble, disclaimers, or commentary about the "
    "transcript's length or completeness."
)


class AbstractiveSummarizer(NLPProcessor):
    name = "abstractive_summary"

    def __init__(self, llm: BaseChatModel | None = None, model: str = "llama3.2:latest"):
        # ponytail: generous 120s timeout — CPU-bound local generation can
        # easily exceed typical HTTP client defaults. keep_alive=0 releases
        # the model from memory after each call instead of Ollama's default
        # 5-minute hold -- a real multi-hour batch run hit a macOS jetsam
        # kill with Spark/sentence-transformers/spaCy/faster-whisper/Ollama
        # all resident at once. num_ctx/num_predict capped to what a 1-2
        # sentence summary of a short transcript segment actually needs.
        # num_gpu=16 offloads roughly half of an 8B model's 32 transformer
        # layers to the Metal GPU, running the rest on CPU, to keep sustained
        # GPU load down during long batch runs -- approximate, not exact;
        # re-measure with `powermetrics --samplers gpu_power` and retune if
        # still too high or unnecessarily low.
        self.llm = llm if llm is not None else ChatOllama(
            model=model, timeout=120, keep_alive=0, num_ctx=512, num_predict=128,
            num_gpu=16,
        )

    def process(self, segments: list[dict]) -> list[dict]:
        results = []
        for segment in segments:
            prompt = f"{_SYSTEM_PROMPT}\n\n{segment['text']}"
            response = self.llm.invoke(prompt)
            results.append({"segment_id": segment["segment_id"], "summary": response.content})
        return results
