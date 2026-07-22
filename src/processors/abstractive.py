"""Abstractive summarization via a local Ollama chat model."""
import re
import subprocess

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

# ponytail: this Mac has 16GB total unified memory shared by Spark/spaCy/
# sentence-transformers/faster-whisper/Ollama all at once (see the jetsam
# kill this module's keep_alive setting already guards against) -- 6GB
# leaves Ollama comfortable headroom above its ~1.5-2.5GB observed resident
# size for either model below while still leaving the rest of the pipeline
# most of the machine. Checked via `ollama ps`, the same signal
# scripts/benchmark.py uses, since powermetrics needs sudo this environment
# doesn't have.
OLLAMA_RESIDENT_CEILING_GB = 6.0

# ponytail: measured, not assumed -- llama3.2:latest (3.2B) vs llama3.2:1b
# generation is memory-bandwidth-bound on this hardware, not compute-bound,
# so the win is ~33% per call rather than proportional to param count. Real
# tradeoff: weaker abstractive summaries than the 3.2B model produced.
_DEFAULT_MODEL = "llama3.2:1b"


def ollama_resident_gb(model: str) -> float:
    """Best-effort parse of `ollama ps`'s SIZE column for `model`. Returns
    0.0 if the model isn't currently resident (already unloaded, or the
    call raced past its keep_alive window) -- that's a legitimate reading,
    not a failure to be swallowed.
    """
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return 0.0
    for line in out.splitlines():
        # exact-match the NAME column (split on whitespace) -- matching by
        # prefix alone (e.g. "llama3.2") would conflate different tags of
        # the same model family (llama3.2:1b vs llama3.2:latest) if more
        # than one happens to be resident at once, which is exactly the
        # multi-model-stacking risk this guardrail exists to catch.
        if line.split()[:1] == [model]:
            m = re.search(r"([\d.]+)\s*GB", line)
            if m:
                return float(m.group(1))
    return 0.0


class AbstractiveSummarizer(NLPProcessor):
    name = "abstractive_summary"

    def __init__(self, llm: BaseChatModel | None = None, model: str = _DEFAULT_MODEL):
        # ponytail: generous 120s timeout — CPU-bound local generation can
        # easily exceed typical HTTP client defaults. keep_alive="2m" holds
        # the model resident across one video's back-to-back per-segment
        # calls (seconds apart) instead of reloading it from disk before
        # every single call -- keep_alive=0 was fixing a real multi-hour
        # batch-run jetsam kill (Spark/sentence-transformers/spaCy/
        # faster-whisper/Ollama all resident at once) but paid that cost on
        # every segment of every video, not just between videos. 2 minutes
        # is short enough to still unload during the multi-minute gap
        # between one video's abstractive stage ending and the next video's
        # starting in a batch run, so the original OOM protection holds.
        # num_ctx/num_predict capped to what a 1-2 sentence summary of a
        # short transcript segment actually needs. num_gpu=16 offloads
        # roughly half of an 8B model's 32 transformer layers to the Metal
        # GPU, running the rest on CPU, to keep sustained GPU load down
        # during long batch runs -- measured, not just approximate: raising
        # num_gpu to 24 or 33 made generation *slower* (2.5s -> 7s -> 5.2s
        # per call) on this hardware, so 16 is the tested-best value here,
        # not just a conservative guess. Re-measure with
        # `powermetrics --samplers gpu_power` (needs sudo) if this pipeline
        # ever runs on different hardware.
        self.model = model
        self.llm = llm if llm is not None else ChatOllama(
            model=model, timeout=120, keep_alive="2m", num_ctx=512, num_predict=128,
            num_gpu=16,
        )

    def process(self, segments: list[dict]) -> list[dict]:
        results = []
        for i, segment in enumerate(segments):
            # ponytail: cheap guardrail, not a speed technique -- checked
            # every 20 calls (a `ollama ps` subprocess call is ~tens of ms,
            # negligible next to the ~1-2s per LLM call) so a runaway
            # resident size fails loudly with a clear message instead of
            # silently pressuring the rest of the machine for the other
            # ~690 segments of a long video.
            if i % 20 == 0:
                resident = ollama_resident_gb(self.model)
                if resident > OLLAMA_RESIDENT_CEILING_GB:
                    raise RuntimeError(
                        f"Ollama resident memory ({resident:.2f} GB) exceeded the "
                        f"{OLLAMA_RESIDENT_CEILING_GB} GB ceiling while summarizing "
                        f"segment {i}/{len(segments)} -- stopping before it pressures "
                        "the rest of the machine. Lower OLLAMA_RESIDENT_CEILING_GB's "
                        "assumptions or switch to a smaller model."
                    )
            prompt = f"{_SYSTEM_PROMPT}\n\n{segment['text']}"
            response = self.llm.invoke(prompt)
            results.append({"segment_id": segment["segment_id"], "summary": response.content})
        return results
