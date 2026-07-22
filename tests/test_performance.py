"""Real end-to-end 5-minute-budget / memory-ceiling test.

Calls the actual pipeline (download, transcribe, NLP, Ollama summarization,
Iceberg write) against the fixed sample video -- real network, real
faster-whisper, real Ollama. Skipped by default, same convention as
process_video/process_batch/main in src/run.py: those are "exercised only
by a manual end-to-end run, not pytest" because there's no real coverage
gain from mocking every I/O boundary, and this test has the same shape.

GitHub-hosted CI runners are Linux with no Ollama and no Apple GPU -- this
test would either error finding no GPU or silently run a CPU-only path that
proves nothing about the actual optimization, so it never runs there. It's
opt-in via PODSCOPE_RUN_PERF_BENCHMARK=1 for a real Mac with Ollama running.
The correctness regression in test_correctness_regression.py is what CI
actually gates on.
"""
import os

import pytest

from scripts.benchmark import OLLAMA_RESIDENT_CEILING_GB, WALL_CLOCK_BUDGET_SECONDS, run_benchmark

pytestmark = pytest.mark.skipif(
    os.environ.get("PODSCOPE_RUN_PERF_BENCHMARK") != "1",
    reason="Real end-to-end benchmark (network + faster-whisper + live Ollama) -- "
    "opt in with PODSCOPE_RUN_PERF_BENCHMARK=1 on a real Mac with Ollama running.",
)


@pytest.fixture(scope="module")
def benchmark_result():
    # module-scoped: the two assertions below share one ~5-minute run
    # instead of paying for the full pipeline twice.
    return run_benchmark()


def test_single_video_meets_five_minute_budget(benchmark_result):
    assert benchmark_result["total_wall_clock"] <= WALL_CLOCK_BUDGET_SECONDS, (
        f"total wall-clock {benchmark_result['total_wall_clock']:.1f}s exceeds "
        f"{WALL_CLOCK_BUDGET_SECONDS}s budget"
    )


def test_ollama_stays_under_memory_ceiling(benchmark_result):
    assert benchmark_result["ollama_peak_resident_gb"] <= OLLAMA_RESIDENT_CEILING_GB, (
        f"Ollama peak resident {benchmark_result['ollama_peak_resident_gb']:.2f} GB exceeds "
        f"{OLLAMA_RESIDENT_CEILING_GB} GB ceiling"
    )
