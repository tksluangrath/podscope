# Performance benchmarks

Tracks the single-video pipeline's wall-clock and Ollama memory footprint
against a fixed sample video, so before/after numbers across changes are
comparable. Reproduce with:

```bash
python scripts/benchmark.py --baseline <previous-run.json> --json <new-run.json>
```

Sample video: [President Kennedy's Speech at Rice University](https://www.youtube.com/watch?v=WZyRbnpGyzQ)
(`WZyRbnpGyzQ`, 18:15) -- inside the 15-25 min "typical episode" range, chosen
once and fixed for every run. Hardware: MacBook Pro, Apple Silicon, 16GB
unified memory, no self-hosted macOS CI runner -- this benchmark is run
locally and its result committed here, not gated automatically in CI (see
"CI" below).

## 2026-07-22 -- keep_alive fix + model swaps (section-perf)

| Stage | Before | After |
|---|---:|---:|
| Setup (Spark init) | 3.4s | 3.4s |
| Download audio | 1.1s | 1.2s |
| Transcribing | 59.1s | 37.2s |
| NLP processors (extractive, entities, topics) | 3.5s | 2.7s |
| Abstractive summarization (Ollama) | 335.8s | 191.5s |
| Writing to Iceberg | 1.8s | 1.8s |
| **Total** | **404.6s (6:45)** | **237.8s (3:58)** |
| Ollama peak resident | 2.50 GB | 1.70 GB |
| Segments / Entities | 173 / 60 | 203 / 57 |

**5-minute budget (300s): before FAILED by 105s, after PASSES with ~62s margin.**
**Memory ceiling (6.0 GB, enforced in `src/processors/abstractive.py`): both well under.**

### What changed and why

Profiling (see stage table) found abstractive summarization was 83% of total
wall-clock, not transcription or NLP processing -- neither of which needed
touching. Two techniques closed the gap, both real, explicitly-approved
accuracy tradeoffs (not silent model swaps):

1. **Ollama model: `llama3.2:latest` (3.2B) -> `llama3.2:1b` (1.2B).**
   Measured directly, not assumed: generation on this hardware is
   memory-bandwidth-bound, not compute-bound, so the win is ~40% per call
   rather than proportional to parameter count. **Tradeoff: weaker
   abstractive summaries.**
2. **Whisper model: `base` -> `tiny`.** **Tradeoff: more transcription
   errors**, especially uncommon words/accents. Segment count shifts from
   173 to 203 (finer VAD chunking at this model size) -- same order of
   magnitude, not the ~5x structural change a batched-inference approach
   would have caused (see "Rejected" below).

Also landed as part of this pass (already merged separately, folded into
this baseline): `keep_alive=0` -> `keep_alive="2m"` in `AbstractiveSummarizer`,
which stopped a full model reload before every single segment's LLM call.

### Rejected / not adopted

- **More Ollama GPU offload** (`num_gpu` 16 -> 24 -> 33): measured *slower*
  generation (2.5s -> 7s -> 5.2s per call) on this hardware, not faster.
  Left at 16.
- **`faster-whisper`'s `BatchedInferencePipeline`**: 2x faster (57s -> 29s)
  but drops segment count ~5x (173 -> 34) -- a structural granularity
  change that would ripple into every downstream metric (topic
  segmentation, entity density, compression ratio). Not adopted without
  separate, explicit sign-off; flagged here for future consideration.
- **Batching multiple segments into one Ollama call**: zero accuracy cost,
  but only ~60-70s of the needed ~105s gap, and adds real parsing-fragility
  (a small model reliably returning N distinct summaries in order isn't
  guaranteed). Not needed once both model swaps were approved.

### Guardrails (memory, not speed)

- `src/processors/abstractive.py`: `OLLAMA_RESIDENT_CEILING_GB = 6.0`,
  checked every 20 segments via `ollama ps` (no sudo needed, unlike
  `powermetrics`) -- raises `RuntimeError` with a clear message if
  exceeded, rather than silently pressuring the rest of the machine.
- `.env.example`: `OLLAMA_MAX_LOADED_MODELS=1` / `OLLAMA_NUM_PARALLEL=1` --
  operator-set env vars for `ollama serve` itself (our Python client can't
  set these per-request), so a stray concurrent run can't stack up
  multiple resident models. (Observed in practice during this benchmark:
  without these set, `ollama ps` showed both `llama3.2:latest` and
  `llama3.2:1b` resident at once from separate interactive test sessions.)

## CI

No self-hosted macOS runner exists. GitHub-hosted runners are Linux with no
Ollama and no Apple GPU -- a real timing/memory benchmark there would
either fail to find a GPU or silently run a CPU-only path that proves
nothing. `.github/workflows/ci.yml`'s existing `pytest tests/` step runs
`tests/test_correctness_regression.py` on every PR instead: same fixture,
real extractive/entity/topic processors, real metrics computation, checked
against a golden output -- this is what actually regresses if a future perf
change breaks segment counts, entity counts, or metric computation.

The real 5-minute/memory benchmark (`tests/test_performance.py`) is opt-in
locally via `PODSCOPE_RUN_PERF_BENCHMARK=1` on a real Mac with Ollama
running, and its result gets logged here by hand as part of the PR, per
this file.
