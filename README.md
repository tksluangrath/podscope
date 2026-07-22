# Podscope

A local, zero-cost multi-video NLP analysis pipeline. Point it at YouTube
videos and it transcribes them, runs four independent NLP techniques over
each transcript segment, scores how much each summary technique preserves
or compresses the original meaning, and stores everything in an Apache
Iceberg lakehouse for cross-video entity co-occurrence analysis.

No paid APIs: transcription runs on [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
abstractive summarization runs on a local [Ollama](https://ollama.com) model.

## Pipeline

```mermaid
flowchart LR
    A[YouTube URL] -->|yt-dlp| B[Audio]
    B -->|faster-whisper| C[Transcript segments]
    C --> D1[Extractive summary\nspaCy + PyTextRank]
    C --> D2[Named entities\nspaCy NER]
    C --> D3[Topic segments]
    C --> D4[Abstractive summary\nOllama LLM]
    D1 --> E[Metrics]
    D4 --> E[Metrics]
    E --> F[(Apache Iceberg\nvia PySpark)]
    D2 --> F
    D3 --> F
    F --> G[Cross-video\nco-occurrence analysis]
    G --> H[Power BI dashboard]
```

## Compression ratio algorithm

Each segment's abstractive summary is scored against the original
transcript text by word-count ratio — how much shorter the summary is
than the source it was generated from:

```mermaid
flowchart TD
    S["segment_text\n(original transcript)"] --> WC1["word count\nlen(text.split())"]
    AS["abs_summary\n(Ollama output)"] --> WC2["word count\nlen(summary.split())"]
    WC1 --> R{"original empty?"}
    R -->|yes| N["None"]
    R -->|no| C["compression_ratio =\nword_count(summary) / word_count(original)"]
    WC2 --> C
```

`compression_ratio(original, summary)` in [`src/metrics.py`](src/metrics.py):

```python
def compression_ratio(original: str, summary: str) -> float | None:
    if not original.strip():
        return None
    return len(summary.split()) / len(original.split())
```

A lower ratio means more aggressive compression (e.g. `0.15` = the summary
is 15% the length of the original segment). It's computed only for the
abstractive summary — the extractive summary is a selected sentence lifted
verbatim from the transcript, not a generated compression, so comparing it
the same way isn't meaningful.

This is one of three per-segment quality metrics computed by
`metrics.compute_all()`, alongside `semantic_similarity` (embedding cosine
similarity between the extractive and abstractive summaries, via
`sentence-transformers`) and `textrank_score` (the extractive summary's
PyTextRank sentence score).

## Project layout

```mermaid
flowchart TD
    subgraph src["src/"]
        ingest["ingest.py\nyt-dlp download + hostname validation"]
        transcribe["transcribe.py\nfaster-whisper transcription"]
        metrics["metrics.py\ncompression_ratio, semantic_similarity, textrank_score"]
        db["db.py\nPySpark + Iceberg read/write\n(only file that imports pyspark)"]
        run["run.py\norchestration: ingest -> transcribe -> NLP -> metrics -> db"]
        subgraph processors["src/processors/"]
            base["base.py\nNLPProcessor interface"]
            registry["registry.py\ndispatches all 4 processors"]
            extractive["extractive.py\nspaCy + PyTextRank extractive summary"]
            entities["entities.py\nspaCy NER"]
            topics["topics.py\ntopic segmentation"]
            abstractive["abstractive.py\nOllama-backed abstractive summary"]
        end
    end
    subgraph analysis["analysis/"]
        co_occurrence["co_occurrence.py\ncross-video entity co-occurrence\n(run separately from run.py)"]
    end
    subgraph tests["tests/"]
        testsuite["pytest suite\nmocks all external I/O except db.py's Spark tests"]
    end

    run --> ingest
    run --> transcribe
    run --> registry
    run --> metrics
    run --> db
    registry --> base
    registry --> extractive
    registry --> entities
    registry --> topics
    registry --> abstractive
    db --> co_occurrence
```

## Requirements

- Python 3.11
- Java 17 (Temurin/OpenJDK) — required by PySpark
- [Ollama](https://ollama.com) running locally, with a model pulled
  (default: `llama3.2:latest`)
- `ffmpeg`

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull llama3.2:latest   # if not already pulled
```

## Usage

```bash
python src/run.py --url "https://www.youtube.com/watch?v=<id>"
python src/run.py --urls-file urls.txt

# after processing multiple videos:
python analysis/co_occurrence.py --min-videos 2 --top-n 30
```

### TUI

```bash
pip install -e .
podscope   # or: python -m src.tui
```

A terminal landing screen to paste a URL, watch progress, and browse history --
launches instantly and runs the pipeline above in the background.

## Docker

Runs the pipeline in a container against a host-run Ollama instance:

```bash
YOUTUBE_URL="https://www.youtube.com/watch?v=<id>" docker-compose up
```

## Tests

```bash
pytest tests/
```

CI (`.github/workflows/ci.yml`) runs the full pytest suite and a
`docker build` check on every push/PR to `main`.

## License

MIT
