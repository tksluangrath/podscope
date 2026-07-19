#!/usr/bin/env bash
# Runs src/run.py repeatedly until it exits 0, resuming automatically after
# a crash (e.g. a macOS jetsam kill under memory pressure). Safe to retry:
# already-processed videos are skipped via db.video_exists()/peek_video_id()
# before any download starts, so a restart picks up where the last one left
# off instead of redoing work.
set -euo pipefail

URLS_FILE="${1:-urls.txt}"
MODEL_SIZE="${2:-tiny}"
LLM_MODEL="${3:-llama3.1:8b}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"

export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "=== attempt $attempt/$MAX_ATTEMPTS ==="
    if python src/run.py --urls-file "$URLS_FILE" --model-size "$MODEL_SIZE" --llm-model "$LLM_MODEL"; then
        echo "=== batch completed successfully ==="
        exit 0
    fi
    echo "=== run.py exited non-zero, retrying in 10s ==="
    sleep 10
done

echo "=== gave up after $MAX_ATTEMPTS attempts ==="
exit 1
