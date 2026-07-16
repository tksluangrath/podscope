FROM python:3.11-slim

# ponytail: default-jdk-headless, not default-jdk — pulls the same JVM
# without X11/fontconfig GUI deps (hard Depends, --no-install-recommends
# doesn't help), which is >100 fewer packages for a headless pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk-headless ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY src/ src/
COPY analysis/ analysis/

CMD ["python", "src/run.py", "--help"]
