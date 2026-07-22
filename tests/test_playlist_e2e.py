"""Real end-to-end playlist test: live network, multiple real videos, a
running Ollama server. Same convention as tests/test_performance.py's
PODSCOPE_RUN_PERF_BENCHMARK opt-in -- this has no place running unattended
on GitHub-hosted CI (no Ollama, live network dependency, and real per-video
processing time), so it's opt-in only.

No playlist URL is hardcoded: a made-up or guessed real playlist ID is
exactly the kind of external dependency that silently breaks when the
content changes or is taken down, with no way to verify its stability from
this test suite. Set PODSCOPE_E2E_PLAYLIST_URL to a real, small (2-3 video)
playlist you've verified yourself.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PODSCOPE_RUN_E2E") != "1",
    reason="Real end-to-end playlist run (live network + multiple real videos + "
    "live Ollama) -- opt in with PODSCOPE_RUN_E2E=1 and "
    "PODSCOPE_E2E_PLAYLIST_URL=<a small real playlist you've verified>.",
)


@pytest.fixture
def playlist_url():
    url = os.environ.get("PODSCOPE_E2E_PLAYLIST_URL")
    if not url:
        pytest.skip("PODSCOPE_RUN_E2E=1 but PODSCOPE_E2E_PLAYLIST_URL is unset")
    return url


def test_expand_playlist_returns_real_entries(playlist_url):
    from src.ingest import expand_playlist

    urls = expand_playlist(playlist_url)
    assert len(urls) >= 1
    assert all(u.startswith("https://www.youtube.com/watch?v=") for u in urls)


def test_process_playlist_writes_every_episode_to_iceberg(playlist_url):
    from src import db
    from src.ingest import expand_playlist
    from src.run import process_playlist

    expected_ids = {u.rsplit("=", 1)[1] for u in expand_playlist(playlist_url)}

    spark = db.build_spark("data/iceberg")
    try:
        results = process_playlist(playlist_url, spark, model_size="tiny")
        assert len(results) == len(expected_ids)
        failures = [(u, e) for u, e in results if e is not None]
        assert not failures, f"episodes failed: {failures}"

        videos = spark.table("local.db.videos").select("video_id").collect()
        video_ids_in_db = {r["video_id"] for r in videos}
        assert expected_ids <= video_ids_in_db
    finally:
        spark.stop()
