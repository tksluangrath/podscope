"""Download audio from a YouTube URL using yt-dlp's Python API."""

from __future__ import annotations

import os
from urllib.parse import parse_qs, urlparse

import yt_dlp

_ALLOWED_HOSTS = ("youtube.com", "youtu.be")


def _is_youtube_url(url: str) -> bool:
    """Check hostname (not substring) against the YouTube allowlist."""
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host in _ALLOWED_HOSTS or any(host.endswith("." + h) for h in _ALLOWED_HOSTS)


def is_playlist_url(url: str) -> bool:
    """True if url looks like a YouTube playlist URL (has a list= query
    param). The one place this rule is defined -- src/run.py and
    src/tui.py both call this instead of each keeping their own copy.
    """
    return "list=" in url


def peek_video_id(url: str) -> str | None:
    """Best-effort extraction of a video_id straight from the URL, with no
    network call -- lets a caller skip a batch's already-processed videos
    without paying for a full download first. Returns None on any URL shape
    this can't confidently parse (caller should fall back to downloading and
    letting download_audio's own info-dict id be authoritative).
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in ("youtu.be",) or host.endswith(".youtu.be"):
        return parsed.path.lstrip("/") or None
    v = parse_qs(parsed.query).get("v")
    return v[0] if v else None


def download_audio(url: str, output_dir: str = "data/audio") -> tuple[str, str, str]:
    """Download the best available audio stream for a YouTube URL.

    Returns:
        Tuple of (audio_path, title, video_id).

    Raises:
        RuntimeError: if the URL is not a YouTube URL, or if yt-dlp fails.
    """
    if not _is_youtube_url(url):
        raise RuntimeError(f"Not a YouTube URL: {url}")

    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info["id"]
            title = info.get("title", "")
            audio_path = ydl.prepare_filename(info)
    except Exception as e:
        raise RuntimeError(f"Failed to download audio from {url}: {e}") from e

    return audio_path, title, video_id


def expand_playlist(url: str) -> list[str]:
    """Expand a YouTube playlist URL into its individual video URLs, without
    downloading any audio. extract_flat skips resolving each entry through
    a full extractor pass -- just enough metadata (each entry's id) to build
    canonical watch URLs, which is all download_audio/peek_video_id need.

    Returns:
        Ordered list of https://www.youtube.com/watch?v=<id> URLs.

    Raises:
        RuntimeError: if the URL is not a YouTube URL, or if yt-dlp fails.
    """
    if not _is_youtube_url(url):
        raise RuntimeError(f"Not a YouTube URL: {url}")

    ydl_opts = {"quiet": True, "extract_flat": True, "skip_download": True}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise RuntimeError(f"Failed to expand playlist {url}: {e}") from e

    entries = info.get("entries") or []
    return [f"https://www.youtube.com/watch?v={e['id']}" for e in entries if e.get("id")]
