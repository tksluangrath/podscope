"""Download audio from a YouTube URL using yt-dlp's Python API."""

from __future__ import annotations

import os
from urllib.parse import urlparse

import yt_dlp

_ALLOWED_HOSTS = ("youtube.com", "youtu.be")


def _is_youtube_url(url: str) -> bool:
    """Check hostname (not substring) against the YouTube allowlist."""
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host in _ALLOWED_HOSTS or any(host.endswith("." + h) for h in _ALLOWED_HOSTS)


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
