from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingest import _is_youtube_url, download_audio


def test_is_youtube_url_rejects_non_youtube_hostname() -> None:
    assert _is_youtube_url("https://example.com/video") is False


def test_is_youtube_url_accepts_youtube_com_and_youtu_be() -> None:
    assert _is_youtube_url("https://youtube.com/watch?v=X") is True
    assert _is_youtube_url("https://youtu.be/X") is True
    assert _is_youtube_url("https://www.youtube.com/watch?v=X") is True


def test_download_audio_raises_runtimeerror_for_non_youtube_url_without_calling_ydl() -> None:
    with patch("src.ingest.yt_dlp.YoutubeDL") as mock_ydl:
        with pytest.raises(RuntimeError):
            download_audio("https://example.com/video")
        mock_ydl.assert_not_called()


def test_download_audio_extracts_path_title_video_id_without_hardcoding_extension() -> None:
    info = {"id": "video123", "title": "Test Video"}

    mock_instance = MagicMock()
    mock_instance.extract_info.return_value = info
    mock_instance.prepare_filename.return_value = "data/audio/video123.webm"

    mock_ydl_cls = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_instance

    with patch("src.ingest.yt_dlp.YoutubeDL", mock_ydl_cls):
        result = download_audio(
            "https://www.youtube.com/watch?v=video123", output_dir="data/audio"
        )

    assert result == ("data/audio/video123.webm", "Test Video", "video123")
