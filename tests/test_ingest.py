from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingest import _is_youtube_url, download_audio, expand_playlist, is_playlist_url, peek_video_id


def test_is_playlist_url_true_when_list_param_present() -> None:
    assert is_playlist_url("https://www.youtube.com/playlist?list=PLabc123") is True


def test_is_playlist_url_false_for_plain_watch_url() -> None:
    assert is_playlist_url("https://www.youtube.com/watch?v=abc123") is False


def test_expand_playlist_raises_runtimeerror_for_non_youtube_url_without_calling_ydl() -> None:
    with patch("src.ingest.yt_dlp.YoutubeDL") as mock_ydl:
        with pytest.raises(RuntimeError):
            expand_playlist("https://example.com/playlist?list=x")
        mock_ydl.assert_not_called()


def test_expand_playlist_builds_canonical_watch_urls_from_entry_ids() -> None:
    info = {"entries": [{"id": "aaa111"}, {"id": "bbb222"}, {"id": "ccc333"}]}

    mock_instance = MagicMock()
    mock_instance.extract_info.return_value = info

    mock_ydl_cls = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_instance

    with patch("src.ingest.yt_dlp.YoutubeDL", mock_ydl_cls):
        result = expand_playlist("https://www.youtube.com/playlist?list=PLabc123")

    assert result == [
        "https://www.youtube.com/watch?v=aaa111",
        "https://www.youtube.com/watch?v=bbb222",
        "https://www.youtube.com/watch?v=ccc333",
    ]
    mock_instance.extract_info.assert_called_once_with(
        "https://www.youtube.com/playlist?list=PLabc123", download=False
    )


def test_expand_playlist_uses_extract_flat_and_skips_download() -> None:
    mock_instance = MagicMock()
    mock_instance.extract_info.return_value = {"entries": []}

    mock_ydl_cls = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_instance

    with patch("src.ingest.yt_dlp.YoutubeDL", mock_ydl_cls) as mock_cls:
        expand_playlist("https://www.youtube.com/playlist?list=PLabc123")

    opts = mock_cls.call_args[0][0]
    assert opts["extract_flat"] is True
    assert opts["skip_download"] is True


def test_expand_playlist_skips_entries_missing_id() -> None:
    info = {"entries": [{"id": "aaa111"}, {"title": "no id here"}, {"id": "ccc333"}]}

    mock_instance = MagicMock()
    mock_instance.extract_info.return_value = info

    mock_ydl_cls = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_instance

    with patch("src.ingest.yt_dlp.YoutubeDL", mock_ydl_cls):
        result = expand_playlist("https://www.youtube.com/playlist?list=PLabc123")

    assert result == [
        "https://www.youtube.com/watch?v=aaa111",
        "https://www.youtube.com/watch?v=ccc333",
    ]


def test_expand_playlist_wraps_ydl_failure_in_runtimeerror() -> None:
    mock_instance = MagicMock()
    mock_instance.extract_info.side_effect = Exception("boom")

    mock_ydl_cls = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_instance

    with patch("src.ingest.yt_dlp.YoutubeDL", mock_ydl_cls):
        with pytest.raises(RuntimeError):
            expand_playlist("https://www.youtube.com/playlist?list=PLabc123")


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


def test_peek_video_id_extracts_from_watch_url() -> None:
    assert peek_video_id("https://www.youtube.com/watch?v=abc123") == "abc123"


def test_peek_video_id_extracts_from_youtu_be_short_link() -> None:
    assert peek_video_id("https://youtu.be/abc123") == "abc123"


def test_peek_video_id_returns_none_for_unparseable_url() -> None:
    assert peek_video_id("https://www.youtube.com/watch") is None
    assert peek_video_id("https://example.com/video") is None


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
