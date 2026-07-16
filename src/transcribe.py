"""Transcribe audio files with faster-whisper."""

from __future__ import annotations

from faster_whisper import WhisperModel


def transcribe(audio_path: str, model_size: str = "base") -> list[dict]:
    """Transcribe audio_path into a list of segment dicts.

    Each dict has keys: segment_id (local 0-based index), start_time,
    end_time, text (stripped).
    """
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments_gen, _info = model.transcribe(
        audio_path, vad_filter=True, condition_on_previous_text=False
    )
    return [
        {
            "segment_id": i,
            "start_time": seg.start,
            "end_time": seg.end,
            "text": seg.text.strip(),
        }
        for i, seg in enumerate(segments_gen)
    ]
