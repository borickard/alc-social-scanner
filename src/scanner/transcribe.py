"""Audio transcription with Whisper (local, runs on CPU)."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import whisper

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    language: str | None  # ISO 639-1 or None


@lru_cache(maxsize=1)
def _load_model(name: str):
    log.info("Loading Whisper model: %s", name)
    return whisper.load_model(name)


def _extract_audio(video: Path, wav: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(wav),
    ]
    subprocess.run(cmd, check=True)


def transcribe_segment(video: Path, model_name: str = "small") -> Transcript:
    """Transcribe a clip. Returns empty transcript with None language on silent clips."""
    wav = video.with_suffix(".wav")
    _extract_audio(video, wav)
    try:
        model = _load_model(model_name)
        result = model.transcribe(str(wav), fp16=False)
        text = (result.get("text") or "").strip()
        lang = result.get("language")
        return Transcript(text=text, language=lang if text else None)
    finally:
        wav.unlink(missing_ok=True)
