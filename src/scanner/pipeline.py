"""End-to-end pipeline: recording → segments → transcript → coding → dataset.

The source video and any derived clips/audio are deleted from the working
directory after results are written. Only the JSON/CSV output persists.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
from dotenv import load_dotenv

from . import PROMPT_VERSION
from .analyze import analyze_segment
from .schema import VideoRecord
from .segment import extract_thumbnail, segment_recording
from .transcribe import transcribe_segment

log = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Snapshot emitted from iter_pipeline after each meaningful step."""

    status: str
    records: List[VideoRecord] = field(default_factory=list)
    csv_path: Optional[Path] = None
    json_path: Optional[Path] = None
    done: bool = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


LIST_COLS = (
    "alcohol_types",
    "brands_detected",
    "container_visible",
    "sponsored_evidence",
    "hashtags",
)


def records_to_dataframe(records: List[VideoRecord]) -> pd.DataFrame:
    """Convert records to a CSV-friendly dataframe (list cols joined with '|')."""
    rows = [r.model_dump(mode="json") for r in records]
    df = pd.DataFrame(rows)
    for col in LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: "|".join(v) if isinstance(v, list) else v)
    return df


def iter_pipeline(
    video_path: Path,
    out_dir: Path,
    whisper_model: str = "small",
    gemini_model: str = "gemini-2.5-flash",
    keep_video: bool = False,
) -> Iterator[PipelineState]:
    """Run the pipeline, yielding a PipelineState after each segment is coded.

    The final yield has `done=True` and populated csv_path / json_path.
    """
    load_dotenv()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="alcscan_"))
    log.info("Working in %s", work_root)

    thumb_dir = out_dir / "thumbnails"

    try:
        yield PipelineState(status="Detecting TikTok segments…")
        segments = segment_recording(video_path, work_root / "segments")
        n = len(segments)
        yield PipelineState(status=f"Found {n} segment(s). Coding…")

        records: List[VideoRecord] = []
        for seg in segments:
            thumb_path = thumb_dir / f"segment_{seg.index:02d}.jpg"
            try:
                # Grab a frame ~1s in to skip the swipe transition.
                offset = min(1.0, max(0.0, seg.duration_sec * 0.25))
                extract_thumbnail(seg.path, thumb_path, at_sec=offset)
            except Exception as e:
                log.warning("Thumbnail extraction failed for segment %d: %s", seg.index, e)
                thumb_path = None

            yield PipelineState(
                status=f"[{seg.index}/{n}] Transcribing ({seg.duration_sec:.1f}s)…",
                records=list(records),
            )
            transcript = transcribe_segment(seg.path, model_name=whisper_model)

            yield PipelineState(
                status=f"[{seg.index}/{n}] Coding with Gemini…",
                records=list(records),
            )
            coding = analyze_segment(
                seg.path,
                transcript=transcript.text,
                audio_language=transcript.language,
                model=gemini_model,
            )

            records.append(
                VideoRecord(
                    video_index=seg.index,
                    start_sec=round(seg.start_sec, 3),
                    end_sec=round(seg.end_sec, 3),
                    duration_sec=round(seg.duration_sec, 3),
                    audio_language=transcript.language,
                    transcript=transcript.text,
                    thumbnail_path=str(thumb_path) if thumb_path else None,
                    model_version=gemini_model,
                    prompt_version=PROMPT_VERSION,
                    processed_at=_now_iso(),
                    **coding.model_dump(),
                )
            )
            yield PipelineState(
                status=f"Coded {seg.index}/{n}.",
                records=list(records),
            )

        df = records_to_dataframe(records)
        stem = video_path.stem
        csv_path = out_dir / f"{stem}_results.csv"
        json_path = out_dir / f"{stem}_results.json"
        df.to_csv(csv_path, index=False)
        json_path.write_text(
            json.dumps([r.model_dump(mode="json") for r in records], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if not keep_video:
            try:
                video_path.unlink()
            except OSError as e:
                log.warning("Could not delete source video: %s", e)

        yield PipelineState(
            status=f"Done. Coded {n} segment(s).",
            records=records,
            csv_path=csv_path,
            json_path=json_path,
            done=True,
        )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def run_pipeline(
    video_path: Path,
    out_dir: Path,
    whisper_model: str = "small",
    gemini_model: str = "gemini-2.5-flash",
    keep_video: bool = False,
    on_status: Optional[callable] = None,
) -> PipelineState:
    """Synchronous wrapper around iter_pipeline. Returns the final state."""
    final: Optional[PipelineState] = None
    for state in iter_pipeline(
        video_path=video_path,
        out_dir=out_dir,
        whisper_model=whisper_model,
        gemini_model=gemini_model,
        keep_video=keep_video,
    ):
        if on_status:
            on_status(state.status)
        final = state
    assert final is not None
    return final
