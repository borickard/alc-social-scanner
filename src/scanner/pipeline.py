"""End-to-end pipeline: recording → segments → transcript → coding → dataset.

The source video and any derived clips/audio are deleted from the working
directory after results are written. Only the JSON/CSV output persists.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from dotenv import load_dotenv

from . import PROMPT_VERSION
from .analyze import analyze_segment
from .schema import VideoRecord
from .segment import segment_recording
from .transcribe import transcribe_segment

log = logging.getLogger(__name__)

ProgressCb = Callable[[str], None]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _records_to_dataframe(records: List[VideoRecord]) -> pd.DataFrame:
    rows = [r.model_dump(mode="json") for r in records]
    # Flatten list-valued cols to "|"-joined strings for CSV friendliness.
    df = pd.DataFrame(rows)
    list_cols = [
        "alcohol_types",
        "brands_detected",
        "container_visible",
        "sponsored_evidence",
        "hashtags",
    ]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: "|".join(v) if isinstance(v, list) else v)
    return df


def run_pipeline(
    video_path: Path,
    out_dir: Path,
    whisper_model: str = "small",
    gemini_model: str = "gemini-2.5-flash",
    keep_video: bool = False,
    progress: Optional[ProgressCb] = None,
) -> tuple[Path, Path, pd.DataFrame]:
    """Run the full pipeline. Returns (csv_path, json_path, dataframe)."""
    load_dotenv()

    def _emit(msg: str) -> None:
        log.info(msg)
        if progress:
            progress(msg)

    out_dir.mkdir(parents=True, exist_ok=True)

    work_root = Path(tempfile.mkdtemp(prefix="alcscan_"))
    log.info("Working in %s", work_root)

    try:
        _emit("Detecting TikTok segments…")
        segments = segment_recording(video_path, work_root / "segments")
        _emit(f"Found {len(segments)} segment(s).")

        records: List[VideoRecord] = []
        for seg in segments:
            _emit(f"[{seg.index}/{len(segments)}] Transcribing ({seg.duration_sec:.1f}s)…")
            transcript = transcribe_segment(seg.path, model_name=whisper_model)

            _emit(f"[{seg.index}/{len(segments)}] Coding with Gemini…")
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
                    model_version=gemini_model,
                    prompt_version=PROMPT_VERSION,
                    processed_at=_now_iso(),
                    **coding.model_dump(),
                )
            )

        df = _records_to_dataframe(records)

        stem = video_path.stem
        csv_path = out_dir / f"{stem}_results.csv"
        json_path = out_dir / f"{stem}_results.json"
        df.to_csv(csv_path, index=False)
        json_path.write_text(
            json.dumps([r.model_dump(mode="json") for r in records], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _emit(f"Wrote {csv_path.name} and {json_path.name}.")

        if not keep_video:
            try:
                video_path.unlink()
                _emit("Source video deleted (no-retention policy).")
            except OSError as e:
                log.warning("Could not delete source video: %s", e)

        return csv_path, json_path, df
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
