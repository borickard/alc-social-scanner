"""Split a screen recording into per-TikTok segments.

PySceneDetect's content-aware detector catches the visual jump that happens
when the user swipes to the next TikTok. For the POC we cap segments at 5 so a
single recording can't blow up the API budget or processing time.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from scenedetect import ContentDetector, SceneManager, open_video

log = logging.getLogger(__name__)

MAX_SEGMENTS = 100
MIN_SEGMENT_SECONDS = 1.5


@dataclass
class Segment:
    index: int
    start_sec: float
    end_sec: float
    path: Path

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def detect_scene_boundaries(video_path: Path, threshold: float = 27.0) -> List[tuple[float, float]]:
    """Return list of (start, end) seconds for each detected scene."""
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()
    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]


def extract_thumbnail(
    clip: Path, dst: Path, at_sec: float = 1.0, height: int = 180
) -> None:
    """Save a single-frame JPEG thumbnail from the clip. Caller picks the path."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{at_sec:.3f}",
        "-i",
        str(clip),
        "-frames:v",
        "1",
        "-vf",
        f"scale=-2:{height}",
        "-q:v",
        "4",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _cut_clip(src: Path, dst: Path, start: float, end: float) -> None:
    """Cut [start, end) out of src into dst using stream copy (fast, no re-encode)."""
    duration = end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def segment_recording(video_path: Path, work_dir: Path) -> List[Segment]:
    """Split video_path into up to MAX_SEGMENTS clips written under work_dir.

    Falls back to a single-segment "whole video" pass if scene detection finds
    nothing usable — better to code one row than zero.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    boundaries = detect_scene_boundaries(video_path)
    log.info("Detected %d raw scenes in %s", len(boundaries), video_path.name)

    # Drop too-short fragments (UI flicker, swipe transitions themselves).
    boundaries = [(s, e) for s, e in boundaries if (e - s) >= MIN_SEGMENT_SECONDS]

    if not boundaries:
        log.warning("No usable scenes detected; treating whole recording as one segment.")
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = float(probe.stdout.strip())
        boundaries = [(0.0, duration)]

    boundaries = boundaries[:MAX_SEGMENTS]

    segments: List[Segment] = []
    for i, (start, end) in enumerate(boundaries, start=1):
        clip_path = work_dir / f"segment_{i:02d}.mp4"
        _cut_clip(video_path, clip_path, start, end)
        segments.append(Segment(index=i, start_sec=start, end_sec=end, path=clip_path))
        log.info("Segment %d: %.2fs–%.2fs (%.2fs)", i, start, end, end - start)

    return segments
