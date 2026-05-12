"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alc-scan",
        description="Scan a TikTok screen recording and code alcohol-related content.",
    )
    parser.add_argument("video", type=Path, help="Path to the screen recording (.mov / .mp4)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="Output directory for CSV/JSON (default: ./results)",
    )
    parser.add_argument("--whisper-model", default="small", help="Whisper model size")
    parser.add_argument("--gemini-model", default="gemini-2.0-flash", help="Gemini model name")
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Skip the post-run deletion of the source video (for debugging).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 2

    csv_path, json_path, df = run_pipeline(
        video_path=args.video,
        out_dir=args.out,
        whisper_model=args.whisper_model,
        gemini_model=args.gemini_model,
        keep_video=args.keep_video,
        progress=lambda m: print(m, flush=True),
    )
    print(f"\nWrote {csv_path} ({len(df)} row(s))")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
