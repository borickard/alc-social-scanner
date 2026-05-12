"""Gradio web UI.

Drop in a recording, click Scan, browse the coded table, download CSV/JSON.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from .pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def _scan(video_file, whisper_model, gemini_model, progress=gr.Progress()):
    if video_file is None:
        raise gr.Error("Please upload a screen recording first.")

    src = Path(video_file)
    work = Path(tempfile.mkdtemp(prefix="alcscan_ui_"))
    staged = work / src.name
    shutil.copy2(src, staged)

    out_dir = work / "results"

    def _emit(msg: str) -> None:
        progress(0.5, desc=msg)

    csv_path, json_path, df = run_pipeline(
        video_path=staged,
        out_dir=out_dir,
        whisper_model=whisper_model,
        gemini_model=gemini_model,
        keep_video=False,
        progress=_emit,
    )

    display_cols = [
        "video_index",
        "start_sec",
        "end_sec",
        "alcohol_present",
        "alcohol_types",
        "brands_detected",
        "consumption_shown",
        "setting",
        "framing",
        "sponsored",
        "audio_language",
        "caption_language",
        "confidence",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    return df[display_cols], str(csv_path), str(json_path)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Alcohol Social Scanner — POC") as demo:
        gr.Markdown(
            "# Alcohol Social Scanner — POC\n"
            "Upload a TikTok screen recording (≤ 5 videos). Output is a coded table "
            "covering alcohol presence, type, brands, framing, sponsorship, and language. "
            "The source video is deleted after processing."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="Screen recording", sources=["upload"])
                whisper_choice = gr.Dropdown(
                    choices=["base", "small", "large-v3"],
                    value="small",
                    label="Whisper model",
                )
                gemini_choice = gr.Dropdown(
                    choices=[
                        "gemini-2.5-flash",
                        "gemini-2.5-flash-lite",
                        "gemini-2.5-pro",
                        "gemini-2.0-flash",
                    ],
                    value="gemini-2.5-flash",
                    label="Gemini model",
                )
                run_btn = gr.Button("Scan", variant="primary")
            with gr.Column(scale=2):
                table_out = gr.Dataframe(label="Coded videos", wrap=True)
                csv_out = gr.File(label="Download CSV")
                json_out = gr.File(label="Download JSON")

        run_btn.click(
            _scan,
            inputs=[video_in, whisper_choice, gemini_choice],
            outputs=[table_out, csv_out, json_out],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue().launch()


if __name__ == "__main__":
    main()
