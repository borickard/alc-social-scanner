"""Gradio web UI.

Drop in a recording, click Scan, watch coded rows appear, download CSV/JSON.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr

from .pipeline import iter_pipeline, records_to_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


DISPLAY_COLUMNS = [
    "video_index",
    "username",
    "alcohol_present",
    "alcohol_types",
    "brands_detected",
    "consumption_shown",
    "setting",
    "framing",
    "sponsored",
    "likes",
    "comments",
    "bookmarks",
    "shares",
    "audio_language",
    "caption_language",
    "confidence",
]


def _display_frame(records):
    if not records:
        return None
    df = records_to_dataframe(records)
    cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    return df[cols]


def _scan(video_file, whisper_model, gemini_model):
    if video_file is None:
        raise gr.Error("Please upload a screen recording first.")

    src = Path(video_file)
    work = Path(tempfile.mkdtemp(prefix="alcscan_ui_"))
    staged = work / src.name
    shutil.copy2(src, staged)

    out_dir = work / "results"

    table: Optional[object] = None
    csv_path: Optional[str] = None
    json_path: Optional[str] = None

    for state in iter_pipeline(
        video_path=staged,
        out_dir=out_dir,
        whisper_model=whisper_model,
        gemini_model=gemini_model,
        keep_video=False,
    ):
        if state.records:
            table = _display_frame(state.records)
        if state.csv_path:
            csv_path = str(state.csv_path)
        if state.json_path:
            json_path = str(state.json_path)
        yield state.status, table, csv_path, json_path


CSS = """
#scan-table { font-size: 13px; }
#scan-table table { table-layout: auto; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Alcohol Social Scanner — POC") as demo:
        gr.Markdown(
            "# Alcohol Social Scanner — POC\n"
            "Upload a TikTok screen recording. Output: a coded row per video "
            "covering alcohol presence, type, brands, framing, sponsorship, language, handle, "
            "and engagement. The source video is deleted after processing."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                video_in = gr.Video(
                    label="Screen recording",
                    sources=["upload"],
                    height=320,
                )
                with gr.Row():
                    whisper_choice = gr.Dropdown(
                        choices=["base", "small", "large-v3"],
                        value="small",
                        label="Whisper",
                        scale=1,
                    )
                    gemini_choice = gr.Dropdown(
                        choices=[
                            "gemini-2.5-flash",
                            "gemini-2.5-flash-lite",
                            "gemini-2.5-pro",
                            "gemini-2.0-flash",
                        ],
                        value="gemini-2.5-flash",
                        label="Gemini",
                        scale=1,
                    )
                run_btn = gr.Button("Scan", variant="primary")
                status_out = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                )

            with gr.Column(scale=3):
                table_out = gr.Dataframe(
                    label="Coded videos",
                    wrap=True,
                    elem_id="scan-table",
                    interactive=False,
                )
                with gr.Row():
                    csv_out = gr.DownloadButton(label="Download CSV", variant="secondary")
                    json_out = gr.DownloadButton(label="Download JSON", variant="secondary")

        run_btn.click(
            _scan,
            inputs=[video_in, whisper_choice, gemini_choice],
            outputs=[status_out, table_out, csv_out, json_out],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue().launch(css=CSS)


if __name__ == "__main__":
    main()
