"""Gradio web UI.

Drop in a recording, click Scan, watch coded rows appear, download CSV/JSON.
"""

from __future__ import annotations

import base64
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


CSS = """
#scan-table { font-size: 13px; }
#scan-table table { table-layout: auto; }
#scan-table td:has(img) { padding: 4px; }
#scan-table img { display: block; height: 110px; border-radius: 4px; }
"""

# Column order for the on-screen table. CSV/JSON still contain every field.
TABLE_COLUMNS = [
    ("thumbnail", "html"),
    ("video_index", "number"),
    ("creator", "str"),
    ("alcohol_present", "bool"),
    ("alcohol_types", "str"),
    ("brands_detected", "str"),
    ("consumption_shown", "str"),
    ("setting", "str"),
    ("framing", "str"),
    ("sponsored", "str"),
    ("likes", "number"),
    ("comments", "number"),
    ("bookmarks", "number"),
    ("shares", "number"),
    ("audio_language", "str"),
    ("caption_language", "str"),
    ("confidence", "number"),
]
TABLE_HEADERS = [c[0] for c in TABLE_COLUMNS]
TABLE_DATATYPES = [c[1] for c in TABLE_COLUMNS]


def _thumb_html(path_str: Optional[str]) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<img src="data:image/jpeg;base64,{data}" alt="frame">'


def _display_frame(records):
    if not records:
        return None
    df = records_to_dataframe(records)
    df["creator"] = df.apply(
        lambda r: (r.get("display_name") or r.get("username") or "") or "",
        axis=1,
    )
    df["thumbnail"] = df["thumbnail_path"].apply(_thumb_html)
    cols = [c for c in TABLE_HEADERS if c in df.columns]
    return df[cols]


def _scan(video_file, whisper_model, gemini_model):
    if video_file is None:
        raise gr.Error("Please upload a screen recording first.")

    src = Path(video_file)
    work = Path(tempfile.mkdtemp(prefix="alcscan_ui_"))
    staged = work / src.name
    shutil.copy2(src, staged)

    out_dir = work / "results"

    table = None
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


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Alcohol Social Scanner — POC") as demo:
        gr.Markdown(
            "# Alcohol Social Scanner — POC\n"
            "Upload a TikTok screen recording. Each detected video becomes a coded row: "
            "alcohol presence, type, brands, framing, sponsorship, language, creator, "
            "and engagement counts. The source video is deleted after processing."
        )

        with gr.Row():
            with gr.Column(scale=2):
                video_in = gr.Video(
                    label="Screen recording",
                    sources=["upload"],
                    height=220,
                )
            with gr.Column(scale=1, min_width=240):
                whisper_choice = gr.Dropdown(
                    choices=["base", "small", "large-v3"],
                    value="small",
                    label="Whisper",
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
                )
                run_btn = gr.Button("Scan", variant="primary", size="lg")

        status_out = gr.Textbox(label="Status", interactive=False, lines=1)

        table_out = gr.Dataframe(
            label="Coded videos",
            headers=TABLE_HEADERS,
            datatype=TABLE_DATATYPES,
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
