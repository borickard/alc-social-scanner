# Alcohol Social Scanner

Proof-of-concept research tool for the Karolinska Institutet. Accepts a screen recording of a participant scrolling TikTok and produces a coded dataset describing alcohol-related content per video: type, brand, framing, sponsorship signals, language, etc. The video is never persisted — only the aggregated coding output is kept.

## Status

POC scope: handle a single screen recording containing up to 5 TikTok videos. Coding categories follow a fixed schema (see `docs/codebook.md`).

## How it works

1. **Segment** the recording into per-TikTok clips with PySceneDetect (content-aware mode catches swipe transitions).
2. **Transcribe** the audio of each segment with Whisper (auto language detection).
3. **Analyse** each segment by sending it to Gemini 2.0 Flash with a strict JSON schema covering presence, type, brands, setting, framing, sponsorship, and language.
4. **Export** results to `results.csv` and `results.json`. The source video is deleted from the working directory after processing.

## Setup

Requires Python 3.11+ and `ffmpeg` on PATH.

```bash
pip install -e .
cp .env.example .env  # then edit .env with your Gemini key
```

Get a free Gemini API key at <https://aistudio.google.com/apikey>.

## Usage

**Web UI (recommended):**

```bash
python -m scanner.app
```

Opens a Gradio interface at <http://localhost:7860>. Drag in a `.mov` or `.mp4`, hit Scan, download results.

**CLI:**

```bash
alc-scan path/to/recording.mov --out results/
```

## Privacy

The pipeline processes the video in a temp directory and deletes it after the JSON/CSV is written. Only the structured coding output persists. The participant's video is never uploaded to any server you don't control, except for frame samples sent to Gemini for analysis — for the production version, swap Gemini for an EU-hosted model.

## Research use

This is a POC. Before using it on a real participant dataset:

- Validate against human coders on ~100 videos and report Cohen's κ per variable.
- Pin the model version and prompt version in every output row (already done).
- Document the pipeline in a DPIA covering the Gemini API hop.
