"""Send a video segment + transcript to Gemini and parse a structured coding row.

We upload the clip via the Files API so Gemini sees frames and audio natively
(it samples at 1 fps internally). The response is constrained to VideoCoding
via response_schema so the parse can't fail on freeform text.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .schema import VideoCoding


def _is_transient(exc: BaseException) -> bool:
    """Retry only on transient failures (network blips, 5xx). Skip 4xx — quota/auth/etc. won't fix themselves."""
    if isinstance(exc, genai_errors.ClientError):
        return False
    return True

log = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """You are a research coder for a public-health study at Karolinska Institutet that quantifies how alcohol appears in young people's TikTok feeds.

For each TikTok video clip you receive, fill in the VideoCoding schema. Be conservative: if a signal isn't clearly present, mark it as unclear / none / false rather than guessing.

Specific guidance:

- alcohol_present: true if alcohol is visible OR audibly referenced OR mentioned in on-screen text. Soda, juice, mocktails, coffee, energy drinks → false.
- brands_detected: only include a brand if you can read its name on a label, can or bottle, or it is explicitly named in audio/caption. Don't infer brand from bottle shape alone.
- consumption_shown: "drinking" = liquid going to mouth; "holding" = visible but not being consumed; "aftermath" = empty containers, intoxication.
- sponsored:
    yes_explicit  -> "#ad", "#sponsored", "#paidpartnership", "#gifted", "anuncio", "reklam", or TikTok's "Paid partnership" UI label, or an explicit verbal disclosure.
    yes_implied   -> discount codes ("use code XYZ"), "link in bio" with a brand, giveaways, conspicuous repeat brand placement without explicit disclosure.
    no            -> no signs of commercial intent.
    unclear       -> ambiguous.
- sponsored_evidence: short strings citing what you saw (e.g. "#ad in caption", "Paid partnership label", "use code SUMMER15").
- caption_language: ISO 639-1 of any on-screen text. Null if no readable text.
- caption_text: verbatim main caption text. Hashtags only in hashtags[].
- people_count: rough count of distinct people visible. 0 if none.
- username: the @handle displayed on the video (usually bottom-left, prefixed with "@"). Include the leading @. Null if not visible.
- display_name: the larger display name shown above or next to the handle, if it differs from the handle. Null otherwise.
- likes / comments / bookmarks / shares: the engagement counts shown on the right-hand side of the TikTok UI (heart / speech bubble / bookmark / arrow icons). Convert TikTok's abbreviated format to a plain integer: "8,262" -> 8262, "1.2K" -> 1200, "3.4M" -> 3400000, "500" -> 500. Null if a count is not visible.
- confidence: your overall confidence in this coding row, 0–1.
- notes: 1–3 sentences of rationale, citing what you observed.
"""

USER_PROMPT_TEMPLATE = """Code the attached TikTok clip.

Whisper transcript of this clip (may be empty or noisy):
\"\"\"
{transcript}
\"\"\"

Audio language detected by Whisper: {audio_language}

Return only the structured VideoCoding JSON."""


def _client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")
    return genai.Client(api_key=key)


def _wait_for_file_active(client: genai.Client, file_name: str, timeout_sec: int = 120) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        f = client.files.get(name=file_name)
        if f.state.name == "ACTIVE":
            return
        if f.state.name == "FAILED":
            raise RuntimeError(f"Gemini file upload failed: {f.name}")
        time.sleep(1.0)
    raise TimeoutError(f"Gemini file did not become ACTIVE within {timeout_sec}s")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=16),
    retry=retry_if_exception(_is_transient),
)
def analyze_segment(
    clip_path: Path,
    transcript: str,
    audio_language: str | None,
    model: str = "gemini-2.5-flash",
) -> VideoCoding:
    """Upload the clip, ask Gemini to fill in VideoCoding, return the parsed row."""
    client = _client()
    log.info("Uploading %s to Gemini Files API…", clip_path.name)
    uploaded = client.files.upload(file=str(clip_path))
    try:
        _wait_for_file_active(client, uploaded.name)
        prompt = USER_PROMPT_TEMPLATE.format(
            transcript=transcript or "(no speech detected)",
            audio_language=audio_language or "unknown",
        )
        log.info("Requesting coding from %s…", model)
        response = client.models.generate_content(
            model=model,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=VideoCoding,
                temperature=0.1,
            ),
        )
        parsed = response.parsed
        if not isinstance(parsed, VideoCoding):
            # Fallback if SDK didn't auto-parse — validate from raw text.
            return VideoCoding.model_validate_json(response.text)
        return parsed
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            log.warning("Could not delete uploaded file %s from Gemini", uploaded.name)
