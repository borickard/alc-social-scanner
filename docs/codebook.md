# Codebook

Variables produced per detected TikTok video. Categorical variables use fixed enums so output is directly importable into R/SPSS/Stata.

| Variable | Type | Values / Notes |
|---|---|---|
| `video_index` | int | 1-based index within the recording |
| `start_sec` | float | Segment start time in source recording |
| `end_sec` | float | Segment end time in source recording |
| `duration_sec` | float | `end_sec - start_sec` |
| `alcohol_present` | bool | Any alcohol-related visual, audio, or text content |
| `alcohol_types` | list[enum] | `beer`, `wine`, `spirits`, `cocktail`, `hard_seltzer`, `champagne`, `unclear` |
| `brands_detected` | list[str] | Free text; standardise downstream against a reference list |
| `container_visible` | list[enum] | `bottle`, `can`, `glass`, `shot`, `pitcher`, `none` |
| `consumption_shown` | enum | `none`, `holding`, `drinking`, `aftermath` |
| `setting` | enum | `home`, `bar`, `restaurant`, `party`, `outdoor`, `unclear` |
| `framing` | enum | `celebratory`, `casual`, `humorous`, `aspirational`, `warning`, `educational`, `critical`, `unclear` |
| `people_count` | int | Approximate count of people visible |
| `audio_language` | str | ISO 639-1 from Whisper |
| `caption_language` | str | ISO 639-1 from on-screen text |
| `sponsored` | enum | `yes_explicit`, `yes_implied`, `no`, `unclear` |
| `sponsored_evidence` | list[str] | e.g. `"#ad"`, `"Paid partnership label"`, `"discount code"` |
| `caption_text` | str | Verbatim on-screen caption |
| `hashtags` | list[str] | Extracted from caption |
| `transcript` | str | Whisper transcript |
| `notes` | str | Model's free-text rationale |
| `confidence` | float | 0–1, model's self-assessed confidence |
| `model_version` | str | e.g. `gemini-2.0-flash` |
| `prompt_version` | str | e.g. `v0.1` |
| `processed_at` | str | ISO timestamp |

## Sponsorship detection

Three layers feed `sponsored`:

1. **Explicit textual disclosure** — `#ad`, `#sponsored`, `#paidpartnership`, `#gifted`, `#collab`, "anuncio", "reklam" (Swedish). Found via OCR / caption text.
2. **Platform-applied label** — TikTok's "Paid partnership with X" UI element. Identified by the vision model.
3. **Soft signals** — discount codes, "link in bio", giveaway language, conspicuous repeat brand placement.

Layer 1 or 2 → `yes_explicit`. Layer 3 only → `yes_implied`. Neither → `no`. Ambiguous → `unclear`.
