"""Pydantic schema for per-video coding output.

The same schema is sent to Gemini as a response_schema, so the model is
constrained to return well-formed JSON we can directly parse.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AlcoholType(str, Enum):
    beer = "beer"
    wine = "wine"
    spirits = "spirits"
    cocktail = "cocktail"
    hard_seltzer = "hard_seltzer"
    champagne = "champagne"
    unclear = "unclear"


class Container(str, Enum):
    bottle = "bottle"
    can = "can"
    glass = "glass"
    shot = "shot"
    pitcher = "pitcher"
    none = "none"


class Consumption(str, Enum):
    none = "none"
    holding = "holding"
    drinking = "drinking"
    aftermath = "aftermath"


class Setting(str, Enum):
    home = "home"
    bar = "bar"
    restaurant = "restaurant"
    party = "party"
    outdoor = "outdoor"
    unclear = "unclear"


class Framing(str, Enum):
    celebratory = "celebratory"
    casual = "casual"
    humorous = "humorous"
    aspirational = "aspirational"
    warning = "warning"
    educational = "educational"
    critical = "critical"
    unclear = "unclear"


class Sponsored(str, Enum):
    yes_explicit = "yes_explicit"
    yes_implied = "yes_implied"
    no = "no"
    unclear = "unclear"


class VideoCoding(BaseModel):
    """Result Gemini returns for a single TikTok segment."""

    alcohol_present: bool
    alcohol_types: List[AlcoholType] = Field(default_factory=list)
    brands_detected: List[str] = Field(default_factory=list)
    container_visible: List[Container] = Field(default_factory=list)
    consumption_shown: Consumption = Consumption.none
    setting: Setting = Setting.unclear
    framing: Framing = Framing.unclear
    people_count: int = 0
    caption_language: Optional[str] = Field(
        default=None, description="ISO 639-1 of on-screen text, if any"
    )
    sponsored: Sponsored = Sponsored.unclear
    sponsored_evidence: List[str] = Field(default_factory=list)
    caption_text: str = ""
    hashtags: List[str] = Field(default_factory=list)
    notes: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VideoRecord(VideoCoding):
    """Full row written to the dataset: Gemini output + pipeline metadata."""

    video_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    audio_language: Optional[str] = None
    transcript: str = ""
    model_version: str
    prompt_version: str
    processed_at: str
