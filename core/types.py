"""Core data structures shared across UI and backend services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DetectionResult:
    """Normalized response payload for image/video detection."""

    media_path: str
    media_type: str
    method: str
    threshold: float
    fake_probability: float
    confidence: float
    is_fake: bool
    duration_seconds: float
    frames_analyzed: int = 0
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BatchItemResult:
    """Result for a single file inside batch processing."""

    media_path: str
    status: str
    fake_probability: float | None = None
    confidence: float | None = None
    error: str | None = None
