"""Tests for report exporting service."""

from pathlib import Path

from core.types import DetectionResult
from services.report_service import export_detection_report


def test_export_detection_report(tmp_path):
    result = DetectionResult(
        media_path="sample.jpg",
        media_type="image",
        method="balanced",
        threshold=0.5,
        fake_probability=0.73,
        confidence=0.73,
        is_fake=True,
        duration_seconds=0.42,
        frames_analyzed=1,
        notes=["unit-test"],
        metadata={"foo": "bar"},
    )

    output_file = tmp_path / "report.txt"
    saved = export_detection_report(result, output_file)

    assert saved == output_file
    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "DeepFake Detector Report" in content
    assert "Prediction: FAKE" in content
