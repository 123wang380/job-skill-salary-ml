import json
from pathlib import Path

from job_ml.config import DEFAULT_DATA_PATH
from job_ml.export_web import export_web_model


def test_export_web_model_writes_browser_artifacts(tmp_path: Path):
    output = export_web_model(DEFAULT_DATA_PATH, tmp_path / "model.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["version"] == 1
    assert payload["skill_model"]["labels"]
    assert payload["skill_model"]["vectorizer"]["vocabulary"]
    assert len(payload["salary_model"]["coefficients"]) > 0
