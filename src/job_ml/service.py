from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from job_ml.config import DEFAULT_MODEL_DIR, SALARY_MODEL_PATH, SKILL_MODEL_PATH
from job_ml.data import build_skill_text, request_to_frame


class ArtifactNotFoundError(FileNotFoundError):
    pass


@lru_cache(maxsize=4)
def load_model_bundle(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, object]:
    model_dir = Path(model_dir)
    skill_path = model_dir / SKILL_MODEL_PATH.name
    salary_path = model_dir / SALARY_MODEL_PATH.name

    if not skill_path.exists() or not salary_path.exists():
        raise ArtifactNotFoundError(
            "Model artifacts are missing. Run `python -m job_ml.pipeline` first."
        )

    return {
        "skill": joblib.load(skill_path),
        "salary": joblib.load(salary_path),
    }


def predict_job(
    payload: dict[str, str],
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    threshold: float = 0.5,
) -> dict[str, object]:
    bundles = load_model_bundle(model_dir)
    frame = request_to_frame(payload)

    skill_bundle = bundles["skill"]
    probabilities = np.asarray(
        skill_bundle["pipeline"].predict_proba(build_skill_text(frame))[0]
    )
    labels = list(skill_bundle["label_binarizer"].classes_)
    skill_predictions = [
        {"skill": label, "probability": round(float(probability), 4)}
        for label, probability in zip(labels, probabilities, strict=True)
        if probability >= threshold
    ]
    skill_predictions.sort(key=lambda item: item["probability"], reverse=True)

    salary_bundle = bundles["salary"]
    predicted_salary = float(salary_bundle["pipeline"].predict(frame)[0])

    return {
        "skills": skill_predictions,
        "predicted_salary_month_k": round(predicted_salary, 2),
        "threshold": threshold,
    }
