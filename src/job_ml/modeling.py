from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

from job_ml.config import FEATURE_COLUMNS
from job_ml.data import build_skill_text
from job_ml.features import build_salary_pipeline, build_text_vectorizer


def train_skill_model(
    dataframe: pd.DataFrame,
    model_path: str | Path,
    test_size: float = 0.25,
    random_state: int = 42,
    min_label_frequency: int = 2,
) -> dict[str, object]:
    if len(dataframe) < 8:
        raise ValueError("at least 8 rows are required to train the demo model")

    label_counts = Counter(
        label for labels in dataframe["skills_list"] for label in labels
    )
    allowed_labels = {
        label
        for label, count in label_counts.items()
        if count >= min_label_frequency
    }
    training_frame = dataframe.copy()
    training_frame["skills_list"] = training_frame["skills_list"].map(
        lambda labels: [label for label in labels if label in allowed_labels]
    )
    training_frame = training_frame[
        training_frame["skills_list"].map(bool)
    ].reset_index(drop=True)

    labels = MultiLabelBinarizer()
    target = labels.fit_transform(training_frame["skills_list"])
    texts = build_skill_text(training_frame)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    classifier = PipelineWithVectorizer(
        vectorizer=build_text_vectorizer(),
        classifier=OneVsRestClassifier(
            LogisticRegression(
                max_iter=2_000,
                class_weight="balanced",
                random_state=random_state,
            )
        ),
    )
    classifier.fit(x_train, y_train)
    probabilities = classifier.predict_proba(x_test)
    predictions = (probabilities >= 0.5).astype(int)

    metrics: dict[str, object] = {
        "samples": int(len(dataframe)),
        "training_rows": int(len(training_frame)),
        "labels": list(labels.classes_),
        "macro_f1": round(
            f1_score(y_test, predictions, average="macro", zero_division=0), 4
        ),
        "micro_f1": round(
            f1_score(y_test, predictions, average="micro", zero_division=0), 4
        ),
        "macro_precision": round(
            precision_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            4,
        ),
        "macro_recall": round(
            recall_score(y_test, predictions, average="macro", zero_division=0),
            4,
        ),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": classifier,
            "label_binarizer": labels,
        },
        model_path,
    )
    return metrics


def train_salary_model(
    dataframe: pd.DataFrame,
    model_path: str | Path,
    test_size: float = 0.25,
    random_state: int = 42,
    use_lightgbm: bool = True,
) -> dict[str, object]:
    if len(dataframe) < 8:
        raise ValueError("at least 8 rows are required to train the demo model")

    x_train, x_test, y_train, y_test = train_test_split(
        dataframe[FEATURE_COLUMNS],
        dataframe["salary_month_k"],
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_salary_pipeline(use_lightgbm=use_lightgbm)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics: dict[str, object] = {
        "samples": int(len(dataframe)),
        "mae_k": round(mean_absolute_error(y_test, predictions), 4),
        "rmse_k": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 4),
        "r2": round(r2_score(y_test, predictions), 4),
        "model": pipeline.named_steps["model"].__class__.__name__,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline}, model_path)
    return metrics


def write_metrics(metrics: dict[str, object], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class PipelineWithVectorizer:
    """Small sklearn-compatible wrapper kept explicit for readability."""

    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def fit(self, x, y):
        self.classifier.fit(self.vectorizer.fit_transform(x), y)
        return self

    def predict_proba(self, x):
        return self.classifier.predict_proba(self.vectorizer.transform(x))

    def predict(self, x):
        probabilities = self.predict_proba(x)
        return (np.asarray(probabilities) >= 0.5).astype(int)
