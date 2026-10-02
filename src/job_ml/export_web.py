from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from job_ml.config import DEFAULT_DATA_PATH
from job_ml.data import build_skill_text, load_and_clean


def _round_values(values) -> list[float]:
    return [
        round(float(value), 6) for value in np.asarray(list(values)).ravel()
    ]


def _export_vectorizer(vectorizer: TfidfVectorizer) -> dict[str, object]:
    return {
        "analyzer": "char",
        "ngram_min": vectorizer.ngram_range[0],
        "ngram_max": vectorizer.ngram_range[1],
        "vocabulary": {
            token: int(index) for token, index in vectorizer.vocabulary_.items()
        },
        "idf": _round_values(vectorizer.idf_),
    }


def build_web_model(data_path: str | Path = DEFAULT_DATA_PATH) -> dict[str, object]:
    dataframe = load_and_clean(data_path)
    texts = build_skill_text(dataframe)

    label_counts = Counter(
        label for labels in dataframe["skills_list"] for label in labels
    )
    allowed_labels = {
        label for label, count in label_counts.items() if count >= 2
    }
    filtered_labels = dataframe["skills_list"].map(
        lambda labels: [label for label in labels if label in allowed_labels]
    )

    label_binarizer = MultiLabelBinarizer()
    target = label_binarizer.fit_transform(filtered_labels)

    skill_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True,
        max_features=1_200,
    )
    skill_matrix = skill_vectorizer.fit_transform(texts)
    skill_classifier = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=42,
        )
    )
    skill_classifier.fit(skill_matrix, target)

    salary_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True,
        max_features=1_200,
    )
    salary_matrix = salary_vectorizer.fit_transform(texts)
    salary_model = Ridge(alpha=1.0)
    salary_model.fit(salary_matrix, dataframe["salary_month_k"])

    return {
        "version": 1,
        "dataset_rows": int(len(dataframe)),
        "note": "Synthetic-data baseline model exported for browser inference.",
        "skill_model": {
            "vectorizer": _export_vectorizer(skill_vectorizer),
            "labels": list(label_binarizer.classes_),
            "coefficients": [
                _round_values(estimator.coef_[0])
                for estimator in skill_classifier.estimators_
            ],
            "intercepts": _round_values(
                estimator.intercept_[0]
                for estimator in skill_classifier.estimators_
            ),
            "threshold": 0.5,
        },
        "salary_model": {
            "vectorizer": _export_vectorizer(salary_vectorizer),
            "coefficients": _round_values(salary_model.coef_),
            "intercept": round(float(salary_model.intercept_), 6),
            "minimum_k": 4.0,
            "maximum_k": 50.0,
        },
    }


def export_web_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = "docs/model.json",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_web_model(data_path)
    output_path.write_text(
        json.dumps(model, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a browser-compatible baseline model."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output", default="docs/model.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_web_model(args.data, args.output)
    print(output)


if __name__ == "__main__":
    main()
