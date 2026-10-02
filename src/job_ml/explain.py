from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from job_ml.config import DEFAULT_DATA_PATH, SALARY_MODEL_PATH
from job_ml.data import load_and_clean


def explain_salary_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = SALARY_MODEL_PATH,
    output_path: str | Path = "reports/salary_feature_importance.csv",
    sample_size: int = 100,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError as error:
        raise RuntimeError(
            "Install the optional ML dependencies first: pip install -e '.[ml]'"
        ) from error

    dataframe = load_and_clean(data_path)
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["features"]
    estimator = pipeline.named_steps["model"]

    transformed = preprocessor.transform(dataframe)
    sample = transformed[: min(sample_size, transformed.shape[0])]
    explainer = shap.TreeExplainer(estimator)
    values = np.asarray(explainer.shap_values(sample))
    if values.ndim == 3:
        values = values.mean(axis=2)
    importance = np.abs(values).mean(axis=0)
    feature_names = preprocessor.get_feature_names_out()

    result = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate salary SHAP importance.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model", default=str(SALARY_MODEL_PATH))
    parser.add_argument(
        "--output", default="reports/salary_feature_importance.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = explain_salary_model(args.data, args.model, args.output)
    print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
