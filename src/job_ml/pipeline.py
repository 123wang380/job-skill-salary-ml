from __future__ import annotations

import argparse
import json
from pathlib import Path

from job_ml.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    METRICS_PATH,
    SALARY_MODEL_PATH,
    SKILL_MODEL_PATH,
)
from job_ml.data import load_and_clean
from job_ml.modeling import train_salary_model, train_skill_model, write_metrics


def train_all(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    use_lightgbm: bool = True,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    dataframe = load_and_clean(data_path)

    skill_metrics = train_skill_model(
        dataframe,
        output_dir / SKILL_MODEL_PATH.name,
    )
    salary_metrics = train_salary_model(
        dataframe,
        output_dir / SALARY_MODEL_PATH.name,
        use_lightgbm=use_lightgbm,
    )
    metrics = {
        "dataset": str(Path(data_path)),
        "rows": int(len(dataframe)),
        "skill_classification": skill_metrics,
        "salary_regression": salary_metrics,
    }
    write_metrics(metrics, output_dir / METRICS_PATH.name)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the job skill and salary baseline models."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--no-lightgbm",
        action="store_true",
        help="Use a sklearn RandomForestRegressor instead of LightGBM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_all(
        data_path=args.data,
        output_dir=args.output_dir,
        use_lightgbm=not args.no_lightgbm,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
