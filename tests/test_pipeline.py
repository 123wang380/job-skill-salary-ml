from pathlib import Path

from job_ml.config import DEFAULT_DATA_PATH
from job_ml.data import load_and_clean
from job_ml.modeling import train_salary_model, train_skill_model


def test_sample_data_cleans_successfully():
    dataframe = load_and_clean(DEFAULT_DATA_PATH)
    assert len(dataframe) >= 20
    assert dataframe["salary_month_k"].gt(0).all()
    assert dataframe["skills_list"].map(bool).all()


def test_demo_models_train_and_write_artifacts(tmp_path: Path):
    dataframe = load_and_clean(DEFAULT_DATA_PATH)
    skill_metrics = train_skill_model(
        dataframe,
        tmp_path / "skill.joblib",
    )
    salary_metrics = train_salary_model(
        dataframe,
        tmp_path / "salary.joblib",
        use_lightgbm=False,
    )

    assert (tmp_path / "skill.joblib").exists()
    assert (tmp_path / "salary.joblib").exists()
    assert 0 <= skill_metrics["macro_f1"] <= 1
    assert salary_metrics["mae_k"] >= 0
