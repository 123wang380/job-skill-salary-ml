from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sample_jobs.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts"

SKILL_MODEL_PATH = DEFAULT_MODEL_DIR / "skill_classifier.joblib"
SALARY_MODEL_PATH = DEFAULT_MODEL_DIR / "salary_regressor.joblib"
METRICS_PATH = DEFAULT_MODEL_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "job_title",
    "city",
    "experience",
    "education",
    "job_description",
]

TARGET_COLUMN = "salary_month_k"
