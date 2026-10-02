from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "job_title",
    "company",
    "city",
    "experience",
    "education",
    "salary_text",
    "job_description",
    "skills",
}

SALARY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*[-~—]\s*(\d+(?:\.\d+)?)\s*[Kk千]")
SKILL_SEPARATOR = re.compile(r"\s*[;；|,，、]\s*")


def parse_salary(salary_text: str) -> float:
    """Convert a monthly salary range such as 15-25K to its midpoint."""
    if not isinstance(salary_text, str) or not salary_text.strip():
        raise ValueError("salary_text must be a non-empty string")

    match = SALARY_PATTERN.search(salary_text)
    if not match:
        raise ValueError(f"unsupported salary format: {salary_text}")

    low, high = (float(value) for value in match.groups())
    if low <= 0 or high <= 0 or low > high:
        raise ValueError(f"invalid salary range: {salary_text}")
    return round((low + high) / 2, 2)


def split_skills(skills_text: str) -> list[str]:
    if not isinstance(skills_text, str):
        return []
    return [item.strip() for item in SKILL_SEPARATOR.split(skills_text) if item.strip()]


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_and_clean(path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")

    dataframe = dataframe.copy()
    for column in REQUIRED_COLUMNS:
        dataframe[column] = dataframe[column].map(clean_text)

    dataframe = dataframe[dataframe["job_description"].ne("")].copy()
    dataframe["salary_month_k"] = dataframe["salary_text"].map(parse_salary)
    dataframe["skills_list"] = dataframe["skills"].map(split_skills)
    dataframe = dataframe[dataframe["skills_list"].map(bool)].copy()
    dataframe = dataframe.drop_duplicates(
        subset=["job_title", "company", "city", "job_description"]
    )

    return dataframe.reset_index(drop=True)


def build_skill_text(dataframe: pd.DataFrame) -> pd.Series:
    columns = ["job_title", "city", "experience", "education", "job_description"]
    return dataframe[columns].fillna("").agg(" ".join, axis=1)


def request_to_frame(payload: dict[str, str]) -> pd.DataFrame:
    row = {
        "job_title": clean_text(payload.get("job_title")),
        "city": clean_text(payload.get("city")),
        "experience": clean_text(payload.get("experience")),
        "education": clean_text(payload.get("education")),
        "job_description": clean_text(payload.get("job_description")),
    }
    return pd.DataFrame([row])
