import pandas as pd

from job_ml.data import parse_salary, split_skills


def test_parse_salary_returns_midpoint():
    assert parse_salary("15-25K") == 20
    assert parse_salary("8~12k") == 10


def test_split_skills_accepts_common_separators():
    assert split_skills("Python; SQL；PyTorch") == ["Python", "SQL", "PyTorch"]


def test_parse_salary_rejects_invalid_range():
    try:
        parse_salary("30-20K")
    except ValueError as error:
        assert "invalid salary range" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_pandas_nan_is_handled_as_empty_skill_list():
    assert split_skills(pd.NA) == []
