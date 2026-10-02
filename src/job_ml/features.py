from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_text_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True,
    )


def build_salary_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "description",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=1,
                    sublinear_tf=True,
                ),
                "job_description",
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["city", "experience", "education"],
            ),
        ],
        remainder="drop",
    )


def build_salary_pipeline(use_lightgbm: bool = True) -> Pipeline:
    preprocessor = build_salary_preprocessor()
    if use_lightgbm:
        try:
            from lightgbm import LGBMRegressor

            estimator = LGBMRegressor(
                n_estimators=250,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=3,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                verbosity=-1,
            )
        except ImportError:
            estimator = _random_forest_regressor()
    else:
        estimator = _random_forest_regressor()

    return Pipeline(
        steps=[
            ("features", preprocessor),
            ("model", estimator),
        ]
    )


def _random_forest_regressor():
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
