from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from job_ml.service import ArtifactNotFoundError, predict_job


app = FastAPI(
    title="Job Skill and Salary API",
    version="0.1.0",
    description="Predict required skills and monthly salary from a job description.",
)


class JobRequest(BaseModel):
    job_title: str = Field(..., examples=["机器学习算法工程师"])
    city: str = Field(..., examples=["广州"])
    experience: str = Field(..., examples=["1-3年"])
    education: str = Field(..., examples=["本科"])
    job_description: str = Field(
        ...,
        examples=[
            "负责文本分类、特征工程与模型上线，熟悉 Python、PyTorch、机器学习。"
        ],
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Job Skill and Salary API", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    try:
        predict_job(
            {
                "job_title": "机器学习工程师",
                "city": "广州",
                "experience": "1-3年",
                "education": "本科",
                "job_description": "负责机器学习模型训练与部署。",
            }
        )
    except ArtifactNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ok"}


@app.post("/predict")
def predict(request: JobRequest) -> dict[str, object]:
    try:
        return predict_job(request.model_dump())
    except ArtifactNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
