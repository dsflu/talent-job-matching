from pydantic import BaseModel
from typing import Any
from src.api.examples import example_talent, example_talent_2, example_job, example_job_2, example_criteria

class Talent(BaseModel):
    talent_id: str
    languages: list[dict[str, Any]]
    job_roles: list[str]
    seniority: str
    salary_expectation: int
    degree: str

    class Config:
        schema_extra = {
            "example": example_talent
        }

class Job(BaseModel):
    job_id: str
    languages: list[dict[str, Any]]
    job_roles: list[str]
    seniorities: list[str]
    max_salary: int
    min_degree: str

    class Config:
        schema_extra = {
            "example": example_job
        }

class MatchRequest(BaseModel):
    talent: Talent
    job: Job

    class Config:
        schema_extra = {
            "example": {
                "talent": example_talent,
                "job": example_job
            }
        }

class MatchBulkRequest(BaseModel):
    talents: list[Talent]
    jobs: list[Job]
    filter_false_predictions: bool = False

    class Config:
        schema_extra = {
            "example": {
                "talents": [example_talent, example_talent_2],
                "jobs": [example_job, example_job_2],
                "filter_false_predictions": False
            }
        }

class RankAndFilterRequest(BaseModel):
    talent: Talent
    jobs: list[Job]
    criteria: dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "talent": example_talent,
                "jobs": [example_job, example_job_2],
                "criteria": example_criteria
            }
        }

class Criteria(BaseModel):
    salary_expectation: int
    seniority: str

    class Config:
        schema_extra = {
            "example": example_criteria
        }