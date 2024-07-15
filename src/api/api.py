from fastapi import FastAPI
from typing import Any
from src.api.search import Search
from src.api.schemas import MatchRequest, MatchBulkRequest, RankAndFilterRequest

app = FastAPI(title="Talent Job Matching API")
search = Search(model_config_path="src/config/model_logistic_regression.yaml")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Talent Job Matching API"}

@app.post("/match", response_model=dict)
def match(request: MatchRequest):
    result = search.match(request.talent.model_dump(), request.job.model_dump())
    return result

@app.post("/match_bulk", response_model=list[dict])
def match_bulk(request: MatchBulkRequest):
    talents_list = [talent.model_dump() for talent in request.talents]
    jobs_list = [job.model_dump() for job in request.jobs]
    results = search.match_bulk(talents_list, jobs_list, request.filter_false_predictions)
    return results

@app.post("/rank_and_filter", response_model=list[dict])
def rank_and_filter(request: RankAndFilterRequest):
    jobs_list = [job.model_dump() for job in request.jobs]
    results = search.rank_and_filter(request.talent.model_dump(), jobs_list, request.criteria)
    return results