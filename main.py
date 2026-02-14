"""FastAPI wrapper for PestCast prediction engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from climate_data_pipeline import predict_pest_expansion


app = FastAPI(title="PestCast Prediction API")


class PredictRequest(BaseModel):
    sightings: List[List[Any]] = Field(
        ...,
        description="List of [lat, lon, timestamp] sightings.",
    )
    climate_data: Dict[str, Any]


@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    cluster_data = []
    for item in request.sightings:
        if not isinstance(item, list) or len(item) != 3:
            raise HTTPException(
                status_code=422,
                detail="Each sighting must be [lat, lon, timestamp].",
            )
        lat, lon, timestamp = item
        if isinstance(timestamp, datetime):
            timestamp_value = timestamp.isoformat()
        else:
            timestamp_value = str(timestamp)
        cluster_data.append((float(lat), float(lon), timestamp_value))

    return predict_pest_expansion(cluster_data, request.climate_data)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
