"""FastAPI wrapper for PestCast prediction engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from climate_data_pipeline import predict_pest_expansion


app = FastAPI(title="PestCast Prediction API")


class Sighting(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    timestamp: datetime


class PredictRequest(BaseModel):
    sightings: List[Sighting]
    climate_geojson: Dict[str, Any]


@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    cluster_data = [
        (s.lat, s.lon, s.timestamp.isoformat()) for s in request.sightings
    ]
    return predict_pest_expansion(cluster_data, request.climate_geojson)
