"""api/schemas.py: shape of the JSON returned by the API (shown in /docs)."""
from typing import List

from pydantic import BaseModel, Field

class Detection(BaseModel):
    class_id: int = Field(..., description="Class index (see data.yaml)")
    class_name: str = Field(..., description="Tumor class, e.g. Glioma")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence between 0 and 1")
    x1: int = Field(..., description="Left edge of the box, in pixels")
    y1: int = Field(..., description="Top edge of the box, in pixels")
    x2: int = Field(..., description="Right edge of the box, in pixels")
    y2: int = Field(..., description="Bottom edge of the box, in pixels")

class PredictResponse(BaseModel):
    message: str = Field(..., description="Human readable summary, e.g. 'No tumor detected'")
    conf_threshold: float
    image_width: int
    image_height: int
    detections: List[Detection]

class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if the model is loaded, otherwise 'degraded'")
    model_loaded: bool
    model_name: str | None = None
    model_alias: str | None = None
    model_version: str | None = None
    error: str | None = None
