from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    latency_ms: float


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    latency_ms: float
