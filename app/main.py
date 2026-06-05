import json
import logging
import time
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from .config import Settings
from .metrics import Metrics
from .model import ModelService
from .preprocessing import InvalidImage, decode_image, preprocess
from .schemas import BatchResponse, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("visionserve")
settings = Settings()
model = ModelService(
    settings.class_names,
    settings.model_version,
    settings.image_size,
    settings.device,
    settings.model_path,
)
metrics = Metrics()
app = FastAPI(title="VisionServe", version="0.1.0")


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    started = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response
    finally:
        logger.info(
            json.dumps(
                {
                    "event": "request",
                    "request_id": request.state.request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
        )


def response_for(output, latency):
    return PredictionResponse(**output.__dict__, model_version=model.version, latency_ms=latency)


async def read_upload(upload: UploadFile):
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if upload.content_type not in allowed:
        raise InvalidImage(f"Content type must be one of: {sorted(allowed)}")
    return decode_image(await upload.read(), settings.max_file_mb * 1024 * 1024)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return model.info()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    started = time.perf_counter()
    try:
        image = await read_upload(file)
        output = model.predict_batch([preprocess(image, model.image_size)])[0]
        latency = (time.perf_counter() - started) * 1000
        metrics.observe(latency, [output])
        logger.info(
            json.dumps(
                {
                    "event": "prediction",
                    "model_version": model.version,
                    "label": output.label,
                    "confidence": output.confidence,
                    "latency_ms": latency,
                }
            )
        )
        return response_for(output, latency)
    except InvalidImage as exc:
        latency = (time.perf_counter() - started) * 1000
        metrics.observe(latency, error=True)
        raise HTTPException(415, str(exc)) from exc
    except Exception:
        latency = (time.perf_counter() - started) * 1000
        metrics.observe(latency, error=True)
        logger.exception(json.dumps({"event": "prediction_error", "model_version": model.version}))
        raise


@app.post("/predict-batch", response_model=BatchResponse)
async def predict_batch(files: list[UploadFile] = File(...)):
    started = time.perf_counter()
    if not files or len(files) > settings.max_batch_size:
        metrics.observe(0, error=True)
        raise HTTPException(400, f"Batch size must be 1..{settings.max_batch_size}")
    try:
        images = [await read_upload(file) for file in files]
        outputs = model.predict_batch([preprocess(image, model.image_size) for image in images])
        latency = (time.perf_counter() - started) * 1000
        metrics.observe(latency, outputs)
        return BatchResponse(
            predictions=[response_for(output, latency / len(outputs)) for output in outputs],
            count=len(outputs),
            latency_ms=latency,
        )
    except InvalidImage as exc:
        metrics.observe((time.perf_counter() - started) * 1000, error=True)
        raise HTTPException(415, str(exc)) from exc
    except Exception:
        metrics.observe((time.perf_counter() - started) * 1000, error=True)
        logger.exception(
            json.dumps({"event": "batch_prediction_error", "model_version": model.version})
        )
        raise


@app.get("/monitoring-summary")
def monitoring_summary():
    return metrics.summary()


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    return metrics.prometheus()
