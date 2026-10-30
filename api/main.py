"""
api/main.py: NeuroVision AI inference API.
 
Run from the project root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from api.schemas import Detection, HealthResponse, PredictResponse
from src.config import cfg
from src.evaluate import load_model
from src.mlflow_utils import start_tracking

load_dotenv(cfg.PROJECT_ROOT / ".env")

MODEL_NAME = os.getenv("MODEL_NAME", "neurovision-yolov8")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
DEFAULT_CONF = 0.25

_predict_lock = threading.Lock() # one inference at a time: YOLO models are not thread-safe

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model ONCE, when the API starts."""
    app.state.model, app.state.info, app.state.error = None, None, None
    try:
        start_tracking()
        app.state.model, app.state.info = load_model(alias=MODEL_ALIAS, model_name=MODEL_NAME)
        print(f"Model loaded: {MODEL_NAME}@{MODEL_ALIAS} (version {app.state.info['version']})")
    except Exception as e:
        app.state.error = str(e)
        print(f"WARNING: model not loaded: {e}")
    yield

app = FastAPI(
    title="NeuroVision AI API",
    description="Brain tumor detection on MRI images with YOLOv8. Research prototype, "
                "not a medical device: every result must be checked by a doctor.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health():
    """Is the API up, and is the model loaded?"""
    info = app.state.info or {}
    loaded = app.state.model is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=info.get("model_name"),
        model_alias=info.get("alias"),
        model_version=info.get("version"),
        error=app.state.error,
    )

def _read_image(file: UploadFile) -> np.adarray:
    """Validate the upload and decode it. Raises HTTPException 400 with a clear message."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in cfg.IMG_EXTS:
        raise HTTPException(400, f"Unsupported format '{suffix or 'none'}'. "
                                 f"Accepted: {', '.join(cfg.IMG_EXTS)}.")
    data = file.file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (maximum {MAX_UPLOAD_MB} MB).")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "The image is unreadable or corrupted.")
    return image

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predect(
    file: UploadFile = File(..., description="MRI image (jpeg, jpg, bmp or png)"),
    conf: float = Query(DEFAULT_CONF, ge=0.0, le=1.0, description="Minimum confidence, between 0 and 1")
):
    """Detect tumors on an MRI image. Returns boxes (pixels), classes and confidence scores."""
    model = app.state.model
    if model is None:
        raise HTTPException(503, f"Model not available: {app.state.error or 'not loaded'}")
    image = _read_image(file)
    height , width = image.shape[:2]

    with _predict_lock:
        result = model.predict(image, conf=conf, image=cfg.IMAGE_SIZE, devic=cfg.DEVICE, verbos=False)[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astypr(int)

    detections = [
        Detection(class_id=int(c), class_name=model.names[int(c)], confidence=round(float(s), 4),
                  x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]))
        for b, s, c in zip(boxes, scores, classes)
    ]

    detections.sort(key=lambda d: d.confidence, reverse=True)

    message = "No tumor detected" if not detections else f"{len(detections)} detection(s)"
    return PredictResponse(message=message, conf_threshold=conf,
                           image_width=width, image_height=height, detections=detections)
