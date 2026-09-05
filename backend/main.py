import io
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import psycopg
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
try:
    from .detections import router as detections_router
except ImportError:
    from detections import router as detections_router

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://app_user:change_me@postgres:5432/app_db",
)
MODEL_NAME = os.getenv("YOLO_MODEL", "yolo26n.pt")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="YOLO26n Camera API", version="1.0.0")
app.include_router(detections_router)
_model = None
_model_lock = threading.Lock()


def get_model() -> YOLO:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(MODEL_NAME)
    return _model


def save_detections(request_id: uuid.UUID, image_name: str, width: int, height: int, inference_ms: float, detections: list[dict]) -> None:
    if not detections:
        return
    rows = [
        (
            request_id,
            MODEL_NAME,
            image_name,
            item["class_id"],
            item["class_name"],
            item["confidence"],
            item["box"]["x_min"],
            item["box"]["y_min"],
            item["box"]["x_max"],
            item["box"]["y_max"],
            width,
            height,
            inference_ms,
        )
        for item in detections
    ]
    with psycopg.connect(DATABASE_URL, connect_timeout=2) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO yolo_detections (
                    request_id, model_name, image_name, class_id, class_name,
                    confidence, x_min, y_min, x_max, y_max,
                    image_width, image_height, inference_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    database = "unavailable"
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=2) as connection:
            connection.execute("SELECT 1")
        database = "ready"
    except psycopg.Error:
        pass
    return {"status": "ready", "database": database, "model": MODEL_NAME}


@app.post("/api/detect")
def detect(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Formato de imagen no soportado")

    payload = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="La imagen supera el límite de 10 MB")

    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="La imagen no es válida") from exc

    started = time.perf_counter()
    model = get_model()
    with _model_lock:
        result = model.predict(np.asarray(image), verbose=False)[0]
    total_ms = (time.perf_counter() - started) * 1000

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        x_min, y_min, x_max, y_max = (float(value) for value in box.xyxy[0].tolist())
        detections.append(
            {
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": round(float(box.conf[0]), 5),
                "box": {
                    "x_min": round(x_min, 2),
                    "y_min": round(y_min, 2),
                    "x_max": round(x_max, 2),
                    "y_max": round(y_max, 2),
                },
            }
        )

    request_id = uuid.uuid4()
    database_saved = False
    try:
        save_detections(request_id, file.filename or "camera.jpg", image.width, image.height, total_ms, detections)
        database_saved = bool(detections)
    except psycopg.Error:
        pass

    return {
        "request_id": str(request_id),
        "database_saved": database_saved,
        "image": {"width": image.width, "height": image.height},
        "inference_ms": round(total_ms, 2),
        "detections": detections,
    }
