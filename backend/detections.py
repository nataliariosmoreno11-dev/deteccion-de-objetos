import os
import uuid
from datetime import datetime

import psycopg
from fastapi import APIRouter, HTTPException, Query
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://app_user:change_me@postgres:5432/app_db",
)

router = APIRouter(prefix="/api/detections", tags=["detections"])


class BoundingBox(BaseModel):
    x_min: float = Field(ge=0)
    y_min: float = Field(ge=0)
    x_max: float = Field(ge=0)
    y_max: float = Field(ge=0)


class DetectionInput(BaseModel):
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0, le=1)
    box: BoundingBox


class DetectionBatchInput(BaseModel):
    request_id: uuid.UUID | None = None
    model_name: str = Field(default="yolo26n.pt", min_length=1, max_length=200)
    image_name: str = Field(min_length=1, max_length=500)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    inference_ms: float | None = Field(default=None, ge=0)
    detections: list[DetectionInput]


class SavedBatch(BaseModel):
    request_id: uuid.UUID
    saved: int


class DetectionRecord(BaseModel):
    id: int
    request_id: uuid.UUID
    model_name: str
    image_name: str
    class_id: int
    class_name: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    image_width: int | None
    image_height: int | None
    inference_ms: float | None
    created_at: datetime


@router.post("", response_model=SavedBatch, status_code=201)
def create_detections(batch: DetectionBatchInput) -> SavedBatch:
    request_id = batch.request_id or uuid.uuid4()
    rows = []
    for item in batch.detections:
        if item.box.x_max < item.box.x_min or item.box.y_max < item.box.y_min:
            raise HTTPException(status_code=422, detail="Las coordenadas máximas deben ser mayores que las mínimas")
        rows.append(
            (
                request_id,
                batch.model_name,
                batch.image_name,
                item.class_id,
                item.class_name,
                item.confidence,
                item.box.x_min,
                item.box.y_min,
                item.box.x_max,
                item.box.y_max,
                batch.image_width,
                batch.image_height,
                batch.inference_ms,
            )
        )

    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as connection:
            if rows:
                connection.executemany(
                    """
                    INSERT INTO yolo_detections (
                        request_id, model_name, image_name, class_id, class_name,
                        confidence, x_min, y_min, x_max, y_max,
                        image_width, image_height, inference_ms
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="PostgreSQL no está disponible") from exc

    return SavedBatch(request_id=request_id, saved=len(rows))


@router.get("", response_model=list[DetectionRecord])
def list_detections(limit: int = Query(default=100, ge=1, le=500)) -> list[dict]:
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3, row_factory=dict_row) as connection:
            return connection.execute(
                """
                SELECT id, request_id, model_name, image_name, class_id, class_name,
                       confidence, x_min, y_min, x_max, y_max,
                       image_width, image_height, inference_ms, created_at
                FROM yolo_detections
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="PostgreSQL no está disponible") from exc
