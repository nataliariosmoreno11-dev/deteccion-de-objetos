CREATE TABLE IF NOT EXISTS app_health (
    id BIGSERIAL PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'ready',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO app_health (status) VALUES ('ready');

CREATE TABLE IF NOT EXISTS yolo_detections (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    model_name TEXT NOT NULL DEFAULT 'yolo26n.pt',
    image_name TEXT NOT NULL,
    class_id INTEGER NOT NULL CHECK (class_id >= 0),
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    x_min REAL NOT NULL CHECK (x_min >= 0),
    y_min REAL NOT NULL CHECK (y_min >= 0),
    x_max REAL NOT NULL CHECK (x_max >= x_min),
    y_max REAL NOT NULL CHECK (y_max >= y_min),
    image_width INTEGER CHECK (image_width > 0),
    image_height INTEGER CHECK (image_height > 0),
    inference_ms REAL CHECK (inference_ms >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_yolo_detections_request_id
    ON yolo_detections (request_id);

CREATE INDEX IF NOT EXISTS idx_yolo_detections_created_at
    ON yolo_detections (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_yolo_detections_class_name
    ON yolo_detections (class_name);
