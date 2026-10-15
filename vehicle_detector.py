from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from requests.exceptions import RequestException

BASE_DIR = Path(__file__).resolve().parent
VEHICLE_MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"


class VehicleDetector:
    VEHICLE_CLASSES = {
        2: "CAR",
        3: "MOTORCYCLE",
        5: "BUS",
        7: "TRUCK",
    }

    def __init__(self) -> None:
        self.model = None
        self.model_error = ""
        self.model_source = ""
        try:
            from ultralytics import YOLO

            if VEHICLE_MODEL_PATH.is_file():
                self.model_source = str(VEHICLE_MODEL_PATH)
                self.model = YOLO(self.model_source)
            else:
                self.model_source = "yolov8n.pt (Ultralytics pretrained)"
                self.model = YOLO("yolov8n.pt")
        except (
            ImportError,
            OSError,
            ValueError,
            RuntimeError,
            ConnectionError,
            TimeoutError,
            RequestException,
        ) as exc:
            self.model_error = str(exc)

    def detect(self, image: np.ndarray) -> List[dict]:
        if self.model is None:
            return []
        try:
            results = self.model(
                image,
                imgsz=480,
                conf=0.25,
                classes=list(self.VEHICLE_CLASSES),
                max_det=5,
                verbose=False,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self.model_error = str(exc)
            return []

        detections: List[dict] = []
        for result in results:
            if not result.boxes:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append({
                    "x1": int(box.xyxy[0][0]),
                    "y1": int(box.xyxy[0][1]),
                    "x2": int(box.xyxy[0][2]),
                    "y2": int(box.xyxy[0][3]),
                    "confidence": float(box.conf[0]),
                    "label": self.VEHICLE_CLASSES[class_id],
                })
        return detections

    def draw_boxes(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        output = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = (
                int(detection[key]) for key in ("x1", "y1", "x2", "y2")
            )
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = (
                f"{detection['label']} "
                f"{float(detection['confidence']):.2f}"
            )
            cv2.putText(
                output,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return output
