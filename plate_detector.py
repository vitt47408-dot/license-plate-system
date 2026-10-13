from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plate_detector_yolov8n.pt"


class PlateDetector:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.model_error = ""
        self.mode = "OpenCV contour (fallback)"
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path.exists():
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(self.model_path))
                self.mode = "YOLO plate detector"
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                self.model_error = str(exc)
        else:
            self.model_error = f"Không tìm thấy trọng số YOLO tại {self.model_path}"

    def _detect_with_contours(self, image: np.ndarray) -> List[dict]:
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[dict] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 40 or h < 15:
                continue
            area = w * h
            if area < 0.005 * width * height:
                continue
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < 1.7 or aspect_ratio > 8.0:
                continue
            if x < 0 or y < 0 or x + w > width or y + h > height:
                continue
            candidates.append({
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + w),
                "y2": int(y + h),
                "confidence": None,
                "method": "OpenCV contour (fallback)",
            })

        if not candidates:
            return []
        candidates = sorted(candidates, key=lambda item: (item["x2"] - item["x1"]) * (item["y2"] - item["y1"]), reverse=True)
        best: List[dict] = []
        for candidate in candidates:
            overlap = False
            for existing in best:
                inter_x1 = max(candidate["x1"], existing["x1"])
                inter_y1 = max(candidate["y1"], existing["y1"])
                inter_x2 = min(candidate["x2"], existing["x2"])
                inter_y2 = min(candidate["y2"], existing["y2"])
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                intersection = inter_w * inter_h
                candidate_area = (candidate["x2"] - candidate["x1"]) * (candidate["y2"] - candidate["y1"])
                if intersection / max(candidate_area, 1) > 0.5:
                    overlap = True
                    break
            if not overlap:
                best.append(candidate)
        return best[:3]

    def detect(self, image: np.ndarray) -> List[dict]:
        if self.model is not None:
            try:
                results = self.model(image, imgsz=640, conf=0.25, verbose=False)
                detections: List[dict] = []
                for result in results:
                    if not result.boxes:
                        continue
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        detections.append({
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": conf,
                            "method": self.mode,
                        })
                if detections:
                    return detections
            except (OSError, ValueError, RuntimeError) as exc:
                self.model_error = str(exc)
        return self._detect_with_contours(image)

    def draw_boxes(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        output = image.copy()
        for det in detections:
            x1 = int(det["x1"])
            y1 = int(det["y1"])
            x2 = int(det["x2"])
            y2 = int(det["y2"])
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            confidence = det.get("confidence")
            label = f"plate {confidence:.2f}" if confidence is not None else "plate (contour)"
            cv2.putText(output, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return output

    def crop_plate(self, image: np.ndarray, detection: dict) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        box_width = max(1, int(detection["x2"]) - int(detection["x1"]))
        box_height = max(1, int(detection["y2"]) - int(detection["y1"]))
        pad_x = round(box_width * 0.06)
        pad_y = round(box_height * 0.08)
        x1 = max(0, int(detection["x1"]) - pad_x)
        y1 = max(0, int(detection["y1"]) - pad_y)
        x2 = min(image_width, int(detection["x2"]) + pad_x)
        y2 = min(image_height, int(detection["y2"]) + pad_y)
        return image[y1:y2, x1:x2]


if __name__ == "__main__":
    print("Plate detector ready. Optional model path:", MODEL_PATH)
