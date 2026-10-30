from __future__ import annotations

import asyncio
import base64
import io
import json
import mimetypes
import re
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import AsyncIterator

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from plate_detector import PlateDetector
from vehicle_detector import VehicleDetector

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads" / "realtime_video"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="License Plate Real-time API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_plate(value: str) -> str:
    value = re.sub(r"[^A-Z0-9]", "", value.upper())
    return value[:15]


class LicensePlateProcessor:
    def __init__(self) -> None:
        self.vehicle_detector = VehicleDetector()
        if getattr(self.vehicle_detector, "model", None) is None:
            raise RuntimeError(self.vehicle_detector.model_error or "Không thể tải YOLO vehicle")
        self.plate_detector = PlateDetector()
        if getattr(self.plate_detector, "model", None) is None:
            self.plate_detector.model_error = "Chưa có YOLO biển số; trạng thái cần kiểm tra lại"
        self._ocr = None
        self._ocr_error = ""
        self._inference_lock = threading.Lock()
        self._ocr_lock = threading.Lock()

    def _ocr_reader(self):
        with self._ocr_lock:
            if self._ocr is None:
                import easyocr
                self._ocr = easyocr.Reader(["en", "vi"], gpu=False, verbose=False)
            return self._ocr

    @staticmethod
    def _laplacian_sharpness(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if gray.size == 0:
            return 0.0
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _plate_detections(self, frame: np.ndarray) -> list[dict]:
        if getattr(self.plate_detector, "model", None) is None:
            return []
        try:
            return self.plate_detector.detect(frame, imgsz=640)
        except (OSError, RuntimeError, ValueError):
            return []

    def _read_plate(self, crop: np.ndarray) -> tuple[str, float]:
        reader = self._ocr_reader()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(
            gray,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            text_threshold=0.5,
            low_text=0.25,
        )
        candidates: list[tuple[str, float]] = []
        for _box, text, confidence in results:
            plate = normalize_plate(text)
            if 5 <= len(plate) <= 12 and confidence >= 0.25:
                candidates.append((plate, float(confidence)))
        if not candidates:
            return "", 0.0
        return max(candidates, key=lambda item: item[1])

    def process_frame(self, frame: np.ndarray, states: dict, frame_number: int) -> dict:
        with self._inference_lock:
            vehicles = self.vehicle_detector.model.track(
                frame,
                imgsz=640,
                conf=0.25,
                classes=list(self.vehicle_detector.VEHICLE_CLASSES),
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]
            plates = self._plate_detections(frame)

        detections: list[dict] = []
        plate_candidates: list[tuple[int, dict, np.ndarray, float]] = []
        boxes = getattr(vehicles, "boxes", None)
        vehicle_boxes: list[dict] = []
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            for track_id, coordinates, class_id, confidence in zip(ids, xyxy, cls, conf):
                x1, y1, x2, y2 = map(int, coordinates)
                label = self.vehicle_detector.VEHICLE_CLASSES.get(class_id, "VEHICLE")
                vehicle_boxes.append({"track_id": int(track_id), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label})
                detections.append({"kind": "vehicle", "track_id": int(track_id), "label": label, "confidence": round(float(confidence), 3), "box": [x1, y1, x2, y2]})

        for plate in plates:
            px1, py1, px2, py2 = (int(plate[k]) for k in ("x1", "y1", "x2", "y2"))
            best_vehicle = None
            plate_center_x, plate_center_y = (px1 + px2) / 2, (py1 + py2) / 2
            best_overlap = 0
            for vehicle in vehicle_boxes:
                if vehicle["x1"] <= plate_center_x <= vehicle["x2"] and vehicle["y1"] <= plate_center_y <= vehicle["y2"]:
                    overlap = max(0, min(px2, vehicle["x2"]) - max(px1, vehicle["x1"])) * max(0, min(py2, vehicle["y2"])-max(py1, vehicle["y1"]))
                    if overlap > best_overlap:
                        best_overlap, best_vehicle = overlap, vehicle
            if best_vehicle is None:
                continue
            crop = plate_detector_crop(frame, plate)
            if crop.size == 0:
                continue
            score = self._laplacian_sharpness(crop)
            track_id = best_vehicle["track_id"]
            state = states.setdefault(track_id, {"best_score": -1, "best_crop": None, "plate": "", "ocr_frame": -999})
            if not state["plate"] and score > float(state["best_score"]):
                state["best_score"] = score
                state["best_crop"] = crop.copy()
            if frame_number - int(state["ocr_frame"]) >= 8 and score >= 180 and not state["plate"]:
                plate_candidates.append((track_id, plate, crop, score))
                state["ocr_frame"] = frame_number
            detections.append({"kind": "plate", "track_id": track_id, "confidence": round(float(plate.get("confidence") or 0), 3), "box": [px1, py1, px2, py2]})

        results: list[dict] = []
        for track_id, plate, crop, score in plate_candidates:
            try:
                text, confidence = self._read_plate(crop)
            except Exception as exc:
                self._ocr_error = str(exc)
                continue
            if text:
                state = states[track_id]
                state["plate"] = text
                results.append({"type": "plate_result", "track_id": track_id, "plate": text, "ocr_confidence": round(confidence, 3), "sharpness": round(score, 1), "frame": frame_number})

        output = frame.copy()
        for vehicle in vehicle_boxes:
            cv2.rectangle(output, (vehicle["x1"], vehicle["y1"]), (vehicle["x2"], vehicle["y2"]), (0, 255, 0), 2)
            cv2.putText(output, f"XE #{vehicle['track_id']} - {vehicle['label']}", (vehicle["x1"], max(20, vehicle["y1"] - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for plate in plates:
            x1, y1, x2, y2 = (int(plate[k]) for k in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 190, 0), 2)
        return {"frame_image": output, "detections": detections, "results": results}


processor: LicensePlateProcessor | None = None
processor_lock = threading.Lock()


def get_processor() -> LicensePlateProcessor:
    global processor
    with processor_lock:
        if processor is None:
            processor = LicensePlateProcessor()
    return processor


def plate_detector_crop(frame: np.ndarray, plate: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(plate[k]) for k in ("x1", "y1", "x2", "y2"))
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - int(bw * 0.06))
    y1 = max(0, y1 - int(bh * 0.08))
    x2 = min(w, x2 + int(bw * 0.06))
    y2 = min(h, y2 + int(bh * 0.08))
    return frame[y1:y2, x1:x2]


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Không thể mã hóa khung hình")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/videos")
async def upload_video(video: UploadFile = File(...)) -> JSONResponse:
    if not video.filename or Path(video.filename).suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ MP4/AVI/MOV/MKV")
    job_id = uuid.uuid4().hex
    destination = UPLOAD_DIR / f"{job_id}{Path(video.filename).suffix.lower()}"
    with destination.open("wb") as output:
        while chunk := await video.read(1024 * 1024):
            output.write(chunk)
    return JSONResponse({"job_id": job_id, "video_path": str(destination), "websocket": f"/ws/videos/{job_id}"})


@app.websocket("/ws/frames")
async def websocket_frames(websocket: WebSocket) -> None:
    await websocket.accept()
    states: dict[int, dict] = {}
    ai = get_processor()
    frame_number = 0
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if "text" in message:
                command = json.loads(message["text"])
                if command.get("type") == "reset":
                    states.clear()
                    frame_number = 0
                elif command.get("type") == "stop":
                    await websocket.send_json({"type": "done"})
                    break
                continue
            data = message.get("bytes")
            if not data:
                continue
            encoded = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"type": "error", "message": "Không giải mã được frame"})
                continue
            frame_number += 1
            processed = await asyncio.to_thread(ai.process_frame, frame, states, frame_number)
            await websocket.send_json({
                "type": "frame",
                "frame": encode_jpeg(processed["frame_image"]),
                "frame_number": frame_number,
                "detections": processed["detections"],
            })
            for result in processed["results"]:
                await websocket.send_json(result)
    except WebSocketDisconnect:
        return
    finally:
        try:
            await websocket.send_json({"type": "done"})
        except RuntimeError:
            pass


@app.websocket("/ws/videos/{job_id}")
async def websocket_video(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    files = list(UPLOAD_DIR.glob(f"{job_id}.*"))
    if not files:
        await websocket.send_json({"type": "error", "message": "Không tìm thấy job video"})
        await websocket.close()
        return

    ai = get_processor()
    states: dict[int, dict] = {}
    capture = cv2.VideoCapture(str(files[0]))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = 1.0 / max(fps, 1.0)
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            # Skip frames only for expensive OCR, while retaining smooth preview frames.
            if frame_number % 2 != 0:
                continue
            processed = await asyncio.to_thread(ai.process_frame, frame, states, frame_number)
            await websocket.send_json({"type": "frame", "frame": encode_jpeg(processed["frame_image"]), "frame_number": frame_number, "detections": processed["detections"]})
            for result in processed["results"]:
                await websocket.send_json(result)
            await asyncio.sleep(frame_interval)
        await websocket.send_json({"type": "done", "frame_count": frame_number})
    except WebSocketDisconnect:
        pass
    finally:
        capture.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("realtime_api:app", host="0.0.0.0", port=8000, reload=False)
