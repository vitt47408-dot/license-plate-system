from __future__ import annotations

import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Callable

import cv2
import numpy as np

from database import normalize_plate


class AutomaticPlateProcessor:
    def __init__(
        self,
        detector,
        ocr_service,
        direction: str,
        record_event: Callable[[str, str, np.ndarray, float], str],
        vehicle_detector=None,
        minimum_confidence: float = 0.65,
        required_sightings: int = 2,
        scan_interval: float = 1.0,
        consensus_window: float = 4.0,
        model_loader: Callable[[], tuple[object, object, object]] | None = None,
        require_vehicle_detection: bool = False,
        background_processing: bool = False,
        frame_publisher: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.detector = detector
        self.ocr_service = ocr_service
        self.direction = direction
        self.record_event = record_event
        self.vehicle_detector = vehicle_detector
        self.model_loader = model_loader
        self.require_vehicle_detection = require_vehicle_detection
        self.background_processing = background_processing
        self.frame_publisher = frame_publisher
        self._models_loaded = model_loader is None
        self._model_load_error = ""
        self._model_load_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.minimum_confidence = minimum_confidence
        self.required_sightings = required_sightings
        self.scan_interval = scan_interval
        self.consensus_window = consensus_window
        self._last_scan_at = float("-inf")
        self._pending_scan: tuple[np.ndarray, float] | None = None
        self._scan_worker_running = False
        self._scan_thread: threading.Thread | None = None
        self._last_detections: list[dict] = []
        self._last_vehicle_detections: list[dict] = []
        self._sightings: dict[str, tuple[int, float]] = {}
        self._saved_in_visit: set[str] = set()
        self._retry_after: dict[str, float] = {}
        self._messages: OrderedDict[str, str] = OrderedDict()
        self._status = "WAITING FOR CAMERA FRAME"

    @property
    def messages(self) -> list[str]:
        with self._state_lock:
            return list(self._messages.values())[-3:]

    def _add_message(self, plate: str, message: str) -> None:
        with self._state_lock:
            self._messages[plate] = message
            self._messages.move_to_end(plate)
            while len(self._messages) > 10:
                self._messages.popitem(last=False)

    def _ensure_models_loaded(self) -> bool:
        if self._models_loaded:
            return not self._model_load_error

        with self._model_load_lock:
            if self._models_loaded:
                return not self._model_load_error
            try:
                (
                    self.detector,
                    self.ocr_service,
                    self.vehicle_detector,
                ) = self.model_loader()
            except (ImportError, OSError, RuntimeError, ValueError, AssertionError, EOFError) as exc:
                self._model_load_error = str(exc)

            if not self._model_load_error and not getattr(self.ocr_service, "_loaded", True):
                self._model_load_error = (
                    getattr(self.ocr_service, "load_error", "")
                    or "OCR chưa sẵn sàng; kiểm tra gói EasyOCR và quyền tải mô hình."
                )

            self._models_loaded = True
            if self._model_load_error:
                self._status = f"MODEL LOAD FAILED · {self._model_load_error[:60]}"
                self._add_message("SYSTEM", f"Không tải được mô hình AI: {self._model_load_error}")
                return False
        return True

    def _scan_frame(self, frame: np.ndarray, current_time: float) -> None:
        if not self._ensure_models_loaded():
            self._last_detections = []
            self._last_vehicle_detections = []
            return

        if self.require_vehicle_detection and self.vehicle_detector is None:
            self._status = "VEHICLE MODEL UNAVAILABLE · AUTO-SAVE DISABLED"
            self._add_message(
                "SYSTEM",
                "Không có mô hình xe; khung thân xe và tự ghi nhận đang bị khóa.",
            )
            self._last_detections = []
            self._last_vehicle_detections = []
            return

        detections = self.detector.detect(frame)
        self._last_detections = detections
        vehicle_detections = (
            self.vehicle_detector.detect(frame)
            if self.vehicle_detector
            else []
        )
        self._last_vehicle_detections = vehicle_detections
        for key, label, model in (
            ("PLATE MODEL", "biển số", self.detector),
            ("VEHICLE MODEL", "xe", self.vehicle_detector),
        ):
            if model is None:
                continue
            model_error = str(getattr(model, "model_error", "") or "")
            if model_error:
                self._add_message(
                    key,
                    f"Mô hình {label} báo lỗi: {model_error}",
                )
        self._status = (
            f"PLATE ROI: {len(detections)} · VEHICLE ROI: {len(vehicle_detections)} · 1s"
            if detections
            else (
                "VEHICLE ROI · NO PLATE · CENTER / BRIGHTEN PLATE"
                if vehicle_detections
                else "NO VEHICLE OR PLATE · CENTER / BRIGHTEN"
            )
        )
        active_plates: set[str] = set()

        if self.vehicle_detector and not vehicle_detections:
            if detections:
                self._status = "PLATE ROI · WAITING FOR VEHICLE ROI"
            else:
                self._status = "NO VEHICLE ROI · AUTO-SAVE DISABLED"
            return

        for detection in detections:
            crop = self.detector.crop_plate(frame, detection)
            result = self.ocr_service.recognize(crop)
            plate = normalize_plate(str(result.get("text") or ""))
            confidence = float(result.get("confidence") or 0)
            if (
                not result.get("plate_like")
                or confidence < self.minimum_confidence
                or len(plate) < 5
            ):
                continue

            if plate in active_plates:
                continue
            active_plates.add(plate)
            previous_count, previous_time = self._sightings.get(plate, (0, 0.0))
            count = previous_count + 1 if current_time - previous_time <= self.consensus_window else 1
            self._sightings[plate] = (count, current_time)

            if count < self.required_sightings:
                self._status = f"PLATE ROI · CHECKING OCR {count}/{self.required_sightings}"
                self._add_message(plate, f"{plate}: đang xác minh OCR ({count}/{self.required_sightings})")
                continue

            if plate in self._saved_in_visit:
                continue
            if current_time < self._retry_after.get(plate, float("-inf")):
                continue

            try:
                message = self.record_event(
                    plate,
                    self.direction,
                    frame.copy(),
                    confidence,
                )
            except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
                message = f"{plate}: không lưu được sự kiện ({exc})"
                self._retry_after[plate] = current_time + 5.0
                self._status = "PLATE ROI · SAVE FAILED"
            else:
                self._saved_in_visit.add(plate)
                self._retry_after.pop(plate, None)
                self._status = "PLATE ROI · AUTO-SAVED"

            self._add_message(plate, message)

        if detections and not active_plates:
            self._status = "GREEN ROI · OCR LOW / PLATE NOT CONFIRMED"

        for plate, (_, last_seen) in list(self._sightings.items()):
            if plate not in active_plates and current_time - last_seen > self.consensus_window:
                del self._sightings[plate]
                self._saved_in_visit.discard(plate)
                self._retry_after.pop(plate, None)

    def _run_scan_worker(self) -> None:
        while True:
            with self._state_lock:
                task = self._pending_scan
                self._pending_scan = None
                if task is None:
                    self._scan_worker_running = False
                    return

            frame, scan_time = task
            try:
                self._scan_frame(frame, scan_time)
            except (OSError, RuntimeError, ValueError, TypeError, IndexError, sqlite3.Error) as exc:
                self._last_detections = []
                self._last_vehicle_detections = []
                self._status = f"SCAN FAILED · {str(exc)[:60]}"
                self._add_message("SYSTEM", f"Lỗi xử lý khung hình: {exc}")

    def _schedule_scan(self, frame: np.ndarray, current_time: float) -> None:
        start_worker = False
        with self._state_lock:
            if current_time - self._last_scan_at < self.scan_interval:
                return
            self._last_scan_at = current_time
            self._pending_scan = (frame.copy(), current_time)
            self._status = "ĐANG PHÂN TÍCH · QUÉT MỖI 1 GIÂY"
            if not self._scan_worker_running:
                self._scan_worker_running = True
                start_worker = True

        if start_worker:
            worker = threading.Thread(
                target=self._run_scan_worker,
                name="plate-camera-scan",
                daemon=True,
            )
            self._scan_thread = worker
            worker.start()

    def process(self, frame: np.ndarray, now: float | None = None) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Camera trả về khung hình rỗng.")

        current_time = time.monotonic() if now is None else now
        if self.background_processing:
            self._schedule_scan(frame, current_time)
            with self._state_lock:
                detections = list(self._last_detections)
                vehicle_detections = list(self._last_vehicle_detections)
                status = self._status
            output = self._annotate(frame, detections, vehicle_detections, status)
        else:
            if current_time - self._last_scan_at >= self.scan_interval:
                self._last_scan_at = current_time
                self._scan_frame(frame, current_time)
            with self._state_lock:
                detections = list(self._last_detections)
                vehicle_detections = list(self._last_vehicle_detections)
                status = self._status
            output = self._annotate(frame, detections, vehicle_detections, status)
        if self.frame_publisher:
            self.frame_publisher(output)
        return output

    def _annotate(
        self,
        frame: np.ndarray,
        detections: list[dict],
        vehicle_detections: list[dict],
        status: str | None = None,
    ) -> np.ndarray:
        output = self.detector.draw_boxes(frame, detections) if detections else frame.copy()
        if self.vehicle_detector and vehicle_detections:
            output = self.vehicle_detector.draw_boxes(output, vehicle_detections)
        status = self._status if status is None else status
        if status:
            cv2.putText(
                output,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if detections else (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
        messages = self.messages
        for index, message in enumerate(messages):
            cv2.putText(
                output,
                message[:90],
                (12, max(28, output.shape[0] - 16 - (len(messages) - index - 1) * 26)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 170),
                2,
                cv2.LINE_AA,
            )
        return output
