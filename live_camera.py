from __future__ import annotations

import sqlite3
import time
from collections import OrderedDict
from typing import Callable

import cv2
import numpy as np
from av import VideoFrame
from streamlit_webrtc import VideoProcessorBase

from database import normalize_plate


class AutomaticPlateProcessor(VideoProcessorBase):
    def __init__(
        self,
        detector,
        ocr_service,
        direction: str,
        record_event: Callable[[str, str, np.ndarray, float], str],
        minimum_confidence: float = 0.65,
        required_sightings: int = 2,
        scan_interval: float = 0.75,
        consensus_window: float = 4.0,
    ) -> None:
        self.detector = detector
        self.ocr_service = ocr_service
        self.direction = direction
        self.record_event = record_event
        self.minimum_confidence = minimum_confidence
        self.required_sightings = required_sightings
        self.scan_interval = scan_interval
        self.consensus_window = consensus_window
        self._last_scan_at = float("-inf")
        self._last_detections: list[dict] = []
        self._sightings: dict[str, tuple[int, float]] = {}
        self._saved_in_visit: set[str] = set()
        self._retry_after: dict[str, float] = {}
        self._messages: OrderedDict[str, str] = OrderedDict()

    @property
    def messages(self) -> list[str]:
        return list(self._messages.values())[-3:]

    def _add_message(self, plate: str, message: str) -> None:
        self._messages[plate] = message
        self._messages.move_to_end(plate)
        while len(self._messages) > 10:
            self._messages.popitem(last=False)

    def _annotate(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        output = self.detector.draw_boxes(frame, detections) if detections else frame.copy()
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

    def process(self, frame: np.ndarray, now: float | None = None) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Camera trả về khung hình rỗng.")

        current_time = time.monotonic() if now is None else now
        if current_time - self._last_scan_at < self.scan_interval:
            return self._annotate(frame, self._last_detections)
        self._last_scan_at = current_time

        detections = self.detector.detect(frame)
        self._last_detections = detections
        active_plates: set[str] = set()

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
            else:
                self._saved_in_visit.add(plate)
                self._retry_after.pop(plate, None)

            self._add_message(plate, message)

        for plate, (_, last_seen) in list(self._sightings.items()):
            if plate not in active_plates and current_time - last_seen > self.consensus_window:
                del self._sightings[plate]
                self._saved_in_visit.discard(plate)
                self._retry_after.pop(plate, None)

        return self._annotate(frame, detections)

    def recv(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        processed = self.process(image)
        return VideoFrame.from_ndarray(processed, format="bgr24")
