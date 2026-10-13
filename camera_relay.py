from __future__ import annotations

import asyncio
import secrets
import threading
import time
from dataclasses import dataclass
from fractions import Fraction

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame


@dataclass
class CameraPair:
    created_at: float
    frame: np.ndarray | None = None
    last_frame_at: float = 0.0


class CameraRelay:
    CODE_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    CODE_LENGTH = 8
    PAIR_LIFETIME_SECONDS = 30 * 60
    FRAME_STALE_SECONDS = 4.0
    MAX_ACTIVE_PAIRS = 10

    def __init__(self) -> None:
        self._pairs: dict[str, CameraPair] = {}
        self._lock = threading.Lock()

    def _purge_expired(self, now: float) -> None:
        expired = [
            code
            for code, pair in self._pairs.items()
            if now - pair.created_at > self.PAIR_LIFETIME_SECONDS
        ]
        for code in expired:
            del self._pairs[code]

    def create_pairing_code(self) -> str:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            if len(self._pairs) >= self.MAX_ACTIVE_PAIRS:
                raise RuntimeError("Đã đạt giới hạn số camera được ghép đang hoạt động.")
            for _ in range(20):
                code = "".join(
                    secrets.choice(self.CODE_ALPHABET)
                    for _ in range(self.CODE_LENGTH)
                )
                if code not in self._pairs:
                    self._pairs[code] = CameraPair(created_at=now)
                    return code
        raise RuntimeError("Không tạo được mã ghép camera duy nhất.")

    def revoke_pairing_code(self, code: str) -> None:
        with self._lock:
            self._pairs.pop(code, None)

    def has_pairing_code(self, code: str) -> bool:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            return code in self._pairs

    def publish(self, code: str, frame: np.ndarray) -> bool:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            pair = self._pairs.get(code)
            if pair is None:
                return False
            pair.frame = frame
            pair.last_frame_at = now
            return True

    def latest_frame(self, code: str) -> tuple[np.ndarray | None, bool]:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            pair = self._pairs.get(code)
            if pair is None:
                return None, False
            if pair.frame is None:
                return None, True
            is_live = now - pair.last_frame_at <= self.FRAME_STALE_SECONDS
            return pair.frame, is_live

    def create_track(self, code: str) -> "CameraRelayTrack":
        if not self.has_pairing_code(code):
            raise ValueError("Mã camera không tồn tại hoặc đã hết hạn.")
        return CameraRelayTrack(self, code)


class CameraRelayTrack(VideoStreamTrack):
    def __init__(self, relay: CameraRelay, code: str) -> None:
        super().__init__()
        self._relay = relay
        self._code = code
        self._timestamp = 0

    async def recv(self) -> VideoFrame:
        await asyncio.sleep(1 / 15)
        self._timestamp += 6000
        pts = self._timestamp
        time_base = Fraction(1, 90000)
        frame, is_live = self._relay.latest_frame(self._code)
        if frame is None:
            output = np.zeros((360, 640, 3), dtype=np.uint8)
        else:
            output = frame.copy()
            if not is_live:
                output = np.zeros_like(output)

        label = "CAMERA DANG TRUYEN" if is_live and frame is not None else "DANG CHO DIEN THOAI"
        cv2.putText(
            output,
            label,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 170) if is_live and frame is not None else (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        video_frame = VideoFrame.from_ndarray(output, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


camera_relay = CameraRelay()
