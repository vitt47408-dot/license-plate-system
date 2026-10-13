from __future__ import annotations

import re
import zipfile
from statistics import median
from typing import Dict, List, Optional

import cv2
import numpy as np


class OCRService:
    def __init__(self) -> None:
        self.reader = None
        self._loaded = False
        self.load_error = ""
        self._load_reader()

    def _load_reader(self) -> None:
        try:
            import easyocr
            self.reader = easyocr.Reader(["en", "vi"], gpu=False)
            self._loaded = True
        except (ImportError, OSError, RuntimeError, ValueError, AssertionError, EOFError, zipfile.BadZipFile) as exc:
            self.reader = None
            self._loaded = False
            self.load_error = str(exc)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Ảnh để OCR rỗng")
        image = np.asarray(image)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        scale = min(6.0, max(2.0, 96.0 / max(1, gray.shape[0])))
        return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def _prepare_variants(self, image: np.ndarray) -> List[np.ndarray]:
        if image is None or image.size == 0:
            raise ValueError("Ảnh để OCR rỗng")
        image = np.asarray(image)
        height, width = image.shape[:2]
        scale = min(6.0, max(2.0, 96.0 / max(1, height)))
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if resized.shape[1] > 1200:
            ratio = 1200 / resized.shape[1]
            resized = cv2.resize(
                resized,
                None,
                fx=ratio,
                fy=ratio,
                interpolation=cv2.INTER_AREA,
            )
        if len(resized.shape) == 2:
            gray = resized
            color = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            color = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        return [color, gray, enhanced]

    def clean_text(self, text: str) -> str:
        cleaned = text.strip().upper()
        cleaned = cleaned.replace(" ", "")
        cleaned = re.sub(r"[^A-Z0-9-]", "", cleaned)
        return cleaned[:15]

    @staticmethod
    def _looks_like_vietnamese_plate(text: str) -> bool:
        normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
        return bool(
            re.fullmatch(r"\d{2}[A-Z]{1,3}\d{4,6}", normalized)
            or re.fullmatch(r"\d{2}[A-Z]\d{1,2}\d{4,6}", normalized)
            or (
                7 <= len(normalized) <= 10
                and re.fullmatch(r"\d{1,2}[A-Z]{1,3}\d{3,6}", normalized)
            )
        )

    def _combine_text_regions(self, results: List[tuple]) -> tuple[str, float]:
        entries = []
        for item in results:
            if len(item) < 3:
                continue
            text = self.clean_text(str(item[1]))
            if not text:
                continue
            points = np.asarray(item[0], dtype=np.float32)
            center_x = float(points[:, 0].mean())
            center_y = float(points[:, 1].mean())
            text_height = float(points[:, 1].max() - points[:, 1].min())
            entries.append((center_x, center_y, max(text_height, 1.0), text, float(item[2])))
        if not entries:
            return "", 0.0

        line_tolerance = max(8.0, median(entry[2] for entry in entries) * 0.35)
        lines: List[List[tuple]] = []
        for entry in sorted(entries, key=lambda value: (value[1], value[0])):
            matching_line = next(
                (
                    line for line in lines
                    if abs(entry[1] - median(item[1] for item in line)) <= line_tolerance
                ),
                None,
            )
            if matching_line is None:
                lines.append([entry])
            else:
                matching_line.append(entry)

        ordered = [
            entry
            for line in sorted(lines, key=lambda row: median(item[1] for item in row))
            for entry in sorted(line, key=lambda value: value[0])
        ]
        combined = "".join(entry[3] for entry in ordered)
        total_chars = sum(len(entry[3]) for entry in ordered)
        confidence = (
            sum(entry[4] * len(entry[3]) for entry in ordered) / total_chars
            if total_chars
            else 0.0
        )
        return combined, confidence

    def recognize(self, image: np.ndarray) -> Dict[str, object]:
        if self.reader is None:
            return {
                "text": "",
                "confidence": 0.0,
                "error": (
                    "EasyOCR chưa sẵn sàng. Kiểm tra gói easyocr và kết nối mạng để tải trọng số."
                    + (f" Chi tiết: {self.load_error}" if self.load_error else "")
                ),
            }

        try:
            variants = self._prepare_variants(image)
            candidates = []
            for variant in variants:
                results = self.reader.readtext(
                    variant,
                    detail=1,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    mag_ratio=1.5,
                    text_threshold=0.5,
                    low_text=0.25,
                )
                text, confidence = self._combine_text_regions(results)
                if text:
                    plate_like = self._looks_like_vietnamese_plate(text)
                    score = confidence + (0.35 if plate_like else 0.0)
                    candidates.append((score, text, confidence, plate_like))
        except (OSError, RuntimeError, ValueError, TypeError, IndexError) as exc:  # pragma: no cover
            return {"text": "", "confidence": 0.0, "error": f"OCR thất bại: {exc}"}

        if not candidates:
            return {"text": "", "confidence": 0.0, "error": "Không nhận dạng được ký tự trên vùng biển số."}

        _, best_text, best_conf, plate_like = max(candidates, key=lambda item: item[0])
        error = "" if plate_like else "Kết quả OCR chưa giống định dạng biển số phổ biến; cần người vận hành kiểm tra."
        return {
            "text": best_text,
            "confidence": round(best_conf, 3),
            "plate_like": plate_like,
            "error": error,
        }


if __name__ == "__main__":
    print("OCR engine ready.")
