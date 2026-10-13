import unittest
from unittest.mock import patch

import cv2
import numpy as np

import app


class FakeVideoCapture:
    def __init__(self, frame_count, fps=30.0):
        self.frame_count = frame_count
        self.fps = fps
        self.frame_number = 0
        self.released = False
        self.grab_calls = 0

    def isOpened(self):
        return True

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        return 0

    def set(self, property_id, value):
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            self.frame_number = int(value)
            return True
        return False

    def read(self):
        frame = np.zeros((80, 160, 3), dtype=np.uint8)
        frame[0, 0] = (self.frame_number % 256, self.frame_number // 256, 0)
        return True, frame

    def grab(self):
        self.frame_number += 1
        self.grab_calls += 1
        return True

    def release(self):
        self.released = True


class FakeDetector:
    def __init__(self):
        self.image_sizes = []

    def detect(self, frame, imgsz=640):
        self.image_sizes.append(imgsz)
        return [{"box": True}]

    def crop_plate(self, frame, detection):
        return frame


class FakeOCR:
    _loaded = True
    load_error = ""

    def recognize(self, crop):
        number = int(crop[0, 0, 0]) + 256 * int(crop[0, 0, 1])
        return {
            "text": f"51A{number:05d}",
            "confidence": 0.8,
            "plate_like": True,
        }


class VideoScanTests(unittest.TestCase):
    def test_scan_caps_unique_results_and_stores_compressed_previews(self):
        capture = FakeVideoCapture(frame_count=420)
        detector = FakeDetector()
        progress = []
        with (
            patch.object(app.cv2, "VideoCapture", return_value=capture),
            patch.object(app, "get_plate_detector", return_value=detector),
            patch.object(app, "get_ocr_service", return_value=FakeOCR()),
        ):
            results, truncated = app.scan_video_for_plates(
                "video.mp4",
                lambda current, total: progress.append((current, total)),
            )

        self.assertEqual(len(results), app.MAX_VIDEO_SCAN_RESULTS)
        self.assertTrue(truncated)
        self.assertTrue(capture.released)
        self.assertEqual(capture.grab_calls, 392)
        self.assertEqual(detector.image_sizes, [416] * 28)
        self.assertEqual(progress[-1], (28, 28))
        self.assertTrue(all(isinstance(item["frame_jpeg"], bytes) for item in results))
        decoded = app.decode_video_scan_preview(results[0]["frame_jpeg"])
        self.assertEqual(decoded.shape, (80, 160, 3))

    def test_scan_preview_resizes_large_frames(self):
        frame = np.zeros((1920, 2560, 3), dtype=np.uint8)

        encoded = app.encode_video_scan_preview(frame)
        decoded = app.decode_video_scan_preview(encoded)

        self.assertLessEqual(max(decoded.shape[:2]), app.MAX_VIDEO_SCAN_PREVIEW_SIZE)


if __name__ == "__main__":
    unittest.main()
