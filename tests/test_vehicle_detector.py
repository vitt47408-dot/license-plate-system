import unittest

import numpy as np

from vehicle_detector import VehicleDetector


class FakeBox:
    def __init__(self, class_id, coordinates, confidence):
        self.cls = np.array([class_id])
        self.xyxy = np.array([coordinates])
        self.conf = np.array([confidence])


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self):
        self.arguments = None

    def __call__(self, image, **kwargs):
        self.arguments = kwargs
        return [
            FakeResult([
                FakeBox(2, [10, 15, 100, 90], 0.9),
                FakeBox(3, [120, 15, 200, 90], 0.8),
            ])
        ]


class VehicleDetectorTests(unittest.TestCase):
    def test_only_detects_common_vehicle_classes(self):
        detector = VehicleDetector.__new__(VehicleDetector)
        detector.model = FakeModel()
        detector.model_error = ""

        detections = detector.detect(np.zeros((120, 220, 3), dtype=np.uint8))

        self.assertEqual([item["label"] for item in detections], ["CAR", "MOTORCYCLE"])
        self.assertEqual(detector.model.arguments["classes"], [2, 3, 5, 7])
        self.assertEqual(detector.model.arguments["imgsz"], 480)
        self.assertEqual(detector.model.arguments["max_det"], 5)

    def test_draws_green_boxes_for_detected_vehicle(self):
        detector = VehicleDetector.__new__(VehicleDetector)
        image = np.zeros((50, 80, 3), dtype=np.uint8)

        result = detector.draw_boxes(image, [{
            "x1": 10,
            "y1": 10,
            "x2": 60,
            "y2": 40,
            "label": "CAR",
            "confidence": 0.9,
        }])

        self.assertTrue(np.any(np.all(result == (0, 255, 0), axis=2)))


if __name__ == "__main__":
    unittest.main()
