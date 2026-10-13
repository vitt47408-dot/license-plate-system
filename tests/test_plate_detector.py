import unittest

from plate_detector import PlateDetector


class PlateDetectionFilteringTests(unittest.TestCase):
    def test_suppresses_duplicate_overlapping_plate_boxes(self):
        detections = [
            {"x1": 10, "y1": 10, "x2": 110, "y2": 70, "confidence": 0.95},
            {"x1": 0, "y1": 0, "x2": 120, "y2": 80, "confidence": 0.70},
            {"x1": 200, "y1": 10, "x2": 300, "y2": 70, "confidence": 0.60},
        ]

        filtered = PlateDetector._suppress_duplicate_detections(detections)

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["confidence"], 0.95)
        self.assertEqual(filtered[1]["x1"], 200)


if __name__ == "__main__":
    unittest.main()
