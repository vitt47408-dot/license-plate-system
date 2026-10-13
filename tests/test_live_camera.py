import unittest

from av import VideoFrame
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from live_camera import AutomaticPlateProcessor


class FakeDetector:
    def detect(self, frame):
        return [{"x1": 1, "y1": 1, "x2": 20, "y2": 10}]

    def crop_plate(self, frame, detection):
        return frame

    def draw_boxes(self, frame, detections):
        return frame.copy()


class FakeOCR:
    def __init__(self, confidence=0.9, plate_like=True):
        self.confidence = confidence
        self.plate_like = plate_like

    def recognize(self, crop):
        return {
            "text": "51A12345",
            "confidence": self.confidence,
            "plate_like": self.plate_like,
        }


class AutomaticPlateProcessorTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((40, 80, 3), dtype=np.uint8)
        self.recorded = []
        self.processor = AutomaticPlateProcessor(
            detector=FakeDetector(),
            ocr_service=FakeOCR(),
            direction="IN",
            record_event=lambda *args: self.recorded.append(args) or "ĐÃ TỰ GHI NHẬN",
        )

    def test_records_after_two_reliable_reads_and_filters_repeats(self):
        self.processor.process(self.frame, now=1.0)
        self.assertEqual(self.recorded, [])

        self.processor.process(self.frame, now=1.75)
        self.assertEqual(len(self.recorded), 1)
        self.assertEqual(self.recorded[0][0:2], ("51A12345", "IN"))

        self.processor.process(self.frame, now=2.5)
        self.assertEqual(len(self.recorded), 1)

        for step in range(1, 40):
            self.processor.process(self.frame, now=2.5 + step * 0.75)
        self.assertEqual(len(self.recorded), 1)

    def test_ignores_low_confidence_and_non_plate_ocr(self):
        self.processor.ocr_service = FakeOCR(confidence=0.5)
        self.processor.process(self.frame, now=1.0)
        self.processor.process(self.frame, now=1.75)
        self.assertEqual(self.recorded, [])

        self.processor.ocr_service = FakeOCR(plate_like=False)
        self.processor.process(self.frame, now=2.5)
        self.processor.process(self.frame, now=3.25)
        self.assertEqual(self.recorded, [])

    def test_restarts_consensus_after_sighting_gap(self):
        self.processor.process(self.frame, now=1.0)
        self.processor.process(self.frame, now=6.0)
        self.assertEqual(self.recorded, [])
        self.processor.process(self.frame, now=6.75)
        self.assertEqual(len(self.recorded), 1)

    def test_implements_webrtc_video_processor_contract(self):
        self.assertIsInstance(self.processor, VideoProcessorBase)
        incoming = VideoFrame.from_ndarray(self.frame, format="bgr24")
        outgoing = self.processor.recv(incoming)
        self.assertIsInstance(outgoing, VideoFrame)
        self.assertEqual(outgoing.to_ndarray(format="bgr24").shape, self.frame.shape)


if __name__ == "__main__":
    unittest.main()
