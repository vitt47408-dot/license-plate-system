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


class NoVehicleDetector:
    model_error = ""

    def detect(self, frame):
        return []

    def draw_boxes(self, frame, detections):
        return frame.copy()


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

        self.processor.process(self.frame, now=2.0)
        self.assertEqual(len(self.recorded), 1)
        self.assertEqual(self.recorded[0][0:2], ("51A12345", "IN"))

        self.processor.process(self.frame, now=3.0)
        self.assertEqual(len(self.recorded), 1)

        for step in range(1, 40):
            self.processor.process(self.frame, now=3.0 + step)
        self.assertEqual(len(self.recorded), 1)

    def test_ignores_low_confidence_and_non_plate_ocr(self):
        self.processor.ocr_service = FakeOCR(confidence=0.5)
        self.processor.process(self.frame, now=1.0)
        self.processor.process(self.frame, now=2.0)
        self.assertEqual(self.recorded, [])

        self.processor.ocr_service = FakeOCR(plate_like=False)
        self.processor.process(self.frame, now=3.0)
        self.processor.process(self.frame, now=4.0)
        self.assertEqual(self.recorded, [])

    def test_restarts_consensus_after_sighting_gap(self):
        self.processor.process(self.frame, now=1.0)
        self.processor.process(self.frame, now=6.0)
        self.assertEqual(self.recorded, [])
        self.processor.process(self.frame, now=7.0)
        self.assertEqual(len(self.recorded), 1)

    def test_does_not_auto_save_a_plate_without_a_detected_vehicle(self):
        self.processor.vehicle_detector = NoVehicleDetector()

        self.processor.process(self.frame, now=1.0)
        self.processor.process(self.frame, now=2.0)

        self.assertEqual(self.recorded, [])
        self.assertEqual(
            self.processor._status,
            "PLATE ROI · WAITING FOR VEHICLE ROI",
        )

    def test_requires_vehicle_detector_when_configured_for_live_app(self):
        recorded = []
        processor = AutomaticPlateProcessor(
            detector=FakeDetector(),
            ocr_service=FakeOCR(),
            direction="IN",
            record_event=lambda *args: recorded.append(args) or "saved",
            require_vehicle_detection=True,
        )

        processor.process(self.frame, now=1.0)

        self.assertEqual(recorded, [])
        self.assertIn("AUTO-SAVE DISABLED", processor._status)

    def test_defers_model_loading_until_first_camera_frame(self):
        loaded = []
        processor = AutomaticPlateProcessor(
            detector=None,
            ocr_service=None,
            direction="IN",
            record_event=lambda *args: "saved",
            model_loader=lambda: (
                loaded.append(True) or FakeDetector(),
                FakeOCR(),
                None,
            ),
        )

        self.assertEqual(loaded, [])
        processor.process(self.frame, now=1.0)
        self.assertEqual(loaded, [True])

    def test_reports_model_load_failure_without_crashing_camera_frame(self):
        processor = AutomaticPlateProcessor(
            detector=None,
            ocr_service=None,
            direction="IN",
            record_event=lambda *args: "saved",
            model_loader=lambda: (_ for _ in ()).throw(RuntimeError("weights unavailable")),
        )

        result = processor.process(self.frame, now=1.0)

        self.assertEqual(result.shape, self.frame.shape)
        self.assertIn("MODEL LOAD FAILED", processor._status)
        self.assertIn("weights unavailable", processor.messages[-1])

    def test_implements_webrtc_video_processor_contract(self):
        self.assertIsInstance(self.processor, VideoProcessorBase)
        self.assertEqual(self.processor.scan_interval, 1.0)
        incoming = VideoFrame.from_ndarray(self.frame, format="bgr24")
        outgoing = self.processor.recv(incoming)
        self.assertIsInstance(outgoing, VideoFrame)
        self.assertEqual(outgoing.to_ndarray(format="bgr24").shape, self.frame.shape)


if __name__ == "__main__":
    unittest.main()
