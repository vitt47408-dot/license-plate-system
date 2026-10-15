import unittest
import threading

import numpy as np

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


class BlockingDetector(FakeDetector):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def detect(self, frame):
        self.started.set()
        self.release.wait(timeout=2)
        return super().detect(frame)


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
        self.processor.ocr_service = FakeOCR(confidence=0.39)
        self.processor.process(self.frame, now=1.0)
        self.processor.process(self.frame, now=2.0)
        self.assertEqual(self.recorded, [])

        self.processor.ocr_service = FakeOCR(plate_like=False)
        self.processor.process(self.frame, now=3.0)
        self.processor.process(self.frame, now=4.0)
        self.assertEqual(self.recorded, [])

    def test_records_repeated_plate_like_ocr_below_high_confidence_threshold(self):
        self.processor.ocr_service = FakeOCR(confidence=0.458)

        self.processor.process(self.frame, now=1.0)
        self.processor.process(self.frame, now=2.0)
        self.assertEqual(self.recorded, [])

        self.processor.process(self.frame, now=3.0)
        self.assertEqual(len(self.recorded), 1)
        self.assertEqual(self.recorded[0][0:2], ("51A12345", "IN"))
        self.assertAlmostEqual(self.recorded[0][3], 0.458)

    def test_restarts_consensus_after_sighting_gap(self):
        self.processor.process(self.frame, now=1.0)
        self.processor.process(self.frame, now=6.0)
        self.assertEqual(self.recorded, [])
        self.processor.process(self.frame, now=7.0)
        self.assertEqual(len(self.recorded), 1)

    def test_records_reliably_read_plate_when_vehicle_roi_is_not_detected(self):
        self.processor.vehicle_detector = NoVehicleDetector()

        self.processor.process(self.frame, now=1.0)
        self.processor.process(self.frame, now=2.0)

        self.assertEqual(len(self.recorded), 1)
        self.assertEqual(self.recorded[0][0:2], ("51A12345", "IN"))
        self.assertEqual(
            self.processor._status,
            "PLATE ROI · AUTO-SAVED",
        )

    def test_records_reliably_read_plate_when_vehicle_detector_is_unavailable(self):
        self.processor.vehicle_detector = None

        self.processor.process(self.frame, now=1.0)
        self.processor.process(self.frame, now=2.0)

        self.assertEqual(len(self.recorded), 1)

    def test_explains_when_detected_plate_is_not_confirmed_by_ocr(self):
        self.processor.ocr_service = FakeOCR(confidence=0.5)
        self.processor.process(self.frame, now=1.0)

        self.assertEqual(self.recorded, [])
        self.assertEqual(
            self.processor.status,
            "PLATE ROI · CHECKING OCR 1/3",
        )
        self.assertTrue(
            any(
                "51A12345: đang xác minh OCR (1/3, tin cậy 50%)" in message
                for message in self.processor.messages
            )
        )

    def test_vehicle_model_failure_does_not_block_confirmed_plate(self):
        class FailedVehicleDetector(NoVehicleDetector):
            model_error = "weight download failed"

        recorded = []
        processor = AutomaticPlateProcessor(
            detector=FakeDetector(),
            ocr_service=FakeOCR(),
            direction="IN",
            record_event=lambda *args: recorded.append(args) or "saved",
            vehicle_detector=FailedVehicleDetector(),
        )

        processor.process(self.frame, now=1.0)
        processor.process(self.frame, now=2.0)

        self.assertEqual(len(recorded), 1)
        self.assertTrue(any("weight download failed" in message for message in processor.messages))

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

    def test_background_scan_does_not_block_camera_preview(self):
        detector = BlockingDetector()
        processor = AutomaticPlateProcessor(
            detector=detector,
            ocr_service=FakeOCR(),
            direction="IN",
            record_event=lambda *args: "saved",
            background_processing=True,
        )

        result = processor.process(self.frame, now=1.0)

        self.assertEqual(result.shape, self.frame.shape)
        self.assertTrue(detector.started.wait(timeout=1))
        self.assertTrue(processor._scan_worker_running)
        detector.release.set()
        processor._scan_thread.join(timeout=2)
        self.assertFalse(processor._scan_worker_running)

    def test_processes_camera_frames_without_webrtc_dependency(self):
        self.assertEqual(self.processor.scan_interval, 1.0)
        outgoing = self.processor.process(self.frame, now=1.0)
        self.assertEqual(outgoing.shape, self.frame.shape)

    def test_publishes_processed_frames_to_remote_camera_relay(self):
        published = []
        processor = AutomaticPlateProcessor(
            detector=FakeDetector(),
            ocr_service=FakeOCR(),
            direction="IN",
            record_event=lambda *args: "saved",
            frame_publisher=lambda frame: published.append(frame.copy()),
        )

        output = processor.process(self.frame, now=1.0)

        self.assertEqual(len(published), 1)
        np.testing.assert_array_equal(published[0], output)


if __name__ == "__main__":
    unittest.main()
