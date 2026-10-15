import base64
import unittest
from pathlib import Path

import cv2
import numpy as np

import app


class BrowserCameraFrameTests(unittest.TestCase):
    def test_decodes_valid_jpeg_frame(self):
        expected = np.full((24, 48, 3), (20, 90, 180), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", expected)
        self.assertTrue(success)

        actual = app.decode_browser_camera_frame({
            "frame": base64.b64encode(encoded.tobytes()).decode("ascii"),
        })

        self.assertEqual(actual.shape, expected.shape)

    def test_rejects_invalid_base64_and_non_image_payload(self):
        with self.assertRaisesRegex(ValueError, "không hợp lệ"):
            app.decode_browser_camera_frame({"frame": "not base64!"})
        with self.assertRaisesRegex(ValueError, "Không giải mã"):
            app.decode_browser_camera_frame({
                "frame": base64.b64encode(b"not an image").decode("ascii"),
            })

    def test_rejects_oversized_frame_before_decoding(self):
        oversized = "A" * (app.MAX_BROWSER_FRAME_BYTES * 4 // 3 + 12)
        with self.assertRaisesRegex(ValueError, "vượt giới hạn"):
            app.decode_browser_camera_frame({"frame": oversized})

    def test_browser_component_uses_https_frame_transport_not_ice(self):
        component_html = (
            Path(app.BASE_DIR) / "browser_camera" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn("getUserMedia", component_html)
        self.assertIn("streamlit:setComponentValue", component_html)
        self.assertIn("isStreamlitMessage: true", component_html)
        self.assertIn("stopped: true", component_html)
        self.assertNotIn("RTCPeerConnection", component_html)
        self.assertNotIn("stun:", component_html.lower())
        self.assertNotIn("turn:", component_html.lower())


if __name__ == "__main__":
    unittest.main()
