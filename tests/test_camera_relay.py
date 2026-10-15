import time
import unittest

import numpy as np

from camera_relay import CameraRelay


class CameraRelayTests(unittest.TestCase):
    def test_creates_valid_unique_pairing_codes(self):
        relay = CameraRelay()

        first = relay.create_pairing_code()
        second = relay.create_pairing_code()

        self.assertEqual(len(first), relay.CODE_LENGTH)
        self.assertTrue(set(first) <= set(relay.CODE_ALPHABET))
        self.assertNotEqual(first, second)
        self.assertTrue(relay.has_pairing_code(first))

    def test_rejects_missing_or_revoked_pairing_codes(self):
        relay = CameraRelay()

        code = relay.create_pairing_code()
        relay.revoke_pairing_code(code)

        self.assertFalse(relay.has_pairing_code(code))

    def test_returns_latest_frame_and_marks_stale_camera(self):
        relay = CameraRelay()
        code = relay.create_pairing_code()
        frame = np.full((40, 80, 3), 73, dtype=np.uint8)

        self.assertTrue(relay.publish(code, frame))
        received, is_live = relay.latest_frame(code)

        self.assertTrue(is_live)
        np.testing.assert_array_equal(received, frame)

        with relay._lock:
            relay._pairs[code].last_frame_at = (
                time.monotonic() - relay.FRAME_STALE_SECONDS - 1
            )
        _, is_live = relay.latest_frame(code)
        self.assertFalse(is_live)

if __name__ == "__main__":
    unittest.main()
