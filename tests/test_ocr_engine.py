import unittest

import numpy as np

from ocr_engine import OCRService


class OCRPlateFormatTests(unittest.TestCase):
    def test_accepts_motorcycle_series_with_letter_and_digit(self):
        self.assertTrue(OCRService._looks_like_vietnamese_plate("12B116888"))

    def test_rejects_text_that_does_not_match_a_plate(self):
        self.assertFalse(OCRService._looks_like_vietnamese_plate("168881281"))

    def test_two_line_plate_text_is_read_top_to_bottom_when_boxes_are_tilted(self):
        service = OCRService.__new__(OCRService)
        results = [
            (
                np.array([[110, 5], [315, 57], [284, 166], [79, 113]], dtype=np.float32),
                "12B1",
                0.43,
            ),
            (
                np.array([[61, 85], [330, 143], [298, 261], [30, 203]], dtype=np.float32),
                "16888",
                0.96,
            ),
        ]

        text, confidence = service._combine_text_regions(results)

        self.assertEqual(text, "12B116888")
        self.assertAlmostEqual(confidence, (4 * 0.43 + 5 * 0.96) / 9)
        self.assertTrue(OCRService._looks_like_vietnamese_plate(text))


if __name__ == "__main__":
    unittest.main()
