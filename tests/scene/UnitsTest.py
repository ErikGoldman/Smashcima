import unittest

from smashcima.geometry.units import mm_to_px, px_to_mm


class UnitsTest(unittest.TestCase):
    def test_an_inch_is_25_4_millimetres(self):
        self.assertAlmostEqual(mm_to_px(25.4, dpi=300), 300)
        self.assertAlmostEqual(px_to_mm(300, dpi=300), 25.4)
