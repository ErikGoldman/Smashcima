import unittest

import numpy as np

from smashcima.scene.SmuflLabels import SmuflLabels
from smashcima.scene.semantic.Clef import Clef
from smashcima.scene.semantic.ClefSign import ClefSign
from smashcima.scene.semantic.Barline import BarStyle
from smashcima.synthesis.notation.column.SignaturesColumn import \
    key_signature_accidentals, time_signature_rows
from smashcima.scene.semantic.TimeSignature import TimeSignature, TimeSymbol
from smashcima.synthesis.notation.column.BarlinesColumn import \
    DOTS, HEAVY, THIN, barline_parts
from smashcima.synthesis.page.Quilter import Quilter

TREBLE = Clef(ClefSign.G, 2)
BASS = Clef(ClefSign.F, 4)
SHARP = SmuflLabels.accidentalSharp.value
FLAT = SmuflLabels.accidentalFlat.value
NATURAL = SmuflLabels.accidentalNatural.value


class KeySignatureTest(unittest.TestCase):
    def test_two_sharps_on_treble_sit_on_f5_and_c5(self):
        self.assertEqual(key_signature_accidentals(2, 2, TREBLE), [(SHARP, 4), (SHARP, 1)])

    def test_bass_clef_moves_the_pattern_two_steps_down(self):
        self.assertEqual(key_signature_accidentals(3, 3, BASS), [(SHARP, 2), (SHARP, -1), (SHARP, 3)])

    def test_three_flats_on_bass_sit_on_b2_e3_a2(self):
        self.assertEqual(key_signature_accidentals(-3, -3, BASS), [(FLAT, -2), (FLAT, 1), (FLAT, -3)])

    def test_a_change_to_fewer_sharps_cancels_the_dropped_ones_first(self):
        self.assertEqual(key_signature_accidentals(1, 3, TREBLE), [(NATURAL, 1), (NATURAL, 5), (SHARP, 4)])

    def test_sharps_to_flats_cancels_every_sharp(self):
        drawn = key_signature_accidentals(-1, 2, TREBLE)
        self.assertEqual([label for label, _ in drawn], [NATURAL, NATURAL, FLAT])


class TimeSignatureTest(unittest.TestCase):
    def test_numbers_stack_with_each_digit_a_glyph(self):
        rows = time_signature_rows(TimeSignature(12, 8, TimeSymbol.normal))
        self.assertEqual(rows, [([SmuflLabels.timeSig1.value, SmuflLabels.timeSig2.value], 2),
                                ([SmuflLabels.timeSig8.value], -2)])

    def test_common_time_is_one_glyph_on_the_middle_line(self):
        rows = time_signature_rows(TimeSignature(4, 4, TimeSymbol.common))
        self.assertEqual(rows, [([SmuflLabels.timeSigCommon.value], 0)])


class BarlineTest(unittest.TestCase):
    def test_a_final_barline_is_thin_then_heavy(self):
        self.assertEqual(barline_parts(BarStyle.light_heavy, False, False), [THIN, HEAVY])

    def test_a_backward_repeat_puts_its_dots_before_the_strokes(self):
        self.assertEqual(barline_parts(BarStyle.light_heavy, True, False), [DOTS, THIN, HEAVY])

    def test_back_to_back_repeats_share_the_heavy_stroke(self):
        self.assertEqual(barline_parts(BarStyle.regular, True, True), [DOTS, THIN, HEAVY, THIN, DOTS])

    def test_a_system_start_draws_only_a_forward_repeat(self):
        self.assertEqual(barline_parts(None, False, False), [])
        self.assertEqual(barline_parts(None, False, True), [HEAVY, THIN, DOTS])


class QuilterSeamTest(unittest.TestCase):
    def test_the_seam_follows_the_cheapest_column_moving_at_most_one_per_row(self):
        errors = np.ones((5, 4))
        cheap = [0, 1, 2, 2, 3]
        for row, column in enumerate(cheap):
            errors[row, column] = 0
        self.assertEqual(list(Quilter.min_error_path(errors)), cheap)


if __name__ == "__main__":
    unittest.main()
