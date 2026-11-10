"""Key and time signature columns: at the start of a system and where they
change within it."""
import random
from typing import Dict, List, Optional, Tuple

from smashcima.geometry.Point import Point
from smashcima.random_between import random_between
from smashcima.scene.Glyph import Glyph
from smashcima.scene.SmuflLabels import SmuflLabels
from smashcima.scene.semantic.Clef import Clef
from smashcima.scene.semantic.Score import Score
from smashcima.scene.semantic.TimeSignature import TimeSignature, TimeSymbol
from smashcima.scene.visual.StaffVisual import StaffVisual
from smashcima.synthesis.GlyphSynthesizer import GlyphSynthesizer

from .ColumnBase import ColumnBase

SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"

# Where each accidental of a key signature sits on a treble staff
# (pitch position: 0 is the middle line, one step per line or space).
TREBLE_SHARP_POSITIONS = {"F": 4, "C": 1, "G": 5, "D": 2, "A": -1, "E": 3, "B": 0}
TREBLE_FLAT_POSITIONS = {"B": 0, "E": 3, "A": -1, "D": 2, "G": -2, "C": 1, "F": -3}


def key_signature_shift(clef: Clef) -> int:
    """How many steps below the treble layout a key signature sits under this
    clef: the treble pattern moved by the letter of the clef's middle line,
    folded into -3..3 (bass clef: 2, alto: 1)."""
    middle = clef.pitch_position_to_pitch(0)
    shift = ("CDEFGAB".index(middle.step.value) - "CDEFGAB".index("B")) % 7
    return shift - 7 if shift > 3 else shift


def key_signature_accidentals(
    fifths: int,
    previous_fifths: int,
    clef: Clef
) -> List[Tuple[str, int]]:
    """(SMuFL label, pitch position) of each accidental drawn for a key,
    left to right. Naturals first cancel the previous key's accidentals that
    the new key drops (all of them when sharps turn to flats or back)."""
    def accidentals_of(f: int) -> List[Tuple[str, str]]:
        if f > 0:
            return [("sharp", step) for step in SHARP_ORDER[:f]]
        return [("flat", step) for step in FLAT_ORDER[:-f]]

    def position(kind: str, step: str) -> int:
        table = TREBLE_SHARP_POSITIONS if kind == "sharp" else TREBLE_FLAT_POSITIONS
        return table[step] - key_signature_shift(clef)

    new = accidentals_of(fifths)
    old = accidentals_of(previous_fifths)
    cancelled = [a for a in old if a not in new]
    labels = {
        "sharp": SmuflLabels.accidentalSharp.value,
        "flat": SmuflLabels.accidentalFlat.value,
    }
    drawn = [(SmuflLabels.accidentalNatural.value, position(kind, step))
             for kind, step in cancelled]
    drawn += [(labels[kind], position(kind, step)) for kind, step in new]
    return drawn


class KeySignaturesColumn(ColumnBase):
    """One key signature per staff, its accidentals placed left to right."""

    def __post_init__(self):
        self.flex_grow = 0.1
        self.accidentals_by_staff: Dict[int, List[Tuple[Glyph, int]]] = {}
        self.spacing = random_between(0.3, 0.9, self.rng)

    def add_accidental(self, staff_index: int, glyph: Glyph, pitch_position: int):
        self.add_glyph(glyph)
        self.accidentals_by_staff.setdefault(staff_index, []) \
            .append((glyph, pitch_position))

    def _position_glyphs(self):
        for staff_index, accidentals in self.accidentals_by_staff.items():
            staff = self.staves[staff_index]
            x = self.time_position
            for glyph, pitch_position in accidentals:
                bbox = glyph.get_bbox_in_space(glyph.space)
                x -= bbox.left  # the glyph's left edge at x
                glyph.space.transform = staff.staff_coordinate_system.get_transform(
                    pitch_position=pitch_position,
                    time_position=x
                )
                x += bbox.right + self.spacing * self.rng.uniform(0.7, 1.3)


def synthesize_key_signatures(
    staves: List[StaffVisual],
    rng: random.Random,
    glyph_synthesizer: GlyphSynthesizer,
    score: Score,
    measure_index: int,
    keys_changed_only: bool
) -> Optional[KeySignaturesColumn]:
    """The key signatures of the first event of a measure. At a system's start
    (keys_changed_only false) every staff shows its current key, without
    cancelling naturals; elsewhere only staves whose key changes there show
    it, with naturals. None when nothing is drawn."""
    column = KeySignaturesColumn(staves, rng.random())
    drawn_any = False
    for part in score.parts:
        event = part.measures[measure_index].first_event
        change = event.attributes_change
        previous = event.previous_attributes
        for staff_number, key in event.attributes.keys.items():
            if keys_changed_only:
                if change is None or staff_number not in change.keys:
                    continue
                if previous is None:
                    continue
                previous_fifths = previous.keys[staff_number].fifths
                if previous_fifths == key.fifths:
                    continue
            else:
                previous_fifths = key.fifths
            staff_index = score.first_staff_index_of_part(part) + staff_number - 1
            clef = event.attributes.clefs[staff_number]
            for label, pitch_position in key_signature_accidentals(
                key.fifths, previous_fifths, clef
            ):
                glyph = glyph_synthesizer.synthesize_glyph_at(
                    label=label,
                    parent_space=staves[staff_index].space,
                    point=Point(0, 0)
                )
                column.add_accidental(staff_index, glyph, pitch_position)
                drawn_any = True
    return column if drawn_any else None


TIME_DIGIT_LABELS = [
    SmuflLabels.timeSig0, SmuflLabels.timeSig1, SmuflLabels.timeSig2,
    SmuflLabels.timeSig3, SmuflLabels.timeSig4, SmuflLabels.timeSig5,
    SmuflLabels.timeSig6, SmuflLabels.timeSig7, SmuflLabels.timeSig8,
    SmuflLabels.timeSig9,
]


class TimeSignaturesColumn(ColumnBase):
    """One time signature per staff: two stacked numbers, or a C symbol."""

    def __post_init__(self):
        self.flex_grow = 0.1
        # per staff: rows of glyphs, each row with its pitch position
        self.rows_by_staff: Dict[int, List[Tuple[List[Glyph], int]]] = {}

    def add_row(self, staff_index: int, glyphs: List[Glyph], pitch_position: int):
        for glyph in glyphs:
            self.add_glyph(glyph)
        self.rows_by_staff.setdefault(staff_index, []) \
            .append((glyphs, pitch_position))

    def _position_glyphs(self):
        for staff_index, rows in self.rows_by_staff.items():
            staff = self.staves[staff_index]
            widths = [sum(g.get_bbox_in_space(g.space).width for g in glyphs)
                      for glyphs, _ in rows]
            for (glyphs, pitch_position), width in zip(rows, widths):
                # rows centered on each other
                x = self.time_position + (max(widths) - width) / 2
                for glyph in glyphs:
                    bbox = glyph.get_bbox_in_space(glyph.space)
                    x -= bbox.left
                    glyph.space.transform = (
                        staff.staff_coordinate_system.get_transform(
                            pitch_position=pitch_position
                                + self.rng.uniform(-0.25, 0.25),
                            time_position=x
                        )
                    )
                    x += bbox.right


def time_signature_rows(time: TimeSignature) -> List[Tuple[List[str], int]]:
    """Glyph labels of each row of a time signature, with its pitch position."""
    if time.symbol == TimeSymbol.common:
        return [([SmuflLabels.timeSigCommon.value], 0)]
    if time.symbol == TimeSymbol.cut:
        return [([SmuflLabels.timeSigCutCommon.value], 0)]
    digits = lambda n: [TIME_DIGIT_LABELS[int(d)].value for d in str(n)]
    return [(digits(time.beats), 2), (digits(time.beat_type), -2)]


def synthesize_time_signatures(
    staves: List[StaffVisual],
    rng: random.Random,
    glyph_synthesizer: GlyphSynthesizer,
    score: Score,
    measure_index: int
) -> Optional[TimeSignaturesColumn]:
    """The time signature set at the first event of a measure, on every staff
    of the part; None when the measure sets none."""
    column = TimeSignaturesColumn(staves, rng.random())
    drawn_any = False
    for part in score.parts:
        change = part.measures[measure_index].first_event.attributes_change
        if change is None or change.time_signature is None:
            continue
        first_staff = score.first_staff_index_of_part(part)
        for staff_index in range(first_staff, first_staff + part.staff_count):
            for labels, pitch_position in time_signature_rows(change.time_signature):
                glyphs = [
                    glyph_synthesizer.synthesize_glyph_at(
                        label=label,
                        parent_space=staves[staff_index].space,
                        point=Point(0, 0)
                    )
                    for label in labels
                ]
                column.add_row(staff_index, glyphs, pitch_position)
                drawn_any = True
    return column if drawn_any else None
