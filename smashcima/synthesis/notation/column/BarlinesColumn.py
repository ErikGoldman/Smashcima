from smashcima.geometry.Point import Point
from smashcima.scene.visual.StaffVisual import StaffVisual
from smashcima.scene.Glyph import Glyph
from smashcima.scene.SmuflLabels import SmuflLabels
from smashcima.scene.semantic.Barline import BarStyle
from smashcima.scene.semantic.Score import Score
from smashcima.synthesis.GlyphSynthesizer import GlyphSynthesizer
from .ColumnBase import ColumnBase
from typing import Dict, List, Optional
import random

THIN = SmuflLabels.barlineSingle.value
HEAVY = SmuflLabels.barlineHeavy.value
DOTS = "dots"
"Two repeat dots, in the spaces either side of the middle line"

STYLE_STROKES: Dict[BarStyle, List[str]] = {
    BarStyle.regular: [THIN],
    BarStyle.light_light: [THIN, THIN],
    BarStyle.light_heavy: [THIN, HEAVY],
    BarStyle.heavy_light: [HEAVY, THIN],
}


def barline_parts(
    bar_style: Optional[BarStyle],
    repeat_end: bool,
    repeat_start: bool
) -> List[str]:
    """What a barline is made of, left to right: the style of the measure it
    closes (None at a system's start, where only a forward repeat is drawn),
    with repeat dots on the side of each repeat."""
    if repeat_end and repeat_start:
        return [DOTS, THIN, HEAVY, THIN, DOTS]
    if repeat_end:
        return [DOTS, THIN, HEAVY]
    if repeat_start:
        return [HEAVY, THIN, DOTS]
    return STYLE_STROKES[bar_style] if bar_style is not None else []


class BarlinesColumn(ColumnBase):
    # TODO: synthesize tall barlines for piano (multi-staff parts)

    def __post_init__(self):
        # this column should not be grown much, really
        self.flex_grow = 0.1
        self.parts_by_staff: Dict[int, List[List[Glyph]]] = {}
        self.gap = self.rng.uniform(0.4, 0.8)

    def add_part(self, staff_index: int, glyphs: List[Glyph]):
        for glyph in glyphs:
            self.add_glyph(glyph)
        self.parts_by_staff.setdefault(staff_index, []).append(glyphs)

    def _position_glyphs(self):
        for staff_index, parts in self.parts_by_staff.items():
            staff = self.staves[staff_index]
            x = self.time_position
            for glyphs in parts:
                widths = [g.get_bbox_in_space(g.space).width for g in glyphs]
                left = min(g.get_bbox_in_space(g.space).left for g in glyphs)
                for glyph in glyphs:
                    # a single stroke is centred on the staff, a dot pair on
                    # the spaces either side of the middle line
                    pitch_position = 0 if len(glyphs) == 1 \
                        else (1 if glyph is glyphs[0] else -1)
                    glyph.space.transform = staff.staff_coordinate_system \
                        .get_transform(
                            pitch_position=pitch_position,
                            time_position=x - left
                        )
                x += max(widths) + self.gap


def synthesize_barlines_column(
    staves: List[StaffVisual],
    rng: random.Random,
    glyph_synthesizer: GlyphSynthesizer,
    parts: Optional[List[str]] = None
) -> Optional[BarlinesColumn]:
    """A barline of the given parts on every staff (a single thin barline by
    default); None when there is nothing to draw."""
    parts = [THIN] if parts is None else parts
    if len(parts) == 0:
        return None
    column = BarlinesColumn(staves, rng.random())

    def glyph(staff: StaffVisual, label: str) -> Glyph:
        return glyph_synthesizer.synthesize_glyph_at(
            label=label,
            parent_space=staff.space,
            point=Point(0, 0) # glyph positioned later
        )

    for staff_index, staff in enumerate(staves):
        for part in parts:
            if part == DOTS:
                column.add_part(staff_index, [
                    glyph(staff, SmuflLabels.repeatDot.value),
                    glyph(staff, SmuflLabels.repeatDot.value),
                ])
            else:
                column.add_part(staff_index, [glyph(staff, part)])

    return column


def measure_barline_parts(
    score: Score,
    measure_index: int,
    system_end: bool
) -> List[str]:
    """The barline closing a measure; a forward repeat opening the next
    measure joins it, unless the system ends here"""
    closing = score.parts[0].measures[measure_index]
    opens_repeat = not system_end \
        and measure_index + 1 < score.measure_count \
        and score.parts[0].measures[measure_index + 1].repeat_start
    return barline_parts(closing.right_bar_style, closing.repeat_end, opens_repeat)
