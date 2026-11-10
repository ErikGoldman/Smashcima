"""Marks drawn once a system's notes, stems and beams are in place: ties,
slurs, tuplet numbers and articulations."""
import random
from dataclasses import dataclass
from typing import Iterator, List, Optional

import cv2
import numpy as np

from smashcima.geometry.Point import Point
from smashcima.geometry.Rectangle import Rectangle
from smashcima.geometry.Transform import Transform
from smashcima.geometry.units import mm_to_px
from smashcima.scene.AffineSpace import AffineSpace
from smashcima.scene.Glyph import Glyph
from smashcima.scene.SmashcimaLabels import SmashcimaLabels
from smashcima.scene.SmuflLabels import SmuflLabels
from smashcima.scene.Sprite import Sprite
from smashcima.scene.semantic.Chord import Chord
from smashcima.scene.semantic.Durable import Durable
from smashcima.scene.semantic.Note import Note
from smashcima.scene.semantic.Slur import Slur
from smashcima.scene.semantic.StemValue import StemValue
from smashcima.scene.semantic.Tuplet import Tuplet
from smashcima.scene.visual.Notehead import Notehead
from smashcima.scene.visual.RestVisual import RestVisual
from smashcima.scene.visual.StaffVisual import StaffVisual
from smashcima.scene.visual.Stem import Stem

from ..GlyphSynthesizer import GlyphSynthesizer

ARTICULATION_LABELS = {
    "staccato": SmuflLabels.articStaccatoAbove.value,
    "accent": SmuflLabels.articAccentAbove.value,
    "tenuto": SmuflLabels.articTenutoAbove.value,
}

MAX_THICKENING_PX = 3
"At most this much is put back: more swells a curve's steep ends into blobs"

CURVE_CANDIDATES = 8
"Curve samples drawn per tie or slur; the one shaped most like the target wins"


@dataclass
class SystemNotes:
    """The notes and rests of one system, in order, with what was drawn
    for them."""
    page_space: AffineSpace
    staves: List[StaffVisual]
    durables: List[Durable]
    left_x: float
    "Where music begins after the system header (page space)"
    right_x: float
    "Where the system's staves end (page space)"

    def notes(self) -> Iterator[Note]:
        return (d for d in self.durables if isinstance(d, Note))

    def bbox(self, durable: Durable) -> Rectangle:
        if isinstance(durable, Note):
            glyph = Notehead.of_note(durable).glyph
        else:
            glyph = RestVisual.of_rest(durable).glyph
        return glyph.get_bbox_in_space(self.page_space)

    def staff_of(self, durable: Durable) -> StaffVisual:
        if isinstance(durable, Note):
            return Notehead.of_note(durable).staff
        return RestVisual.of_rest(durable).staff

    def staff_space(self, durable: Durable) -> float:
        return self.staff_of(durable).staff_coordinate_system.staff_space


def stem_points_up(note: Note) -> Optional[bool]:
    chord = Chord.of_note(note)
    if chord.stem_value == StemValue.up:
        return True
    if chord.stem_value == StemValue.down:
        return False
    return None


def ink_rows(sprite: Sprite) -> np.ndarray:
    return sprite.bitmap[:, :, 3] > 127


def arches_up(sprite: Sprite) -> bool:
    """Whether a curve sample bows upwards (a slur above its notes): its
    middle lies higher than its ends."""
    ink = ink_rows(sprite)
    ys, xs = np.nonzero(ink)
    width = ink.shape[1]
    middle = ys[(xs > width / 3) & (xs < 2 * width / 3)]
    ends = ys[(xs < width / 8) | (xs > 7 * width / 8)]
    if len(middle) == 0 or len(ends) == 0:
        return True
    return middle.mean() < ends.mean()


def pen_width_px(bitmap: np.ndarray) -> float:
    """Stroke thickness: twice the typical distance from the ink's middle
    to its edge"""
    ink = (bitmap[:, :, 3] > 127).astype(np.uint8)
    if ink.sum() == 0:
        return 0.0
    depth = cv2.distanceTransform(np.pad(ink, 1), cv2.DIST_L2, 3)
    return 2 * float(np.percentile(depth[depth > 0], 90))


def with_pen_width_of(curve: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """The stretched curve with ink made solid again and its stroke brought
    back to the sample's: squashing a sample thins it, enlarging thickens it"""
    alpha = np.where(curve[:, :, 3] > 40, 255, 0).astype(np.uint8)
    change = pen_width_px(sample) - pen_width_px(
        np.dstack([np.zeros_like(alpha)] * 3 + [alpha])
    )
    if change >= 1:
        kernel = np.ones((int(round(min(MAX_THICKENING_PX, change))) + 1, 1), np.uint8)
        alpha = cv2.dilate(alpha, kernel)
    elif change <= -1:
        kernel = np.ones((int(round(-change)) + 1, 1), np.uint8)
        alpha = cv2.erode(alpha, kernel)
    return np.dstack([np.zeros_like(alpha)] * 3 + [alpha])


def resized(bitmap: np.ndarray, width_px: int, height_px: int) -> np.ndarray:
    """Area averaging when shrinking; when enlarging it repeats pixels into
    steps, so interpolate instead"""
    shrinking = width_px <= bitmap.shape[1] and height_px <= bitmap.shape[0]
    return cv2.resize(bitmap, (width_px, height_px),
                      interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


def curve_glyph(
    sample: Glyph,
    label: str,
    page_space: AffineSpace,
    start: Point,
    end: Point,
    height: float,
    above: bool
) -> Glyph:
    """A tie or slur from start to end (page space, millimetres) bowing by
    height, above or below the line between them: the sample curve stretched
    and sheared onto that line."""
    sprite = sample.sprites[0]
    bitmap = sprite.bitmap
    if arches_up(sprite) != above:
        bitmap = bitmap[::-1]
    width_px = max(2, int(round(mm_to_px(end.x - start.x, sprite.dpi))))
    height_px = max(2, int(round(mm_to_px(height, sprite.dpi))))
    curve = with_pen_width_of(resized(bitmap, width_px, height_px), bitmap)

    # shear columns so the ends meet start and end; the ends of a curve
    # bowing up sit on its bottom edge, of one bowing down on its top edge
    rise_px = mm_to_px(end.y - start.y, sprite.dpi)
    pad = int(np.ceil(abs(rise_px))) + 1
    shift = pad if rise_px < 0 else 0
    shear = np.float32([[1, 0, 0], [rise_px / width_px, 1, shift]])
    canvas = cv2.warpAffine(
        curve, shear, (width_px, height_px + pad),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    start_row = shift + (height_px if above else 0)

    space = AffineSpace(
        parent_space=page_space,
        transform=Transform.translate(start.vector)
    )
    curve_sprite = Sprite(
        space=space,
        bitmap=np.ascontiguousarray(canvas),
        bitmap_origin=Point(0, start_row / canvas.shape[0]),
        dpi=sprite.dpi
    )
    return Glyph(
        space=space,
        region=Glyph.build_region_from_sprites_alpha_channel(
            label=label, sprites=[curve_sprite]
        ),
        sprites=[curve_sprite]
    )


class MarksSynthesizer:
    """Draws ties, slurs, tuplet numbers and articulations of a system"""

    def __init__(self, glyph_synthesizer: GlyphSynthesizer, rng: random.Random):
        self.glyph_synthesizer = glyph_synthesizer
        self.rng = rng

    def synthesize_marks(self, system: SystemNotes):
        self.synthesize_ties(system)
        self.synthesize_slurs(system)
        self.synthesize_tuplet_numbers(system)
        self.synthesize_articulations(system)

    # --- curves ---

    def pick_curve(self, label: str, width: float, height: float) -> Glyph:
        """The sample of the label shaped and sized most like the target
        (millimetres), among a few drawn at random: stretched little, a
        sample keeps its stroke; enlarged a lot, a small one turns blocky"""
        candidates = [
            self.glyph_synthesizer.create_glyph(label)
            for _ in range(CURVE_CANDIDATES)
        ]

        def mismatch(glyph: Glyph) -> float:
            sprite = glyph.sprites[0]
            h, w = sprite.bitmap.shape[:2]
            target_h = mm_to_px(height, sprite.dpi)
            return abs(np.log((w / h) / (width / height))) \
                + abs(np.log(target_h / h))

        return min(candidates, key=mismatch)

    def draw_curve(
        self,
        label: str,
        page_space: AffineSpace,
        start: Point,
        end: Point,
        height: float,
        above: bool
    ):
        if end.x - start.x <= 0:
            return
        sample = self.pick_curve(label, end.x - start.x, height)
        curve_glyph(sample, label, page_space, start, end, height, above)

    def synthesize_ties(self, system: SystemNotes):
        notes = list(system.notes())
        for i, note in enumerate(notes):
            if note.tie_starts:
                target = next(
                    (n for n in notes[i + 1:]
                     if n.tie_stops and n.pitch == note.pitch
                     and system.staff_of(n) is system.staff_of(note)),
                    None
                )
                self.draw_tie(system, note, target)
            if note.tie_stops and not any(
                n.tie_starts and n.pitch == note.pitch
                and system.staff_of(n) is system.staff_of(note)
                for n in notes[:i]
            ):
                self.draw_tie(system, None, note)

    def tie_above(self, note: Note) -> bool:
        """Ties bow away from the stem; in a chord, the upper half of the
        notes tie above and the lower half below"""
        chord = Chord.of_note(note)
        if len(chord.notes) > 1:
            pitches = sorted(n.pitch.get_linear_pitch() for n in chord.notes)
            return note.pitch.get_linear_pitch() >= pitches[len(pitches) // 2]
        up = stem_points_up(note)
        return not up if up is not None else True

    def draw_tie(
        self,
        system: SystemNotes,
        start: Optional[Note],
        end: Optional[Note],
    ):
        """A tie between two notes; with one of them missing, a half tie
        from the system's start or to its end"""
        note = start or end
        assert note is not None
        space = system.staff_space(note)
        above = self.tie_above(note)
        gap = space * self.rng.uniform(0.1, 0.3)

        def anchor(n: Note, right_side: bool) -> Point:
            box = system.bbox(n)
            x = box.right + gap if right_side else box.left - gap
            y = box.top - gap if above else box.bottom + gap
            return Point(x, y + (space * 0.15 if above else -space * 0.15))

        a = anchor(start, True) if start is not None else None
        b = anchor(end, False) if end is not None else None
        if a is None:
            a = Point(system.left_x, b.y)
        if b is None:
            b = Point(system.right_x, a.y)
        length = b.x - a.x
        height = min(space * 0.9, max(space * 0.35, length * 0.15)) \
            * self.rng.uniform(0.8, 1.2)
        self.draw_curve(
            SmashcimaLabels.tie.value, system.page_space,
            a, b, height, above
        )

    def synthesize_slurs(self, system: SystemNotes):
        notes = list(system.notes())
        seen = set()
        for note in notes:
            for slur in Slur.starting_at(note) + Slur.ending_at(note):
                if id(slur) in seen:
                    continue
                seen.add(id(slur))
                start = slur.start if slur.start in notes else None
                end = slur.end if slur.end in notes else None
                self.draw_slur(system, slur, start, end)

    def slur_above(self, slur: Slur, note: Note) -> bool:
        if slur.placement in ("above", "below"):
            return slur.placement == "above"
        up = stem_points_up(note)
        return not up if up is not None else True

    def draw_slur(
        self,
        system: SystemNotes,
        slur: Slur,
        start: Optional[Note],
        end: Optional[Note]
    ):
        note = start or end
        if note is None:
            return
        space = system.staff_space(note)
        above = self.slur_above(slur, note)
        gap = space * self.rng.uniform(0.3, 0.7)

        def anchor(n: Note) -> Point:
            """Just beyond the notehead on the slur's side, or beyond the
            stem tip when the stem points that way"""
            box = system.bbox(n)
            x = box.center.x
            y = box.top - gap if above else box.bottom + gap
            stem = Stem.of_chord_or_none(Chord.of_note(n))
            up = stem_points_up(n)
            if stem is not None and up is not None and up == above:
                tip = stem.tip.transform_to(system.page_space)
                y = tip.y - gap if above else tip.y + gap
                x = tip.x
            return Point(x, y)

        a = anchor(start) if start is not None else None
        b = anchor(end) if end is not None else None
        if a is None:
            a = Point(system.left_x, b.y)
        if b is None:
            b = Point(system.right_x, a.y)
        length = b.x - a.x
        height = min(space * 2.0, max(space * 0.5, length * 0.12)) \
            * self.rng.uniform(0.8, 1.25)
        self.draw_curve(
            SmashcimaLabels.slur.value, system.page_space,
            a, b, height, above
        )

    # --- tuplet numbers ---

    def synthesize_tuplet_numbers(self, system: SystemNotes):
        seen = set()
        for durable in system.durables:
            tuplet = Tuplet.of_durable_or_none(durable)
            if tuplet is None or id(tuplet) in seen:
                continue
            seen.add(id(tuplet))
            members = [d for d in tuplet.durables if d in system.durables]
            if not tuplet.number_shown or tuplet.actual_notes != 3 \
                    or len(members) == 0:
                continue
            self.draw_tuplet_number(system, tuplet, members)

    def draw_tuplet_number(
        self,
        system: SystemNotes,
        tuplet: Tuplet,
        members: List[Durable]
    ):
        notes = [d for d in members if isinstance(d, Note)]
        ups = [stem_points_up(n) for n in notes]
        if tuplet.placement in ("above", "below"):
            above = tuplet.placement == "above"
        else:
            above = not any(u is False for u in ups)
        # the extreme of noteheads, rests and stem tips on that side
        ys = []
        for d in members:
            box = system.bbox(d)
            ys.append(box.top if above else box.bottom)
            if isinstance(d, Note):
                stem = Stem.of_chord_or_none(Chord.of_note(d))
                if stem is not None:
                    ys.append(stem.tip.transform_to(system.page_space).y)
        space = system.staff_space(members[0])
        y = (min(ys) - space * self.rng.uniform(0.6, 1.1)) if above \
            else (max(ys) + space * self.rng.uniform(0.6, 1.1))
        first, last = system.bbox(members[0]), system.bbox(members[-1])
        x = (first.center.x + last.center.x) / 2 + space * self.rng.uniform(-0.2, 0.2)
        self.glyph_synthesizer.synthesize_glyph_at(
            label=SmuflLabels.tuplet3.value,
            parent_space=system.page_space,
            point=Point(x, y)
        )

    # --- articulations ---

    def synthesize_articulations(self, system: SystemNotes):
        done_chords = set()
        for note in system.notes():
            chord = Chord.of_note(note)
            if id(chord) in done_chords:
                continue
            done_chords.add(id(chord))
            names = []
            for n in chord.notes:
                names += [a for a in n.articulations if a not in names]
            if len(names) > 0:
                self.draw_articulations(system, chord, names)

    def draw_articulations(self, system: SystemNotes, chord: Chord, names: List[str]):
        """Articulations stacked beyond the chord's outer notehead, on the
        side away from the stem"""
        up = stem_points_up(chord.notes[0])
        above = not up if up is not None else True
        boxes = [system.bbox(n) for n in chord.notes]
        outer = min(boxes, key=lambda b: b.top) if above \
            else max(boxes, key=lambda b: b.bottom)
        space = system.staff_space(chord.notes[0])
        y = outer.top if above else outer.bottom
        step = -1 if above else 1
        for name in names:
            y += step * space * self.rng.uniform(0.7, 1.0)
            glyph = self.glyph_synthesizer.synthesize_glyph_at(
                label=ARTICULATION_LABELS[name],
                parent_space=system.page_space,
                point=Point(outer.center.x + space * self.rng.uniform(-0.1, 0.1), y)
            )
            height = glyph.get_bbox_in_space(system.page_space).height
            y += step * height / 2
