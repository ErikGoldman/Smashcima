import random
from typing import Callable, Dict

import numpy as np

from smashcima.assets.AssetRepository import AssetRepository
from smashcima.assets.glyphs.muscima_pp.MuscimaPPGlyphs import MuscimaPPGlyphs
from smashcima.geometry.Point import Point
from smashcima.scene.Glyph import Glyph
from smashcima.scene.Sprite import Sprite
from smashcima.synthesis.style.MuscimaPPStyleDomain import MuscimaPPStyleDomain

from ...scene.SmuflLabels import SmuflLabels
from .RepositoryGlyphSynthesizer import RepositoryGlyphSynthesizer

LABEL_MAP: Dict[str, str] = {
    # barlines
    SmuflLabels.barlineSingle.value: SmuflLabels.barlineSingle.value,

    # clefs    (clefs ignore the normal/small distinction)
    SmuflLabels.gClef.value: SmuflLabels.gClef.value,
    SmuflLabels.gClefSmall.value: SmuflLabels.gClef.value,
    SmuflLabels.fClef.value: SmuflLabels.fClef.value,
    SmuflLabels.fClefSmall.value: SmuflLabels.fClef.value,
    SmuflLabels.cClef.value: SmuflLabels.cClef.value,
    SmuflLabels.cClefSmall.value: SmuflLabels.cClef.value,

    # noteheads
    SmuflLabels.noteheadWhole.value: SmuflLabels.noteheadWhole.value,
    SmuflLabels.noteheadHalf.value: SmuflLabels.noteheadWhole.value,
    SmuflLabels.noteheadBlack.value: SmuflLabels.noteheadBlack.value,

    # augmentation dot
    SmuflLabels.augmentationDot.value:SmuflLabels.augmentationDot.value,

    # flags
    SmuflLabels.flag8thUp.value: SmuflLabels.flag8thUp.value,
    SmuflLabels.flag8thDown.value: SmuflLabels.flag8thDown.value,
    SmuflLabels.flag16thUp.value: SmuflLabels.flag16thUp.value,
    SmuflLabels.flag16thDown.value: SmuflLabels.flag16thDown.value,

    # accidentals
    SmuflLabels.accidentalDoubleSharp.value: SmuflLabels.accidentalDoubleSharp.value,
    SmuflLabels.accidentalFlat.value: SmuflLabels.accidentalFlat.value,
    SmuflLabels.accidentalNatural.value: SmuflLabels.accidentalNatural.value,
    SmuflLabels.accidentalSharp.value: SmuflLabels.accidentalSharp.value,

    # rests
    SmuflLabels.restWhole.value: SmuflLabels.restWhole.value,
    SmuflLabels.restHalf.value: SmuflLabels.restHalf.value,
    SmuflLabels.restQuarter.value: SmuflLabels.restQuarter.value,
    SmuflLabels.rest8th.value: SmuflLabels.rest8th.value,
    SmuflLabels.rest16th.value: SmuflLabels.rest16th.value,

    # articulations (MUSCIMA++ does not tell above from below)
    SmuflLabels.articStaccatoAbove.value: SmuflLabels.articStaccatoBelow.value,
    SmuflLabels.articStaccatoBelow.value: SmuflLabels.articStaccatoBelow.value,

    # time signatures
    **{
        label.value: label.value for label in [
            SmuflLabels.timeSig0, SmuflLabels.timeSig1, SmuflLabels.timeSig2,
            SmuflLabels.timeSig3, SmuflLabels.timeSig4, SmuflLabels.timeSig5,
            SmuflLabels.timeSig6, SmuflLabels.timeSig7, SmuflLabels.timeSig8,
            SmuflLabels.timeSigCommon,
        ]
    },
}

SMALL_CLEF_SCALE = 0.7
CUT_TIME_STROKE_HEIGHT = 1.5
"Height of the stroke through a cut-time C, relative to the C"


class MuscimaPPGlyphSynthesizer(RepositoryGlyphSynthesizer):
    """Synthesizes glyphs by sampling from the MUSCIMA++ dataset"""
    
    def __init__(
        self,
        assets: AssetRepository,
        style_domain: MuscimaPPStyleDomain,
        rng: random.Random
    ):
        symbol_repository = (
            assets.resolve_bundle(MuscimaPPGlyphs).load_symbol_repository()
        )

        super().__init__(
            symbol_repository=symbol_repository,
            style_domain=style_domain,
            rng=rng
        )

    def supports_label(self, label: str) -> bool:
        return label in LABEL_MAP or label in self._derived_glyphs

    @property
    def _derived_glyphs(self) -> Dict[str, Callable[[], Glyph]]:
        """Glyphs MUSCIMA++ lacks, made from ones it has"""
        return {
            SmuflLabels.timeSig9.value: lambda: half_turned(
                super(MuscimaPPGlyphSynthesizer, self).create_glyph(
                    SmuflLabels.timeSig6.value
                )
            ),
            SmuflLabels.timeSigCutCommon.value: lambda: with_stroke_through(
                super(MuscimaPPGlyphSynthesizer, self).create_glyph(
                    SmuflLabels.timeSigCommon.value
                ),
                super(MuscimaPPGlyphSynthesizer, self).create_glyph(
                    SmuflLabels.barlineSingle.value
                ),
            ),
            **{
                small.value: (lambda normal=normal: scaled(
                    super(MuscimaPPGlyphSynthesizer, self).create_glyph(
                        normal.value
                    ),
                    SMALL_CLEF_SCALE
                ))
                for small, normal in [
                    (SmuflLabels.gClefSmall, SmuflLabels.gClef),
                    (SmuflLabels.fClefSmall, SmuflLabels.fClef),
                    (SmuflLabels.cClefSmall, SmuflLabels.cClef),
                ]
            },
        }

    def create_glyph(self, label: str) -> Glyph:
        derived = self._derived_glyphs.get(label)
        if derived is not None:
            glyph = derived()
            glyph.label = label
            return glyph

        glyph = super().create_glyph(LABEL_MAP[label])

        # adjust glyph label to match what the user wants
        # (because the label map is not 1:1)
        glyph.label = label

        return glyph


def _rebuilt(glyph: Glyph, sprites: "list[Sprite]") -> Glyph:
    glyph.sprites = sprites
    glyph.region = Glyph.build_region_from_sprites_alpha_channel(
        label=glyph.label,
        sprites=sprites
    )
    return glyph


def half_turned(glyph: Glyph) -> Glyph:
    """The glyph rotated by 180 degrees about its origin (a 6 into a 9)"""
    for sprite in glyph.sprites:
        sprite.bitmap = np.ascontiguousarray(np.rot90(sprite.bitmap, 2))
        sprite.bitmap_origin = Point(
            1 - sprite.bitmap_origin.x,
            1 - sprite.bitmap_origin.y
        )
    return _rebuilt(glyph, glyph.sprites)


def scaled(glyph: Glyph, factor: float) -> Glyph:
    """The glyph drawn smaller or larger about its origin. Scene transforms
    keep scale by convention, so the sprites' DPI changes instead."""
    for sprite in glyph.sprites:
        sprite.dpi = sprite.dpi / factor
    return _rebuilt(glyph, glyph.sprites)


def with_stroke_through(glyph: Glyph, stroke: Glyph) -> Glyph:
    """The glyph with a vertical stroke through its origin (a C into a cut C):
    the stroke's sprites moved into the glyph and scaled to its height."""
    height = glyph.get_bbox_in_space(glyph.space).height
    stroke_height = stroke.get_bbox_in_space(stroke.space).height
    for sprite in stroke.sprites:
        sprite.space = glyph.space
        sprite.dpi = sprite.dpi * stroke_height / (CUT_TIME_STROKE_HEIGHT * height)
        sprite.bitmap_origin = Point(sprite.bitmap_origin.x, 0.5)
    return _rebuilt(glyph, glyph.sprites + stroke.sprites)
