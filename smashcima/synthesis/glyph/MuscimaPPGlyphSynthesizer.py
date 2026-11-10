import random
from typing import Callable, Dict

from smashcima.assets.AssetRepository import AssetRepository
from smashcima.assets.glyphs.muscima_pp.MuscimaPPGlyphs import MuscimaPPGlyphs
from smashcima.scene.Glyph import Glyph
from smashcima.synthesis.style.MuscimaPPStyleDomain import MuscimaPPStyleDomain

from ...scene.SmuflLabels import SmuflLabels
from ...scene.SmashcimaLabels import SmashcimaLabels
from .RepositoryGlyphSynthesizer import RepositoryGlyphSynthesizer
from .reshaping import half_turned, scaled, with_stroke_through

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

    SmuflLabels.articAccentAbove.value: SmuflLabels.articAccentAbove.value,
    SmuflLabels.articAccentBelow.value: SmuflLabels.articAccentAbove.value,
    SmuflLabels.articTenutoAbove.value: SmuflLabels.articTenutoAbove.value,
    SmuflLabels.articTenutoBelow.value: SmuflLabels.articTenutoAbove.value,

    # curves, stretched to fit when drawn
    SmashcimaLabels.tie.value: SmashcimaLabels.tie.value,
    SmashcimaLabels.slur.value: SmashcimaLabels.slur.value,

    # tuplets, barlines
    SmuflLabels.tuplet3.value: SmuflLabels.tuplet3.value,
    SmuflLabels.barlineHeavy.value: SmuflLabels.barlineHeavy.value,
    SmuflLabels.repeatDot.value: SmuflLabels.repeatDot.value,

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

