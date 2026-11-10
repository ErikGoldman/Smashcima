import random
from typing import Dict

from smashcima.assets.AssetRepository import AssetRepository
from smashcima.assets.glyphs.muscima_pp.MuscimaPPGlyphs import MuscimaPPGlyphs
from smashcima.geometry.Vector2 import Vector2
from smashcima.scene.LineGlyph import LineGlyph
from smashcima.scene.SmashcimaLabels import SmashcimaLabels
from smashcima.scene.SmuflLabels import SmuflLabels
from smashcima.synthesis.style.MuscimaPPStyleDomain import MuscimaPPStyleDomain

from .RepositoryLineSynthesizer import RepositoryLineSynthesizer

LABEL_MAP: Dict[str, str] = {
    SmashcimaLabels.legerLine.value: SmashcimaLabels.legerLine.value,
    SmuflLabels.stem.value: SmuflLabels.stem.value,
    SmashcimaLabels.beam.value: SmashcimaLabels.beam.value,
    SmashcimaLabels.beamHook.value: SmashcimaLabels.beamHook.value,
    SmuflLabels.brace.value: SmuflLabels.brace.value,
    SmuflLabels.bracket.value: SmuflLabels.bracket.value,
    # drawn with stem strokes, for want of MUSCIMA++ samples of its own
    SmashcimaLabels.systemStartLine.value: SmuflLabels.stem.value,
}


class MuscimaPPLineSynthesizer(RepositoryLineSynthesizer):
    """Synthesizes line glyphs by sampling from the MUSCIMA++ dataset"""
    
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
        return label in LABEL_MAP
    
    def create_glyph(self, label: str, delta: Vector2) -> LineGlyph:
        glyph = super().create_glyph(LABEL_MAP[label], delta)

        # adjust glyph label to match what the user wants
        # (because the label map is not 1:1)
        glyph.label = label

        return glyph
