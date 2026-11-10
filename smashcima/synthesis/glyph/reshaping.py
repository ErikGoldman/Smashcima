"""Glyphs reshaped from others: turned, scaled, or combined"""
from typing import List

import numpy as np

from smashcima.geometry.Point import Point
from smashcima.scene.Glyph import Glyph
from smashcima.scene.Sprite import Sprite

CUT_TIME_STROKE_HEIGHT = 1.5
"Height of the stroke through a cut-time C, relative to the C"


def _rebuilt(glyph: Glyph, sprites: List[Sprite]) -> Glyph:
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


def scaled_to_height(glyph: Glyph, height: float) -> Glyph:
    """The glyph scaled about its origin to the given height (millimetres)"""
    current = glyph.get_bbox_in_space(glyph.space).height
    return scaled(glyph, height / current) if current > 0 else glyph
