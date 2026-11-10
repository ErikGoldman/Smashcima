from enum import Enum


class BarStyle(str, Enum):
    """How a barline is drawn (the MusicXML bar-style values we draw)"""
    # https://www.w3.org/2021/06/musicxml40/musicxml-reference/data-types/bar-style/
    regular = "regular"
    light_light = "light-light"
    light_heavy = "light-heavy"
    heavy_light = "heavy-light"
