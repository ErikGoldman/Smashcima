from dataclasses import dataclass
from typing import Optional

from ..SceneObject import SceneObject
from .Note import Note


@dataclass
class Slur(SceneObject):
    """A slur from one note to another. Either end is None when the slur
    begins before, or ends after, the music loaded."""

    start: Optional[Note] = None
    "The note the slur begins at"

    end: Optional[Note] = None
    "The note the slur ends at"

    placement: Optional[str] = None
    "'above', 'below', or None when not specified"

    @classmethod
    def starting_at(cls, note: Note) -> "list[Slur]":
        return cls.many_of(note, lambda s: s.start)

    @classmethod
    def ending_at(cls, note: Note) -> "list[Slur]":
        return cls.many_of(note, lambda s: s.end)
