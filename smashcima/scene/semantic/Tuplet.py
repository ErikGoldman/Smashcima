from dataclasses import dataclass, field
from typing import List, Optional

from ..SceneObject import SceneObject
from .Durable import Durable


@dataclass
class Tuplet(SceneObject):
    """A group of notes and rests played in another number's time, such as
    a triplet (3 in the time of 2)"""

    durables: List[Durable] = field(default_factory=list)
    "The notes and rests of the group, in order"

    actual_notes: int = 3
    "How many notes are played (the 3 of a triplet)"

    number_shown: bool = True
    "Whether the number is printed (MusicXML show-number other than none)"

    placement: Optional[str] = None
    "'above', 'below', or None when not specified"

    @classmethod
    def of_durable_or_none(cls, durable: Durable) -> Optional["Tuplet"]:
        return cls.of_or_none(durable, lambda t: t.durables)
