from dataclasses import dataclass, field
from typing import List, Optional
from .Durable import Durable
from .Pitch import Pitch
from .AccidentalValue import AccidentalValue


@dataclass
class Note(Durable):
    pitch: Pitch
    "Musical pitch of the note"

    accidental_value: Optional[AccidentalValue]
    "What accidental is attached to the note"

    articulations: List[str] = field(default_factory=list)
    "MusicXML articulation names drawn on the note (staccato, accent, tenuto)"

    tie_starts: bool = False
    "A tie leaves this note for the next note of the same pitch"

    tie_stops: bool = False
    "A tie arrives at this note from the previous note of the same pitch"
