import unittest

from smashcima.loading import load_score
from smashcima.scene.semantic import MeasureRest, RestSemantic


def measure_rests_without_a_time_signature(divisions: int, duration: int) -> str:
    """A whole-measure rest on each of two staves, the second after a <backup>
    of the first's duration, in a measure no time signature measures"""
    rest = lambda staff: (f"<note><rest measure='yes'/><duration>{duration}</duration>"
                          f"<voice>{staff}</voice><staff>{staff}</staff></note>")
    return f"""<?xml version="1.0"?>
<score-partwise><part-list><score-part id="P1"><part-name/></score-part></part-list>
<part id="P1"><measure number="1">
<attributes><divisions>{divisions}</divisions><key><fifths>0</fifths></key><staves>2</staves>
<clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>
{rest(1)}<backup><duration>{duration}</duration></backup>{rest(2)}</measure></part></score-partwise>"""


def quarter_rests(hidden_first: bool) -> str:
    rest = lambda hidden: (f"<note{' print-object=\'no\'' if hidden else ''}><rest/><duration>1</duration>"
                           f"<type>quarter</type></note>")
    return f"""<?xml version="1.0"?>
<score-partwise><part-list><score-part id="P1"><part-name/></score-part></part-list>
<part id="P1"><measure number="1">
<attributes><divisions>1</divisions><key><fifths>0</fifths></key><time><beats>2</beats><beat-type>4</beat-type></time>
<clef><sign>G</sign><line>2</line></clef></attributes>
{rest(hidden_first)}{rest(False)}</measure></part></score-partwise>"""


class HiddenRestTest(unittest.TestCase):
    def test_a_hidden_rest_keeps_its_time_and_is_not_a_durable(self):
        s = load_score(data=quarter_rests(hidden_first=True), format=".musicxml")
        events = s.parts[0].measures[0].events
        rests = [(e.fractional_measure_onset, d) for e in events for d in e.durables if isinstance(d, RestSemantic)]
        self.assertEqual([onset for onset, _ in rests], [1])


class MeasureRestTest(unittest.TestCase):
    def test_a_measure_rest_lasts_its_own_duration_when_no_time_is_known(self):
        s = load_score(data=measure_rests_without_a_time_signature(8, 24), format=".musicxml")
        events = s.parts[0].measures[0].events
        rests = [d for e in events for d in e.durables if isinstance(d, MeasureRest)]
        self.assertEqual([r.fractional_duration for r in rests], [3, 3])
        self.assertEqual([e.fractional_measure_onset for e in events], [0])
