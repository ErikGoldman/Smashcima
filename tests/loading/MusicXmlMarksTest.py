import unittest

from smashcima.loading import load_score
from smashcima.scene.semantic import BarStyle, Note, Slur, Tuplet


def score(notes: str, barline: str = "") -> str:
    return f"""<?xml version="1.0"?>
<score-partwise><part-list><score-part id="P1"><part-name/></score-part></part-list>
<part id="P1"><measure number="1">
<attributes><divisions>3</divisions><key><fifths>0</fifths></key>
<time><beats>2</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
{notes}{barline}</measure></part></score-partwise>"""


def note(step: str, notations: str = "", duration: int = 2, kind: str = "eighth",
         modification: str = "<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>") -> str:
    return f"""<note><pitch><step>{step}</step><octave>4</octave></pitch><duration>{duration}</duration>
<type>{kind}</type>{modification}<notations>{notations}</notations></note>"""


def notes_of(xml: str) -> list:
    s = load_score(data=xml, format=".musicxml")
    return [d for e in s.parts[0].measures[0].events for d in e.durables if isinstance(d, Note)]


class MusicXmlMarksTest(unittest.TestCase):
    def test_a_triplet_holds_its_three_notes_and_whether_its_number_shows(self):
        notes = notes_of(score(note("C", '<tuplet type="start" show-number="none"/>') + note("D")
                               + note("E", '<tuplet type="stop"/>') + note("F", "", 3, "quarter", "")))
        tuplet = Tuplet.of_durable_or_none(notes[0])
        self.assertEqual(tuplet.durables, notes[:3])
        self.assertEqual((tuplet.actual_notes, tuplet.number_shown), (3, False))
        self.assertIsNone(Tuplet.of_durable_or_none(notes[3]))

    def test_a_slur_joins_its_start_and_stop_notes(self):
        notes = notes_of(score(note("C", '<slur type="start" number="1"/>', 3, "quarter", "")
                               + note("D", '<slur type="stop" number="1"/>', 3, "quarter", "")))
        [slur] = Slur.starting_at(notes[0])
        self.assertIs(slur.end, notes[1])

    def test_a_slur_stopping_without_a_start_begins_before_the_music(self):
        notes = notes_of(score(note("C", '<slur type="stop"/>', 6, "half", "")))
        [slur] = Slur.ending_at(notes[0])
        self.assertIsNone(slur.start)

    def test_ties_and_articulations_are_read_from_notations(self):
        notes = notes_of(score(note("C", '<tied type="start"/><articulations><staccato/><fermata/></articulations>', 3, "quarter", "")
                               + note("C", '<tied type="stop"/>', 3, "quarter", "")))
        self.assertEqual((notes[0].tie_starts, notes[0].tie_stops, notes[0].articulations), (True, False, ["staccato"]))
        self.assertEqual((notes[1].tie_starts, notes[1].tie_stops), (False, True))

    def test_a_backward_repeat_and_its_bar_style_close_the_measure(self):
        xml = score(note("C", "", 6, "half", ""),
                    '<barline location="right"><bar-style>light-heavy</bar-style><repeat direction="backward"/></barline>')
        measure = load_score(data=xml, format=".musicxml").parts[0].measures[0]
        self.assertEqual((measure.right_bar_style, measure.repeat_end, measure.repeat_start),
                         (BarStyle.light_heavy, True, False))


if __name__ == "__main__":
    unittest.main()
