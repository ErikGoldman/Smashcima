import random
import unittest
from types import SimpleNamespace

from smashcima.loading import load_score
from smashcima.scene.semantic import Note
from smashcima.scene.semantic.Chord import Chord
from smashcima.synthesis.notation.StemModel import StemLengths


def chords_of_beamed_pair_and_quarter():
    eighth = lambda step, beam: (f"<note><pitch><step>{step}</step><octave>4</octave></pitch>"
                                 f"<duration>1</duration><type>eighth</type><stem>up</stem>"
                                 f"<beam number='1'>{beam}</beam></note>")
    xml = f"""<?xml version="1.0"?>
<score-partwise><part-list><score-part id="P1"><part-name/></score-part></part-list>
<part id="P1"><measure number="1">
<attributes><divisions>2</divisions><key><fifths>0</fifths></key><time><beats>2</beats><beat-type>4</beat-type></time>
<clef><sign>G</sign><line>2</line></clef></attributes>
{eighth("C", "begin")}{eighth("D", "end")}
<note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type><stem>up</stem></note>
</measure></part></score-partwise>"""
    s = load_score(data=xml, format=".musicxml")
    return [Chord.of_note(d) for e in s.parts[0].measures[0].events
            for d in e.durables if isinstance(d, Note)]


def spread(writer=0.0, single=0.0, group=0.0, inside_group=0.0):
    return SimpleNamespace(writer_std=writer, single_std=single, group_std=group,
                           inside_group_std=inside_group)


class StemSpreadTest(unittest.TestCase):
    def test_the_stems_of_a_beam_group_share_its_offset(self):
        first, second, _ = chords_of_beamed_pair_and_quarter()
        lengths = StemLengths(spread(group=1.0), random.Random(0))
        self.assertEqual(lengths.offset_of(first), lengths.offset_of(second))
        self.assertNotEqual(lengths.offset_of(first), 0.0)

    def test_a_lone_stem_draws_its_own_offset(self):
        _, _, quarter = chords_of_beamed_pair_and_quarter()
        lengths = StemLengths(spread(group=1.0), random.Random(0))
        self.assertEqual(lengths.offset_of(quarter), 0.0)

    def test_one_writer_offset_moves_every_stem(self):
        first, _, quarter = chords_of_beamed_pair_and_quarter()
        lengths = StemLengths(spread(writer=1.0), random.Random(0))
        self.assertEqual(lengths.offset_of(first), lengths.offset_of(quarter))
        self.assertNotEqual(lengths.writer_offset, 0.0)
