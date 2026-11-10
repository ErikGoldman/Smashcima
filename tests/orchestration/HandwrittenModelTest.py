import os
import unittest

import smashcima as sc


def unison_beside_a_second() -> str:
    """F5 in an up-stem voice, and E5 with F5 in a down-stem voice: the F5s share
    one notehead, a second above E5's, which would push it aside"""
    pitch = lambda step: f"<pitch><step>{step}</step><octave>5</octave></pitch>"
    note = lambda step, voice, stem, chord="": (
        f"<note>{chord}{pitch(step)}<duration>2</duration><voice>{voice}</voice>"
        f"<type>half</type><stem>{stem}</stem></note>")
    return f"""<?xml version="1.0"?>
<score-partwise><part-list><score-part id="P1"><part-name/></score-part></part-list>
<part id="P1"><measure number="1">
<attributes><divisions>1</divisions><key><fifths>0</fifths></key><time><beats>2</beats><beat-type>4</beat-type></time>
<clef><sign>G</sign><line>2</line></clef></attributes>
{note("F", 1, "up")}<backup><duration>2</duration></backup>
{note("E", 2, "down")}{note("F", 2, "down", "<chord/>")}
</measure></part></score-partwise>"""


@unittest.skipUnless(os.environ.get("MC_ASSETS_CACHE"), "needs the installed assets (MC_ASSETS_CACHE)")
class HandwrittenModelTest(unittest.TestCase):
    def test_a_unison_beside_a_second_keeps_a_side_for_each_stem(self):
        scene = sc.orchestration.BaseHandwrittenModel()(data=unison_beside_a_second(), format=".musicxml")
        self.assertEqual(len(scene.pages), 1)
