"""Stem lengths predicted from each chord's context by a graph network trained
on MUSCIMA++ writers (noteworthy's train/stem_positions.py, after Shatri et
al., arXiv 2610.28899), run here in numpy.

Each chord of a staff is a node holding what is known before stems are drawn:
its noteheads' staff positions, stem direction, beams or flags, its beam group
and its gap to the previous chord. Edges join neighbours along the staff and
the chords of a beam group. The network predicts the length from the notehead
nearest the stem's tip to the tip. Handwriting strays from that as much as the
network's errors on writers it did not see, in parts drawn separately: one
writer's habit for the whole system, then each lone stem's own, or each beam
group's and each of its stems' within it."""
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from smashcima.scene.AffineSpace import AffineSpace
from smashcima.scene.semantic.BeamedGroup import BeamedGroup
from smashcima.scene.semantic.Chord import Chord
from smashcima.scene.semantic.StemValue import StemValue
from smashcima.scene.semantic.TypeDuration import TypeDuration
from smashcima.scene.visual.Notehead import Notehead

WEIGHTS = Path(__file__).parent / "stem_gnn.npz"

FEATURES = ["top_position", "bottom_position", "noteheads", "stem_up", "filled",
            "beams", "flags", "beamed", "group_size", "group_index", "gap_mm"]
SEQUENCE, BEAM_GROUP = 0, 1
EDGE_TYPES = 2
SHORTEST_STEM_MM = 3.0

LINES_OF_DURATION = {
    TypeDuration.eighth: 1, TypeDuration.sixteenth: 2,
    TypeDuration.thirty_second: 3, TypeDuration.sixty_fourth: 4,
}
"Beams, or flags, a note of each duration carries"


def layer_norm(h: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = h.mean(-1, keepdims=True)
    var = h.var(-1, keepdims=True)
    return (h - mean) / np.sqrt(var + 1e-5) * weight + bias


class StemGNN:
    """Numpy forward pass of train/stem_positions.StemGNN"""

    def __init__(self, path: Path = WEIGHTS):
        data = np.load(path)
        assert list(data["features"]) == FEATURES, \
            "The stem model was trained on other features"
        self.w = {k: data[k] for k in data.files}
        self.feature_scale = data["feature_scale"]
        self.writer_std = float(data["length_writer_std"])
        self.single_std = float(data["length_single_std"])
        self.group_std = float(data["length_group_std"])
        self.inside_group_std = float(data["length_inside_group_std"])
        self.layers = len([k for k in self.w if k.startswith("norms.") and k.endswith(".weight")])

    def linear(self, name: str, h: np.ndarray) -> np.ndarray:
        out = h @ self.w[f"{name}.weight"].T
        bias = self.w.get(f"{name}.bias")
        return out if bias is None else out + bias

    def predict(self, features: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """(length_mm, dx_mm) per chord"""
        h = np.maximum(self.linear("embed", features / self.feature_scale), 0)
        for layer in range(self.layers):
            update = self.linear(f"self_weights.{layer}", h)
            for t in range(EDGE_TYPES):
                e = edges[edges[:, 2] == t] if len(edges) else edges
                total = np.zeros_like(h)
                count = np.zeros((len(h), 1))
                if len(e):
                    np.add.at(total, e[:, 1], h[e[:, 0]])
                    np.add.at(count, e[:, 1], 1)
                update = update + self.linear(
                    f"neighbour_weights.{layer}.{t}", total / np.maximum(count, 1)
                )
            h = layer_norm(
                h + np.maximum(update, 0),
                self.w[f"norms.{layer}.weight"], self.w[f"norms.{layer}.bias"]
            )
        hidden = np.maximum(self.linear("head.0", h), 0)
        return self.linear("head.2", hidden)


def beam_group_of(chord: Chord) -> Optional[BeamedGroup]:
    group = BeamedGroup.of_chord_or_none(chord)
    return group if group is not None and len(group.chords) > 1 else None


def chord_x(chord: Chord, page_space: AffineSpace) -> float:
    return float(np.mean([
        Notehead.of_note(n).glyph.get_bbox_in_space(page_space).center.x
        for n in chord.notes
    ]))


def chord_features(chord: Chord) -> List[float]:
    """FEATURES but the gap, which needs the staff (staff_graph)"""
    positions = [Notehead.of_note(n).pitch_position for n in chord.notes]
    duration = chord.get_type_duration()
    lines = LINES_OF_DURATION.get(duration, 0)
    group = beam_group_of(chord)
    filled = duration not in (TypeDuration.half, TypeDuration.whole)
    if group is not None:
        group_size = len(group.chords)
        group_index = group.chords.index(chord) / (group_size - 1)
    else:
        group_size, group_index = 1, 0.0
    return [
        max(positions), min(positions), len(positions),
        float(chord.stem_value == StemValue.up), float(filled),
        lines if group is not None else 0, 0 if group is not None else lines,
        float(group is not None), group_size, group_index,
    ]


def staff_graph(
    chords: List[Chord],
    page_space: AffineSpace
) -> Tuple[List[Chord], np.ndarray, np.ndarray]:
    """The chords of one staff left to right, their features and edges"""
    ordered = sorted(chords, key=lambda c: chord_x(c, page_space))
    xs = np.array([chord_x(c, page_space) for c in ordered])
    gaps = np.diff(xs, prepend=xs[:1])
    features = np.array(
        [chord_features(c) + [gap] for c, gap in zip(ordered, gaps)],
        dtype=np.float32
    )
    edges = []
    for i in range(len(ordered) - 1):
        edges += [(i, i + 1, SEQUENCE), (i + 1, i, SEQUENCE)]
    groups: Dict[int, List[int]] = {}
    for i, chord in enumerate(ordered):
        group = beam_group_of(chord)
        if group is not None:
            groups.setdefault(id(group), []).append(i)
    for members in groups.values():
        edges += [(i, j, BEAM_GROUP) for i in members for j in members if i != j]
    return ordered, features, np.array(edges, dtype=np.int64).reshape(-1, 3)


class StemLengths:
    """Sampled stem lengths for the chords of a system, by chord"""

    def __init__(self, model: StemGNN, rng: random.Random):
        self.model = model
        self.rng = rng
        self.lengths: Dict[int, float] = {}
        self.writer_offset = rng.gauss(0, model.writer_std)
        self.group_offsets: Dict[int, float] = {}

    def offset_of(self, chord: Chord) -> float:
        group = beam_group_of(chord)
        if group is None:
            return self.writer_offset + self.rng.gauss(0, self.model.single_std)
        if id(group) not in self.group_offsets:
            self.group_offsets[id(group)] = self.rng.gauss(0, self.model.group_std)
        return self.writer_offset + self.group_offsets[id(group)] \
            + self.rng.gauss(0, self.model.inside_group_std)

    def predict_staff(self, chords: List[Chord], page_space: AffineSpace):
        stemmed = [c for c in chords
                   if c.stem_value in (StemValue.up, StemValue.down)
                   and len(c.notes) > 0]
        if len(stemmed) == 0:
            return
        ordered, features, edges = staff_graph(stemmed, page_space)
        predicted = self.model.predict(features, edges)
        for chord, (length, _) in zip(ordered, predicted):
            sampled = length + self.offset_of(chord)
            self.lengths[id(chord)] = max(SHORTEST_STEM_MM, float(sampled))

    def length_of(self, chord: Chord) -> Optional[float]:
        return self.lengths.get(id(chord))
