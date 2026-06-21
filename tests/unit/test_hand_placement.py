"""Tests for hand ground-alignment helpers in dexim.brainco.viz."""

import numpy as np
import pinocchio as pin

from dexim.brainco.viz import _ground_aligned_hand_base_position


class _EmptyGeometry:
    geometryObjects: list[object] = []


class _EmptyGeometryData:
    oMg: list[object] = []


class _FakeHandModel:
    def __init__(self) -> None:
        self.model = pin.Model()
        self.geometry_model = _EmptyGeometry()
        self.geometry_data = _EmptyGeometryData()
        self.last_q = None

    def compute_forward_kinematics(self, q):
        self.last_q = q


def test_ground_aligned_hand_base_position_offsets_left_hand() -> None:
    position = _ground_aligned_hand_base_position(_FakeHandModel(), "left")

    np.testing.assert_allclose(position, np.array([0.0, 0.35, 0.0]))


def test_ground_aligned_hand_base_position_offsets_right_hand() -> None:
    position = _ground_aligned_hand_base_position(_FakeHandModel(), "right")

    np.testing.assert_allclose(position, np.array([0.0, -0.35, 0.0]))
