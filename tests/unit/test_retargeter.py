"""Unit tests for dexim.brainco.node.retargeter.Retargeter."""

from __future__ import annotations

import numpy as np
import pytest

from dexim.brainco.interface._conversion import JOINT_LIMITS
from dexim.brainco.node.retargeter import Retargeter

# Pre-compute joint limit arrays
_JOINT_LOWER = np.array([lo for lo, _ in JOINT_LIMITS], dtype=np.float64)
_JOINT_UPPER = np.array([hi for _, hi in JOINT_LIMITS], dtype=np.float64)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_alpha_is_ones(self):
        r = Retargeter()
        assert r.alpha == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_custom_alpha(self):
        r = Retargeter(alpha=[0.5, 0.6, 0.7, 0.8, 0.9])
        assert r.alpha == [0.5, 0.6, 0.7, 0.8, 0.9]

    def test_alpha_wrong_length_raises(self):
        with pytest.raises(ValueError, match="alpha must have length 5"):
            Retargeter(alpha=[1.0, 2.0, 3.0])

    def test_setter_wrong_length_raises(self):
        r = Retargeter()
        with pytest.raises(ValueError, match="alpha must have length 5"):
            r.alpha = [1.0, 2.0]

    def test_setter_updates(self):
        r = Retargeter()
        r.alpha = [0.2, 0.3, 0.4, 0.5, 0.6]
        assert r.alpha == [0.2, 0.3, 0.4, 0.5, 0.6]

    def test_custom_max_vector_norm(self):
        r = Retargeter(max_vector_norm=0.3)
        assert r._max_vector_norm == 0.3


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


class TestScale:
    def test_identity_scale(self):
        r = Retargeter()
        features = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        scaled = r.scale(features)
        np.testing.assert_allclose(scaled, features, atol=1e-10)

    def test_half_scale(self):
        r = Retargeter(alpha=[0.5, 0.5, 0.5, 0.5, 0.5])
        features = np.ones((5, 3), dtype=np.float64)
        scaled = r.scale(features)
        np.testing.assert_allclose(scaled, np.full((5, 3), 0.5), atol=1e-10)

    def test_per_finger_scale(self):
        r = Retargeter(alpha=[1.0, 2.0, 3.0, 4.0, 5.0])
        features = np.ones((5, 3), dtype=np.float64)
        scaled = r.scale(features)
        expected = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(scaled, expected)


# ---------------------------------------------------------------------------
# Retarget
# ---------------------------------------------------------------------------


class TestRetarget:
    def test_zero_vectors_give_lower_limits(self):
        """Zero features = open hand (lower limits)."""
        r = Retargeter()
        features = np.zeros((5, 3), dtype=np.float64)
        q = r.retarget(features)
        assert q is not None
        np.testing.assert_allclose(q, _JOINT_LOWER, atol=1e-10)

    def test_max_flexion_vectors_give_upper_limits(self):
        """At max_vector_norm along Y axis, joints hit upper limits."""
        r = Retargeter(max_vector_norm=0.15)
        # Full Y-axis vectors at max norm
        features = np.array(
            [
                [0.0, 0.15, 0.0],  # thumb
                [0.0, 0.15, 0.0],  # index
                [0.0, 0.15, 0.0],  # middle
                [0.0, 0.15, 0.0],  # ring
                [0.0, 0.15, 0.0],  # pinky
            ],
            dtype=np.float64,
        )
        q = r.retarget(features)
        assert q is not None
        # Index, middle, ring, pinky should be at upper limits
        np.testing.assert_allclose(q[2], _JOINT_UPPER[2], atol=1e-10)
        np.testing.assert_allclose(q[3], _JOINT_UPPER[3], atol=1e-10)
        np.testing.assert_allclose(q[4], _JOINT_UPPER[4], atol=1e-10)
        np.testing.assert_allclose(q[5], _JOINT_UPPER[5], atol=1e-10)
        # Thumb flex should also be at upper limit
        np.testing.assert_allclose(q[0], _JOINT_UPPER[0], atol=1e-10)

    def test_half_flexion_gives_midpoint(self):
        """Half the max vector norm = halfway between limits."""
        r = Retargeter(max_vector_norm=0.15)
        features = np.array(
            [
                [0.0, 0.075, 0.0],
                [0.0, 0.075, 0.0],
                [0.0, 0.075, 0.0],
                [0.0, 0.075, 0.0],
                [0.0, 0.075, 0.0],
            ],
            dtype=np.float64,
        )
        q = r.retarget(features)
        assert q is not None
        expected_mid = (_JOINT_LOWER + _JOINT_UPPER) / 2.0
        np.testing.assert_allclose(q[2], expected_mid[2], atol=1e-8)
        np.testing.assert_allclose(q[3], expected_mid[3], atol=1e-8)
        np.testing.assert_allclose(q[4], expected_mid[4], atol=1e-8)
        np.testing.assert_allclose(q[5], expected_mid[5], atol=1e-8)

    def test_clamped_above_max(self):
        """Features beyond max_vector_norm are clamped to upper limits."""
        r = Retargeter(max_vector_norm=0.15)
        features = np.full((5, 3), 10.0, dtype=np.float64)  # way above max
        q = r.retarget(features)
        assert q is not None
        np.testing.assert_allclose(q, _JOINT_UPPER, atol=1e-8)

    def test_negative_flexion_clamps_to_lower(self):
        """Negative Y (opposite direction) clamps to lower limits."""
        r = Retargeter(max_vector_norm=0.15)
        features = np.full((5, 3), -10.0, dtype=np.float64)
        q = r.retarget(features)
        assert q is not None
        np.testing.assert_allclose(q, _JOINT_LOWER, atol=1e-10)

    def test_thumb_lateral_maps_to_thumb_aux(self):
        """X-axis thumb vector maps to thumb_aux joint."""
        r = Retargeter(max_vector_norm=0.15)
        features = np.zeros((5, 3), dtype=np.float64)
        features[0] = [0.15, 0.0, 0.0]  # thumb: full X (lateral)
        q = r.retarget(features)
        assert q is not None
        # thumb_aux should be at upper limit
        np.testing.assert_allclose(q[1], _JOINT_UPPER[1], atol=1e-10)
        # thumb_flex should be at lower (no Y component)
        np.testing.assert_allclose(q[0], _JOINT_LOWER[0], atol=1e-10)

    def test_output_shape(self):
        r = Retargeter()
        q = r.retarget(np.ones((5, 3), dtype=np.float64))
        assert q is not None
        assert q.shape == (6,)

    def test_too_few_fingers_returns_none(self):
        r = Retargeter()
        q = r.retarget(np.ones((3, 3), dtype=np.float64))
        assert q is None
