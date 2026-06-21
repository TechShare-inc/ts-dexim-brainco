"""Unit tests for dexim.brainco.node.extractor.FeatureExtractor."""

from __future__ import annotations

import numpy as np
import pytest
from dexim.core.messages import HandState, SkeletonJoint

from dexim.brainco.node.config import FeatureExtractionConfig
from dexim.brainco.node.extractor import FeatureExtractor


def _make_hand_state(
    num_joints: int = 25,
    with_root_rotation: bool = False,
) -> HandState:
    """Build a minimal HandState with position-only joints in a line."""
    joints = []
    for i in range(num_joints):
        x = float(i) * 0.03  # spread along Y
        joints.append(
            SkeletonJoint(
                id=i,
                position=(0.0, x, 0.0),
                rotation=(
                    (1.0, 0.0, 0.0, 0.0)
                    if with_root_rotation and i == 0
                    else (0.0, 0.0, 0.0, 1.0)
                ),
            )
        )
    return HandState(glove_id=0, side="left", timestamp=1.0, joints=joints)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_accepts_default_config(self):
        cfg = FeatureExtractionConfig()
        extractor = FeatureExtractor(cfg)
        assert extractor._config is cfg


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


class TestExtract:
    def test_returns_vectors(self):
        cfg = FeatureExtractionConfig(
            src_indices=[1, 6],
            dst_indices=[4, 9],
            apply_rotation=False,
        )
        extractor = FeatureExtractor(cfg)
        state = _make_hand_state()
        result = extractor.extract(state)

        assert result is not None
        assert result.shape == (2, 3)

    def test_vector_direction_correct(self):
        cfg = FeatureExtractionConfig(
            src_indices=[1],
            dst_indices=[4],
            apply_rotation=False,
        )
        extractor = FeatureExtractor(cfg)
        state = _make_hand_state()
        result = extractor.extract(state)

        # Joint 1 at (0, 0.03, 0), Joint 4 at (0, 0.12, 0)
        # Expected vector: (0, 0.09, 0)
        assert result is not None
        np.testing.assert_allclose(result[0], [0.0, 0.09, 0.0], atol=1e-10)

    def test_missing_joint_returns_none(self):
        cfg = FeatureExtractionConfig(
            src_indices=[1, 999],
            dst_indices=[4, 998],
            apply_rotation=False,
        )
        extractor = FeatureExtractor(cfg)
        state = _make_hand_state()
        result = extractor.extract(state)
        assert result is None

    def test_five_fingers(self):
        cfg = FeatureExtractionConfig(
            src_indices=[1, 6, 11, 16, 21],
            dst_indices=[4, 9, 14, 19, 24],
            apply_rotation=False,
        )
        extractor = FeatureExtractor(cfg)
        state = _make_hand_state()
        result = extractor.extract(state)
        assert result is not None
        assert result.shape == (5, 3)

    def test_apply_rotation_no_scipy(self):
        """When apply_rotation=True and scipy unavailable, returns vectors unrotated."""
        cfg = FeatureExtractionConfig(
            src_indices=[1, 6],
            dst_indices=[4, 9],
            apply_rotation=True,
        )
        extractor = FeatureExtractor(cfg)
        state = _make_hand_state(with_root_rotation=True)
        # Without scipy, _apply_root_rotation falls through
        result = extractor.extract(state)
        assert result is not None
        assert result.shape == (2, 3)
