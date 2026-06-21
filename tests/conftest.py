"""Shared test fixtures for dexim-brainco test suites.

Provides:
- ``propagate_loguru``: autouse fixture that routes loguru messages into
  pytest's ``caplog`` so ``caplog.messages`` assertions work with loguru.
- ``mock_brainco_model``: a ``MagicMock`` standing in for ``BrainCoModel``
  (left hand, 6 DOF) with realistic attributes; available without pinocchio.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
from loguru import logger

# Joint limits in internal order (mirrors JOINT_LIMITS in interface/_conversion.py).
_JOINT_LIMITS_LOWER = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_JOINT_LIMITS_UPPER = np.array(
    [
        np.deg2rad(59.0),  # thumb flex
        np.deg2rad(90.0),  # thumb aux
        np.deg2rad(81.0),  # index
        np.deg2rad(81.0),  # middle
        np.deg2rad(81.0),  # ring
        np.deg2rad(81.0),  # pinky
    ]
)

# Pinocchio model.names convention: index 0 = "universe", then active joints.
_LEFT_PIN_NAMES = [
    "universe",
    "L_thumb_proximal",
    "L_thumb_metacarpal",
    "L_index_proximal",
    "L_middle_proximal",
    "L_ring_proximal",
    "L_pinky_proximal",
]


# ---------------------------------------------------------------------------
# Loguru -> caplog bridge
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def propagate_loguru(caplog):
    """Route loguru messages into pytest's caplog for assertion in tests.

    loguru does not use Python's standard ``logging`` module by default, so
    ``caplog.messages`` would otherwise be empty.  This fixture adds a loguru
    sink that forwards to the caplog handler for the duration of each test.
    """
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
    )
    caplog.set_level(logging.DEBUG)
    yield
    logger.remove(handler_id)


# ---------------------------------------------------------------------------
# Mock BrainCoModel
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_brainco_model():
    """A MagicMock standing in for ``BrainCoModel`` (left hand, 6 DOF).

    Attributes are shaped to satisfy ``MockInterface.__init__`` without
    requiring pinocchio to be installed.

    Returns:
        MagicMock with ``nq``, ``model`` (pinocchio shape), joint names,
        and joint limits.
    """
    pin_model = MagicMock()
    pin_model.nq = 6
    pin_model.nv = 6
    pin_model.names = _LEFT_PIN_NAMES
    pin_model.lowerPositionLimit = _JOINT_LIMITS_LOWER.copy()
    pin_model.upperPositionLimit = _JOINT_LIMITS_UPPER.copy()

    brainco_model = MagicMock()
    brainco_model.nq = 6
    brainco_model.nv = 6
    brainco_model.model = pin_model
    brainco_model.lower_joint_limits = _JOINT_LIMITS_LOWER.copy()
    brainco_model.upper_joint_limits = _JOINT_LIMITS_UPPER.copy()
    return brainco_model
