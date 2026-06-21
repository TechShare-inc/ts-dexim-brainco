"""Shared conversion utilities for the BrainCo Revo2 hand interface.

Provides joint enumeration, limits, index mappings, and pure conversion
functions shared by all BrainCo Revo2 interface implementations
(Modbus RS-485, mock).

Unit convention:
    - The internal (URDF / RobotInterface) layer always uses *radians*.
    - The BrainCo SDK uses a normalized integer range of 0-1000:
        SDK    0  <->  open hand  <->  lower joint limit (q_min)
        SDK 1000  <->  closed hand <->  upper joint limit (q_max)

Revo2 active joint order (SDK / device-native):
    [thumb_flex, thumb_aux, index, middle, ring, pinky]

BrainCo SDK documentation:
    https://www.brainco-hz.com/docs/revolimb-hand/revo2/parameters.html
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Joint enumeration
# ---------------------------------------------------------------------------


class InternalJoint(IntEnum):
    """Internal joint indices for BrainCo Revo2 (6 active DOF).

    Order matches the BrainCo SDK device-native order:
    [thumb_flex, thumb_aux, index, middle, ring, pinky]
    """

    THUMB_FLEX = 0  # thumb proximal flexion
    THUMB_AUX = 1  # thumb metacarpal (aux/abd)
    INDEX = 2  # index proximal
    MIDDLE = 3  # middle proximal
    RING = 4  # ring proximal
    PINKY = 5  # pinky proximal


class ApiJoint(IntEnum):
    """BrainCo SDK API joint indices (device-native order).

    Matches InternalJoint 1:1 -- BrainCo SDK and internal order are the same.
    Kept as a separate enum for clarity and future mapping changes.
    """

    THUMB_FLEX = 0
    THUMB_AUX = 1
    INDEX = 2
    MIDDLE = 3
    RING = 4
    PINKY = 5


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_JOINTS: int = 6
API_VALUE_MIN: int = 0
API_VALUE_MAX: int = 1000

# Joint limits in internal order (radians).
# Each entry is (lower_limit, upper_limit).
# SDK 0 corresponds to the lower limit (open hand).
# SDK 1000 corresponds to the upper limit (closed hand).
#
# Source: BrainCo Revo2 documentation
#   Thumb flex: 0 to 59 deg
#   Thumb aux:  0 to 90 deg
#   Index:      0 to 81 deg
#   Middle:     0 to 81 deg
#   Ring:       0 to 81 deg
#   Pinky:      0 to 81 deg
JOINT_LIMITS: list[tuple[float, float]] = [
    (0.0, np.deg2rad(59.0)),  # InternalJoint.THUMB_FLEX
    (0.0, np.deg2rad(90.0)),  # InternalJoint.THUMB_AUX
    (0.0, np.deg2rad(81.0)),  # InternalJoint.INDEX
    (0.0, np.deg2rad(81.0)),  # InternalJoint.MIDDLE
    (0.0, np.deg2rad(81.0)),  # InternalJoint.RING
    (0.0, np.deg2rad(81.0)),  # InternalJoint.PINKY
]

# Internal order and API order are currently 1:1 for BrainCo Revo2.
# Maps are kept for explicit documentation and future-proofing.
INTERNAL_TO_API_MAP: list[int] = [
    int(ApiJoint.THUMB_FLEX),  # InternalJoint.THUMB_FLEX(0) -> ApiJoint.THUMB_FLEX(0)
    int(ApiJoint.THUMB_AUX),  # InternalJoint.THUMB_AUX(1)  -> ApiJoint.THUMB_AUX(1)
    int(ApiJoint.INDEX),  # InternalJoint.INDEX(2)      -> ApiJoint.INDEX(2)
    int(ApiJoint.MIDDLE),  # InternalJoint.MIDDLE(3)     -> ApiJoint.MIDDLE(3)
    int(ApiJoint.RING),  # InternalJoint.RING(4)       -> ApiJoint.RING(4)
    int(ApiJoint.PINKY),  # InternalJoint.PINKY(5)      -> ApiJoint.PINKY(5)
]

# Maps API joint index -> internal joint index (inverse of INTERNAL_TO_API_MAP).
API_TO_INTERNAL_MAP: list[int] = [
    int(
        InternalJoint.THUMB_FLEX
    ),  # ApiJoint.THUMB_FLEX(0) -> InternalJoint.THUMB_FLEX(0)
    int(
        InternalJoint.THUMB_AUX
    ),  # ApiJoint.THUMB_AUX(1)  -> InternalJoint.THUMB_AUX(1)
    int(InternalJoint.INDEX),  # ApiJoint.INDEX(2)       -> InternalJoint.INDEX(2)
    int(InternalJoint.MIDDLE),  # ApiJoint.MIDDLE(3)      -> InternalJoint.MIDDLE(3)
    int(InternalJoint.RING),  # ApiJoint.RING(4)        -> InternalJoint.RING(4)
    int(InternalJoint.PINKY),  # ApiJoint.PINKY(5)       -> InternalJoint.PINKY(5)
]

# URDF joint name pattern (used for model integration).
# ROS2 canonical naming: {left|right}_{thumb_proximal|thumb_metacarpal|index_proximal|...}
_INTERNAL_TO_URDF_SUFFIX: list[str] = [
    "thumb_proximal",  # THUMB_FLEX
    "thumb_metacarpal",  # THUMB_AUX
    "index_proximal",
    "middle_proximal",
    "ring_proximal",
    "pinky_proximal",
]

# Indices of the 6 actively-controlled joints within the full URDF model
# (0-based, in *internal* order).
#
# The BrainCo Revo2 URDF defines 11 revolute joints in this order:
#   thumb_metacarpal(0), thumb_proximal(1), thumb_distal(2),
#   index_proximal(3),   index_distal(4),
#   middle_proximal(5),  middle_distal(6),
#   ring_proximal(7),    ring_distal(8),
#   pinky_proximal(9),   pinky_distal(10)
#
# Only the proximal (and thumb metacarpal) joints are actively commanded;
# the distal joints are mechanically coupled.  The mapping below selects
# the 6 active URDF joints in the internal (SDK) order:
#   THUMB_FLEX  -> thumb_proximal    (URDF idx 1)
#   THUMB_AUX   -> thumb_metacarpal  (URDF idx 0)
#   INDEX       -> index_proximal    (URDF idx 3)
#   MIDDLE      -> middle_proximal   (URDF idx 5)
#   RING        -> ring_proximal     (URDF idx 7)
#   PINKY       -> pinky_proximal    (URDF idx 9)
URDF_ACTIVE_JOINT_INDICES: list[int] = [1, 0, 3, 5, 7, 9]


def urdf_joint_names(hand_side: str = "left") -> list[str]:
    """Build URDF-prefixed joint names for the given hand side.

    Args:
        hand_side: ``"left"`` or ``"right"``.

    Returns:
        List of 6 URDF joint names, e.g. ``["L_thumb_proximal", ...]``.
    """
    prefix = "L_" if hand_side.lower() == "left" else "R_"
    return [f"{prefix}{suffix}" for suffix in _INTERNAL_TO_URDF_SUFFIX]


# ---------------------------------------------------------------------------
# Pure conversion functions
# ---------------------------------------------------------------------------


def radians_to_api(q_rad: npt.ArrayLike) -> npt.NDArray[np.int32]:
    """Convert internal-order joint radians to API-order 0-1000 integer positions.

    Clips each joint to its limits, then applies the direct scale
    (API 0 = open/lower-limit, API 1000 = closed/upper-limit) and reorders
    to the device-native API layout.

    Args:
        q_rad: Joint positions in internal order, shape (6,), in radians.

    Returns:
        Integer positions in API order, shape (6,), values in [0, 1000].
    """
    q = np.asarray(q_rad, dtype=float)
    lower = np.array([lo for lo, _ in JOINT_LIMITS])
    upper = np.array([hi for _, hi in JOINT_LIMITS])
    clipped = np.clip(q, lower, upper)
    denom = upper - lower
    ratio = np.where(denom > 0.0, (clipped - lower) / denom, 0.0)
    api_float = np.clip(ratio * API_VALUE_MAX, API_VALUE_MIN, API_VALUE_MAX)

    api_ordered = np.empty(NUM_JOINTS, dtype=np.int32)
    for internal_idx, api_idx in enumerate(INTERNAL_TO_API_MAP):
        api_ordered[api_idx] = int(api_float[internal_idx])

    return api_ordered


def api_to_radians(api_values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert API-order 0-1000 positions to internal-order joint radians.

    Clips to the API range, applies the direct scale (API 0 = lower-limit)
    and reorders to internal layout.

    Args:
        api_values: Joint positions in API order, shape (6,), values 0-1000.

    Returns:
        Joint positions in internal radians order, shape (6,).
    """
    api = np.clip(np.asarray(api_values, dtype=float), API_VALUE_MIN, API_VALUE_MAX)

    positions_rad = np.empty(NUM_JOINTS, dtype=float)
    for internal_idx, api_idx in enumerate(INTERNAL_TO_API_MAP):
        q_min, q_max = JOINT_LIMITS[internal_idx]
        normalized = api[api_idx] / API_VALUE_MAX
        positions_rad[internal_idx] = normalized * (q_max - q_min) + q_min

    return positions_rad


__all__ = [
    "InternalJoint",
    "ApiJoint",
    "NUM_JOINTS",
    "API_VALUE_MIN",
    "API_VALUE_MAX",
    "JOINT_LIMITS",
    "INTERNAL_TO_API_MAP",
    "API_TO_INTERNAL_MAP",
    "URDF_ACTIVE_JOINT_INDICES",
    "urdf_joint_names",
    "radians_to_api",
    "api_to_radians",
]
