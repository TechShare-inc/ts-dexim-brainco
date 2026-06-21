"""Retargeter -- finger-vector -> joint-angle retargeting for BrainCo Revo2.

Unlike the Inspire hand which uses NLopt VectorOptimizer, BrainCo Revo2 uses
a direct linear mapping from finger-direction vectors to joint angles.

Mapping (5 finger vectors → 6 joint angles):
    - thumb vector  → thumb_flex (flexion component) + thumb_aux (lateral component)
    - index vector  → index joint (flexion)
    - middle vector → middle joint (flexion)
    - ring vector   → ring joint (flexion)
    - pinky vector  → pinky joint (flexion)

Each finger vector's magnitude (projected onto a configurable flexion axis) is
linearly interpolated between the joint's lower and upper limits.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from dexim.brainco.interface._conversion import JOINT_LIMITS, NUM_JOINTS

# Default finger-to-joint mapping
# For each finger (thumb, index, middle, ring, pinky), list the joint indices
# that the finger vector maps to.  Thumb maps to 2 joints; others to 1.
_DEFAULT_JOINT_MAP: list[list[int]] = [
    [0, 1],  # thumb  -> thumb_flex(0) + thumb_aux(1)
    [2],  # index  -> index(2)
    [3],  # middle -> middle(3)
    [4],  # ring   -> ring(4)
    [5],  # pinky  -> pinky(5)
]

# Per-finger axis weights for mapping a 3D vector to flexion magnitude.
# Each row weights the [x, y, z] components of the finger vector.
# Default: use the Y component (vertical flexion) primarily.
_DEFAULT_FLEXION_AXIS: list[list[float]] = [
    [0.0, 1.0, 0.0],  # thumb
    [0.0, 1.0, 0.0],  # index
    [0.0, 1.0, 0.0],  # middle
    [0.0, 1.0, 0.0],  # ring
    [0.0, 1.0, 0.0],  # pinky
]

# Per-finger lateral axis for thumb abduction
_DEFAULT_THUMB_LATERAL_AXIS: list[float] = [1.0, 0.0, 0.0]


class Retargeter:
    """Maps finger-direction vectors to BrainCo Revo2 joint angles.

    Uses a direct linear interpolation approach: each finger vector's
    projected magnitude is linearly mapped between the joint's lower
    (open) and upper (closed) limits.

    Args:
        alpha: Per-finger scaling factors applied before retargeting.
            Length must equal the number of fingers (5).
        joint_map: Per-finger list of target joint indices.
            Default maps thumb→[0,1], index→[2], middle→[3], ring→[4], pinky→[5].
        flexion_axis: Per-finger axis weights for computing flexion magnitude
            from the 3D finger vector.  Each entry is a 3-element list of
            [x_weight, y_weight, z_weight].
        thumb_lateral_axis: Axis weights for thumb abduction component
            (used for joint index 1 = thumb_aux).
        max_vector_norm: Expected maximum finger-vector norm — used to
            normalise the mapping.  A vector of this length maps to the
            upper joint limit.
    """

    def __init__(
        self,
        alpha: list[float] | None = None,
        joint_map: list[list[int]] | None = None,
        flexion_axis: list[list[float]] | None = None,
        thumb_lateral_axis: list[float] | None = None,
        max_vector_norm: float = 0.15,
    ) -> None:
        self._alpha = (
            np.array(alpha, dtype=np.float64)
            if alpha is not None
            else np.ones(5, dtype=np.float64)
        )
        if len(self._alpha) != 5:
            raise ValueError(f"alpha must have length 5, got {len(self._alpha)}")

        self._joint_map = joint_map if joint_map is not None else _DEFAULT_JOINT_MAP

        self._flexion_axis = (
            np.array(flexion_axis, dtype=np.float64)
            if flexion_axis is not None
            else np.array(_DEFAULT_FLEXION_AXIS, dtype=np.float64)
        )

        self._thumb_lateral_axis = np.array(
            (
                thumb_lateral_axis
                if thumb_lateral_axis is not None
                else _DEFAULT_THUMB_LATERAL_AXIS
            ),
            dtype=np.float64,
        )

        self._max_vector_norm = max_vector_norm

        # Pre-compute joint limit arrays
        self._joint_lower = np.array([lo for lo, _ in JOINT_LIMITS], dtype=np.float64)
        self._joint_upper = np.array([hi for _, hi in JOINT_LIMITS], dtype=np.float64)

        logger.info(
            f"Retargeter: alpha={self._alpha.tolist()}, "
            f"max_vector_norm={max_vector_norm}"
        )

    @property
    def alpha(self) -> list[float]:
        """Per-finger scaling factors."""
        return self._alpha.tolist()

    @alpha.setter
    def alpha(self, value: list[float]) -> None:
        if len(value) != 5:
            raise ValueError(f"alpha must have length 5, got {len(value)}")
        self._alpha = np.array(value, dtype=np.float64)

    def scale(self, features: np.ndarray) -> np.ndarray:
        """Multiply feature vectors by per-finger alpha scaling.

        Args:
            features: Raw feature vectors of shape (num_fingers, 3).

        Returns:
            Scaled feature vectors.
        """
        return features * self._alpha[:, np.newaxis]

    def retarget(self, features: np.ndarray) -> np.ndarray | None:
        """Map scaled finger vectors to joint angles.

        For each finger:
        1. Project the 3D vector onto the flexion axis to get a scalar.
        2. Normalise by ``max_vector_norm`` and clamp to [0, 1].
        3. Linearly interpolate between the joint's lower and upper limits.
        4. Thumb additionally projects onto the lateral axis for thumb_aux.

        Args:
            features: Scaled feature vectors of shape (num_fingers, 3).

        Returns:
            Joint angles in radians of shape (NUM_JOINTS,), or None on error.
        """
        if features.shape[0] < 5:
            logger.error(f"Expected at least 5 finger vectors, got {features.shape[0]}")
            return None

        q = np.zeros(NUM_JOINTS, dtype=np.float64)

        for finger_idx in range(5):
            vec = features[finger_idx]  # (3,)
            target_joints = self._joint_map[finger_idx]

            # Flexion: dot product with flexion axis, normalised
            flexion_axis = self._flexion_axis[finger_idx]
            flexion_scalar = np.dot(vec, flexion_axis)
            t = np.clip(flexion_scalar / self._max_vector_norm, 0.0, 1.0)

            # Map to each target joint for this finger
            for joint_idx in target_joints:
                if joint_idx == 1:  # thumb_aux: use lateral component
                    lateral_scalar = np.dot(vec, self._thumb_lateral_axis)
                    t_joint = np.clip(lateral_scalar / self._max_vector_norm, 0.0, 1.0)
                else:
                    t_joint = t

                lo = self._joint_lower[joint_idx]
                hi = self._joint_upper[joint_idx]
                q[joint_idx] = lo + t_joint * (hi - lo)

        return q
