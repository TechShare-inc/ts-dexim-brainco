"""FeatureExtractor -- skeleton -> finger-vector feature extraction."""

from __future__ import annotations

import numpy as np
from dexim.core.messages import HandState
from loguru import logger

from dexim.brainco.node.config import FeatureExtractionConfig


class FeatureExtractor:
    """Extracts finger-direction vectors from a HandState message.

    This is a nearly pure transformation: given joint positions, compute
    per-finger direction vectors using configured src/dst index pairs.

    Args:
        config: Index pairs and rotation flag.
    """

    def __init__(self, config: FeatureExtractionConfig) -> None:
        self._config = config
        logger.info(
            f"Feature extraction: {config.src_indices} -> {config.dst_indices}, "
            f"rotation={'on' if config.apply_rotation else 'off'}"
        )

    def extract(self, state: HandState) -> np.ndarray | None:
        """Extract finger-direction vectors from a HandState.

        Args:
            state: HandState with populated joints list.

        Returns:
            Array of shape (num_fingers, 3), or None on failure.
        """
        src = self._config.src_indices
        dst = self._config.dst_indices

        if not src or not dst:
            logger.error("Feature config missing src_indices or dst_indices")
            return None

        joint_positions: dict[int, np.ndarray] = {
            j.id: np.array(j.position, dtype=np.float64) for j in state.joints
        }

        vectors: list[np.ndarray] = []
        for s, d in zip(src, dst):
            sp = joint_positions.get(s)
            dp = joint_positions.get(d)
            if sp is None or dp is None:
                logger.warning(f"Joint index not found: src={s}, dst={d}")
                return None
            vectors.append(dp - sp)

        if not vectors:
            return None

        result = np.array(vectors, dtype=np.float64)

        if self._config.apply_rotation:
            result = self._apply_root_rotation(state, result)

        return result

    @staticmethod
    def _apply_root_rotation(state: HandState, vectors: np.ndarray) -> np.ndarray:
        """Rotate vectors into hand-local space using root joint orientation."""
        root_joint = state.joint_by_id(0)
        if root_joint is None:
            return vectors
        try:
            from scipy.spatial.transform import Rotation

            w, x, y, z = root_joint.rotation
            rot = Rotation.from_quat([x, y, z, w])
            return rot.apply(vectors)
        except ImportError:
            logger.warning(
                "scipy not available; rotation requested by config "
                "but will be skipped — install scipy for full functionality"
            )
        except Exception as exc:
            logger.warning(f"Rotation application failed: {exc}")
        return vectors
