"""Retargeter -- VectorOptimizer-based finger-vector -> joint-angle retargeting."""

from __future__ import annotations

import numpy as np
from dexim.core.nodes.protocols import VectorOptimizerProtocol
from loguru import logger


class Retargeter:
    """Wraps a VectorOptimizer for optimisation-based retargeting.

    Args:
        optimizer: A VectorOptimizer (or compatible) instance.
        alpha: Per-finger scaling factors (length 5).  Applied by
            :meth:`scale` before retargeting.
    """

    def __init__(
        self,
        optimizer: VectorOptimizerProtocol,
        alpha: list[float] | None = None,
    ) -> None:
        self._optimizer = optimizer
        if alpha is not None:
            try:
                self._optimizer.alpha = alpha
            except Exception:
                logger.warning(
                    f"Failed to set optimizer alpha={alpha}, "
                    f"falling back to [1.0]*5"
                )
                self._optimizer.alpha = [1.0] * 5
        logger.info(f"Optimizer alpha: {self._optimizer.alpha}")

    def retarget(self, features: np.ndarray) -> np.ndarray | None:
        """Run VectorOptimizer retargeting on feature vectors.

        Args:
            features: Feature vectors of shape (num_fingers, 3).

        Returns:
            Optimal joint angles, or None if optimisation failed.
        """
        q_optimal, nlopt_result = self._optimizer.retarget(features)

        if nlopt_result.value < 1:
            logger.warning(f"Optimizer failed to converge: {nlopt_result.name}")
            return None

        return q_optimal

    def scale(self, features: np.ndarray) -> np.ndarray:
        """Multiply feature vectors by per-finger alpha scaling.

        Args:
            features: Raw feature vectors of shape (num_fingers, 3).

        Returns:
            Scaled feature vectors.
        """
        alpha = np.array(self._optimizer.alpha)
        return features * alpha[:, np.newaxis]
