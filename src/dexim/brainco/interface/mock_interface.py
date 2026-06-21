"""Mock (loopback) interface for the BrainCo Revo2 hand.

This module provides an in-memory mock implementation of the RobotInterface
protocol. It stores commanded positions internally and returns them on read(),
acting as a loopback device for testing and simulation scenarios.

Behavior:
    - ``write()`` stores ``cmd.q`` internally, clipped to joint limits.
    - ``read()`` returns the last commanded positions (velocities/torques = 0).
    - ``connect()`` / ``disconnect()`` are no-ops (no hardware).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from dexim.core.robot_interface import JointCommand, JointState, RobotInterface
from loguru import logger

from dexim.brainco.interface._conversion import (
    JOINT_LIMITS,
    NUM_JOINTS,
    urdf_joint_names,
)

if TYPE_CHECKING:
    from dexim.brainco.model.model import BrainCoModel


class MockInterface(RobotInterface):
    """Loopback interface for the BrainCo Revo2 hand (no hardware required).

    Accepts an optional ``BrainCoModel`` to derive joint names and limits from
    the URDF.  When no model is supplied the class falls back to built-in
    defaults so it can be used in lightweight test setups.

    Args:
        brainco_model: Optional ``BrainCoModel`` for joint metadata.
        hand_side: ``"left"`` or ``"right"`` (used for joint-name prefixes
            when no model is provided).
    """

    def __init__(
        self,
        brainco_model: BrainCoModel | None = None,
        hand_side: str = "left",
    ) -> None:
        self._model = brainco_model
        self._hand_side = hand_side.lower()

        # Derive metadata from model when available.
        if brainco_model is not None:
            self._nq: int = brainco_model.nq
            if brainco_model.model is not None and brainco_model.model.names:
                self._joint_names: list[str] = [
                    brainco_model.model.names[i] for i in range(1, self._nq + 1)
                ]
            else:
                self._joint_names = self._default_joint_names()
            if brainco_model.model is not None:
                self._lower = np.array(
                    brainco_model.model.lowerPositionLimit[: self._nq]
                )
                self._upper = np.array(
                    brainco_model.model.upperPositionLimit[: self._nq]
                )
            else:
                self._lower, self._upper = self._default_limits()
        else:
            self._nq = NUM_JOINTS
            self._joint_names = self._default_joint_names()
            self._lower, self._upper = self._default_limits()

        # Internal state -- start at lower limits (open hand).
        self._q = self._lower.copy()
        self._connected = False
        self._estop = False
        self._start_time = 0.0

        logger.info("MockInterface initialized: %d joints", self._nq)

    # ------------------------------------------------------------------
    # RobotInterface protocol
    # ------------------------------------------------------------------

    def connect(self) -> None:  # noqa: D401
        """No-op connect (no hardware)."""
        if self._connected:
            logger.warning("MockInterface already connected")
            return
        self._connected = True
        self._start_time = time.time()
        logger.info("MockInterface connected (loopback)")

    def disconnect(self) -> None:  # noqa: D401
        """No-op disconnect."""
        if not self._connected:
            return
        self._connected = False
        logger.info("MockInterface disconnected")

    def is_connected(self) -> bool:
        """Return whether the interface is connected."""
        return self._connected

    def read(self) -> JointState:
        """Return the last commanded joint state."""
        if not self._connected:
            raise RuntimeError("MockInterface not connected")
        return JointState(
            q=self._q.copy(),
            qd=np.zeros(self._nq),
            tau=np.zeros(self._nq),
            stamp=time.time(),
        )

    def write(self, cmd: JointCommand) -> None:
        """Store commanded positions (clipped to joint limits)."""
        if not self._connected:
            raise RuntimeError("MockInterface not connected")
        if cmd.mode != "position":
            raise NotImplementedError("MockInterface only supports position mode")
        if cmd.q is None:
            raise ValueError("Position command (q) is required")

        q = np.asarray(cmd.q, dtype=float)
        if q.shape[0] != self._nq:
            raise ValueError(f"Expected {self._nq} positions, got {q.shape[0]}")

        self._q = np.clip(q, self._lower, self._upper)
        logger.debug("MockInterface: wrote q=%s", self._q)

    # ------------------------------------------------------------------
    # Metadata helpers (RobotInterface protocol)
    # ------------------------------------------------------------------

    def num_joint_configurations(self) -> int:
        return self._nq

    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    def num_actuated_configurations(self) -> int:
        return self._nq

    def actuated_joint_names(self) -> list[str]:
        return list(self._joint_names)

    def num_full_configurations(self) -> int:
        return self._nq

    def full_joint_names(self) -> list[str]:
        return list(self._joint_names)

    def time(self) -> float:
        if not self._connected:
            return 0.0
        return time.time() - self._start_time

    def estop(self) -> bool:
        return self._estop

    def set_estop(self, state: bool) -> None:
        self._estop = state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_joint_names(self) -> list[str]:
        """Return default URDF-prefixed joint names."""
        return urdf_joint_names(self._hand_side)

    @staticmethod
    def _default_limits() -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) limits from built-in constants."""
        lower = np.array([lo for lo, _ in JOINT_LIMITS])
        upper = np.array([hi for _, hi in JOINT_LIMITS])
        return lower, upper
