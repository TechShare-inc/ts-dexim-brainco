"""Strict model-specific mock implementing core RobotInterface conventions."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from dexim.core.robot_interface import JointCommand, JointState

from .model import ModelDescriptor


class OfflineMockInterface:
    """Copied snapshots with explicit unavailable torque/velocity feedback."""

    def __init__(
        self, descriptor: ModelDescriptor, *, clock: Callable[[], float] = time.time
    ) -> None:
        self.descriptor = descriptor
        self.clock = clock
        self._connected = False
        self._q = np.clip(
            np.zeros(len(descriptor.joint_names)), descriptor.lower, descriptor.upper
        )

    def connect(self) -> None:
        """Connect only this in-memory mock."""
        self._connected = True

    def disconnect(self) -> None:
        """Revoke mock connectivity."""
        self._connected = False

    def is_connected(self) -> bool:
        """Return mock connection state."""
        return self._connected

    def write(self, cmd: JointCommand) -> None:
        """Reject unsupported modes, malformed commands and limit violations."""
        if not self._connected:
            raise RuntimeError("Mock disconnected")
        if cmd.mode != "position" or cmd.q is None or cmd.tau is not None:
            raise ValueError("Mock supports position commands only")
        q = self.descriptor.vector(cmd.q)
        if cmd.qd is not None:
            self.descriptor.vector(cmd.qd)
        if (q < self.descriptor.lower).any() or (q > self.descriptor.upper).any():
            raise ValueError("Command outside model bounds")
        self._q = q.copy()

    def read(self) -> JointState:
        """Read a fresh mock snapshot; NaN means unavailable, never measured zero."""
        if not self._connected:
            raise RuntimeError("Mock disconnected")
        unavailable = np.full(len(self._q), np.nan)
        return JointState(
            self._q.copy(),
            unavailable.copy(),
            unavailable.copy(),
            self.clock(),
            q_full=np.array(self.descriptor.expand(self._q)),
        )

    def time(self) -> float:
        """Return Unix host time."""
        return self.clock()

    def estop(self) -> bool:
        """Disconnect mock; no physical stop claim."""
        self.disconnect()
        return True

    def joint_names(self) -> list[str]:
        """Independent joint names."""
        return list(self.descriptor.joint_names)

    actuated_joint_names = joint_names

    def full_joint_names(self) -> list[str]:
        """Full joint names including derived mimic joints."""
        return list(self.descriptor.full_joint_names)

    def num_joint_configurations(self) -> int:
        """Independent dimension."""
        return len(self.descriptor.joint_names)

    num_actuated_configurations = num_joint_configurations

    def num_full_configurations(self) -> int:
        """Full dimension."""
        return len(self.descriptor.full_joint_names)
