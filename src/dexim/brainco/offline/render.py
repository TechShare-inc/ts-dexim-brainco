"""Explicit external-asset FK and observed-state rendering, imported on demand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .controller import ObservationRecord
from .model import ModelDescriptor


class NativeModel:
    """Load pinned external URDF for joint-frame FK; no mesh redistribution.

    This renders a joint skeleton. Mesh rendering additionally requires separately
    supplied licensed meshes; ordinary model/mock inspection uses no native code.
    """

    def __init__(self, descriptor: ModelDescriptor, urdf_path: str | Path) -> None:
        descriptor.validate_assets(urdf_path, require_meshes=False)
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise ImportError(
                "Native FK requires optional pinocchio installation"
            ) from exc
        self.pin = pin
        self.descriptor = descriptor
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        names = tuple(self.model.names)[1:]
        if set(names) != set(descriptor.full_joint_names):
            raise ValueError("Native model joint identity mismatch")

    def segments(self, positions: object) -> np.ndarray:
        """Return FK parent-child segments from independent observed radians."""
        expanded = dict(
            zip(self.descriptor.full_joint_names, self.descriptor.expand(positions))
        )
        q = np.zeros(self.model.nq)
        for index in range(1, self.model.njoints):
            joint = self.model.joints[index]
            q[joint.idx_q] = expanded[self.model.names[index]]
        self.pin.forwardKinematics(self.model, self.data, q)
        return np.array(
            [
                [
                    self.data.oMi[self.model.parents[i]].translation,
                    self.data.oMi[i].translation,
                ]
                for i in range(1, self.model.njoints)
            ]
        )


class ObservedRenderer:
    """Draw independent observations only, retaining their model/side identity."""

    def __init__(
        self,
        descriptor: ModelDescriptor,
        urdf_path: str | Path,
        server: Any,
        scene_root: str = "/brainco_observed",
    ) -> None:
        self.descriptor = descriptor
        self.kinematics = NativeModel(descriptor, urdf_path)
        self.server, self.scene_root = server, scene_root
        self._handle: Any = None

    def update(self, observation: ObservationRecord) -> None:
        """Render observed FK; refuse actions, mismatched identity and bounds."""
        if not isinstance(observation, ObservationRecord):
            raise ValueError(
                "Renderer requires ObservationRecord, never issued actions"
            )
        if (
            observation.model != self.descriptor.model
            or observation.side != self.descriptor.side
            or observation.joint_names != self.descriptor.joint_names
        ):
            raise ValueError("Observed model/side/named mapping mismatch")
        q = self.descriptor.vector(observation.positions)
        if (q < self.descriptor.lower).any() or (q > self.descriptor.upper).any():
            raise ValueError("Observed positions exceed model bounds")
        segments = self.kinematics.segments(q)
        colors = np.broadcast_to(np.array([50, 180, 230]), segments.shape).copy()
        if self._handle is not None:
            self._handle.remove()
        self._handle = self.server.scene.add_line_segments(
            self.scene_root, points=segments, colors=colors, line_width=3.0
        )

    def close(self) -> None:
        """Remove only owned scene handles."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
