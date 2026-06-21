"""BrainCo Revo2 kinematic model powered by Pinocchio.

URDF assets are vendored from BrainCoTech/revo2_description.
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from dexim.core.model import BaseHandModel

if TYPE_CHECKING:
    import pinocchio  # pragma: no cover

#: Absolute path to the vendored revo2_description package root.
_VENDOR_ROOT = (
    pathlib.Path(__file__).resolve().parents[4] / "vendors" / "revo2_description"
)


class BrainCoModel(BaseHandModel):
    """Pinocchio-based kinematic model for BrainCo Revo2 hand.

    Loads the URDF from the vendored ``revo2_description`` package.
    The model is built lazily on first access to any kinematic property.

    Parameters
    ----------
    hand_side :
        ``"left"`` (default) or ``"right"``.
    """

    #: Tip-link suffixes shared by both hands.
    _TIP_LINK_SUFFIXES = [
        "thumb_tip_link",
        "index_tip_link",
        "middle_tip_link",
        "ring_tip_link",
        "pinky_tip_link",
    ]

    def __init__(self, hand_side: str = "left") -> None:
        super().__init__()
        if hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")
        self.hand_side: str = hand_side

        self._model: pinocchio.Model | None = None
        self._data: pinocchio.Data | None = None
        self._geometry_model: pinocchio.GeometryModel | None = None
        self._geometry_data: pinocchio.GeometryData | None = None
        self._built = False

        # BaseHandModel required attributes
        self.valid_frame_names: list[str] = []
        self.remaining_frame_names: list[str] = []
        self.ee_frame_names: list[str] = []

    # -- public properties (all lazy-build the model on first access) -----------

    @property
    def prefix(self) -> str:
        """Joint/link name prefix for this hand side (``"left_"`` or ``"right_"``)."""
        return f"{self.hand_side}_"

    @property
    def model(self) -> pinocchio.Model:
        """The underlying Pinocchio kinematic model."""
        self._ensure_built()
        return self._model  # type: ignore[return-value]

    @property
    def data(self) -> pinocchio.Data:
        """Pinocchio data structure (must be paired with :attr:`model`)."""
        self._ensure_built()
        return self._data  # type: ignore[return-value]

    @property
    def geometry_model(self) -> pinocchio.GeometryModel:
        """Pinocchio visual geometry model for mesh rendering."""
        self._ensure_built()
        return self._geometry_model  # type: ignore[return-value]

    @property
    def geometry_data(self) -> pinocchio.GeometryData:
        """Pinocchio geometry data (updated by :meth:`compute_forward_kinematics`)."""
        self._ensure_built()
        return self._geometry_data  # type: ignore[return-value]

    @property
    def nq(self) -> int:
        """Number of position variables (generalised coordinates)."""
        self._ensure_built()
        return self._model.nq  # type: ignore[union-attr]

    @property
    def nv(self) -> int:
        """Number of velocity variables (tangent space dimension)."""
        self._ensure_built()
        return self._model.nv  # type: ignore[union-attr]

    @property
    def tip_frame_names(self) -> list[str]:
        """Frame (link) names of the five finger-tips."""
        prefix = f"{self.hand_side}_"
        return [prefix + s for s in self._TIP_LINK_SUFFIXES]

    @property
    def valid_frames(self) -> dict[str, pinocchio.Frame]:
        """Mapping of all frame names to their Pinocchio Frame objects."""
        self._ensure_built()
        return {f.name: f for f in self._model.frames}  # type: ignore[union-attr]

    @property
    def tip_frames(self) -> dict[str, pinocchio.Frame]:
        """Mapping of tip-frame names to their Pinocchio Frame objects."""
        self._ensure_built()
        tip_names = set(self.tip_frame_names)
        return {
            name: frame
            for name, frame in self.valid_frames.items()
            if name in tip_names
        }

    @property
    def lower_joint_limits(self) -> list[float]:
        """Lower position limits (rad) for each active joint."""
        self._ensure_built()
        return self._model.lowerPositionLimit.tolist()  # type: ignore[union-attr]

    @property
    def upper_joint_limits(self) -> list[float]:
        """Upper position limits (rad) for each active joint."""
        self._ensure_built()
        return self._model.upperPositionLimit.tolist()  # type: ignore[union-attr]

    # -- kinematics ------------------------------------------------------------

    def compute_forward_kinematics(self, q: np.ndarray) -> None:
        """Compute forward kinematics and update geometry placements.

        Args:
            q: Joint configuration vector of shape ``(nq,)``.
        """
        import pinocchio

        self._ensure_built()
        pinocchio.forwardKinematics(self._model, self._data, q)  # type: ignore[arg-type]
        pinocchio.updateFramePlacements(self._model, self._data)  # type: ignore[arg-type]
        if self._geometry_model is not None and self._geometry_data is not None:
            pinocchio.updateGeometryPlacements(
                self._model,
                self._data,  # type: ignore[arg-type]
                self._geometry_model,
                self._geometry_data,
            )

    def get_frame_id(self, frame_name: str) -> int:
        """Get the frame ID for a given frame name.

        Args:
            frame_name: Full frame name (e.g. ``"left_thumb_tip_link"``).

        Returns:
            Frame ID in the model.

        Raises:
            ValueError: If the frame name is not found.
        """
        self._ensure_built()
        try:
            return self._model.getFrameId(frame_name)  # type: ignore[union-attr]
        except Exception:
            raise ValueError(
                f"Frame '{frame_name}' not found in model. "
                f"Available frames: {list(self.valid_frames.keys())[:20]}..."
            )

    def get_frame_pose(self, frame_name: str) -> pinocchio.SE3:
        """Return the SE3 pose of a named frame in world coordinates.

        Args:
            frame_name: Full frame name (e.g. ``"left_thumb_tip_link"``).

        Returns:
            Pinocchio SE3 placement after the last FK call.

        Raises:
            ValueError: If the frame name is not found.
        """
        frame_id = self.get_frame_id(frame_name)
        return self._data.oMf[frame_id]  # type: ignore[union-attr, index]

    def get_keypoint_targets(self) -> list[tuple[str, str]]:
        """Get the list of keypoint target pairs for optimisation.

        Returns:
            List of ``(source_frame, destination_frame)`` tuples, one per
            finger.  Source frames are the proximal / metacarpal links;
            destination frames are the tip links.
        """
        p = self.prefix
        return [
            (f"{p}thumb_metacarpal_link", f"{p}thumb_tip_link"),
            (f"{p}index_proximal_link", f"{p}index_tip_link"),
            (f"{p}middle_proximal_link", f"{p}middle_tip_link"),
            (f"{p}ring_proximal_link", f"{p}ring_tip_link"),
            (f"{p}pinky_proximal_link", f"{p}pinky_tip_link"),
        ]

    def get_joint_limits(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Get joint position limits.

        Returns:
            Tuple of ``(lower_limits, upper_limits)`` in radians.
        """
        self._ensure_built()
        return (
            self._model.lowerPositionLimit.copy(),  # type: ignore[union-attr]
            self._model.upperPositionLimit.copy(),  # type: ignore[union-attr]
        )

    # -- internal helpers -------------------------------------------------------

    def _ensure_built(self) -> None:
        if self._built:
            return
        try:
            self._build()
        except Exception as exc:
            raise RuntimeError(
                f"BrainCoModel build failed for hand_side={self.hand_side!r}. "
                f"Ensure revo2_description is vendored at "
                f"{_VENDOR_ROOT / 'urdf' / f'revo2_{self.hand_side}_hand.urdf'} "
                f"and pinocchio is installed."
            ) from exc

    def _build(self) -> None:
        """Build the Pinocchio model and geometry from the vendored URDF."""
        import pinocchio

        urdf_path = self._resolve_urdf()

        # Pinocchio resolves ``package://`` via ``ROS_PACKAGE_PATH``.
        vendor_parent = str(_VENDOR_ROOT.parent.resolve())
        old_ros_path = os.environ.get("ROS_PACKAGE_PATH", "")
        os.environ["ROS_PACKAGE_PATH"] = vendor_parent
        try:
            self._model = pinocchio.buildModelFromUrdf(str(urdf_path))
            self._geometry_model = pinocchio.buildGeomFromUrdf(
                self._model, str(urdf_path), pinocchio.VISUAL
            )
        finally:
            if old_ros_path:
                os.environ["ROS_PACKAGE_PATH"] = old_ros_path
            else:
                os.environ.pop("ROS_PACKAGE_PATH", None)

        self._data = self._model.createData()
        self._geometry_data = pinocchio.GeometryData(self._geometry_model)
        self._built = True

    def _resolve_urdf(self) -> pathlib.Path:
        urdf_path = _VENDOR_ROOT / "urdf" / f"revo2_{self.hand_side}_hand.urdf"
        if not urdf_path.is_file():
            raise FileNotFoundError(
                f"URDF not found: {urdf_path}\n"
                f"Make sure the revo2_description vendor is cloned to "
                f"{_VENDOR_ROOT}"
            )
        return urdf_path
