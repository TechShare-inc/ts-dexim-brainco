"""BrainCo Revo2 kinematic model (placeholder).

Full implementation pending URDF asset vendoring from
BrainCoTech/brainco_hand_ros2 (revo2_description package).
"""

from __future__ import annotations


class BrainCoModel:
    """Pinocchio-based kinematic model for BrainCo Revo2 hand.

    Placeholder -- will be populated after URDF assets are vendored.
    """

    def __init__(self) -> None:
        self.nq: int = 6
        self.nv: int = 6
        self.model = None
        self.tip_frame_names: list[str] = []
        self.lower_joint_limits: list[float] = [0.0] * 6
        self.upper_joint_limits: list[float] = [0.0] * 6
