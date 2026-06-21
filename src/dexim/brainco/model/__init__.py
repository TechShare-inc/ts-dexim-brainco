"""
dexim.brainco.model -- BrainCo Revo2 kinematic model.

Classes:
    BrainCoModel: Pinocchio-based FK and joint metadata for Revo2.

Note:
    This module requires the ``[kinematics]`` extra (pinocchio) and Revo2
    URDF assets.  The URDF is sourced from BrainCoTech/brainco_hand_ros2
    under the ``revo2_description`` package.  See README for license and
    vendoring notes.
"""

__version__ = "0.1.0"

from dexim.brainco.model.factory import create_model
from dexim.brainco.model.model import BrainCoModel

__all__ = ["BrainCoModel", "create_model"]
