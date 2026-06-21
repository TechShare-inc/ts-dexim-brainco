"""
dexim.brainco.viz -- 3D visualization for BrainCo Revo2 dexterous hand.

Provides:
- ``BrainCoRenderer``: pure Viser rendering engine.
- ``VizSubscriber``: standalone ZMQ-subscriber-driven visualizer.
- ``ActionMessage``: deserialized action message type for the subscriber.

Depends on dexim.brainco.model for kinematics data.
"""

from dexim.brainco.viz.renderer import BrainCoRenderer
from dexim.brainco.viz.subscriber import ActionMessage, VizSubscriber

__all__ = [
    "ActionMessage",
    "BrainCoRenderer",
    "VizSubscriber",
]
