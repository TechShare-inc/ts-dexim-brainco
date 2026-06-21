"""
dexim.brainco.node -- Control node for BrainCo Revo2 dexterous hand.

Classes:
    BrainCoControlNode:  Main control node (collaborator-composed orchestrator)
    BrainCoNodeConfig:   Top-level configuration dataclass

Example:
    >>> from dexim.brainco.node import BrainCoControlNode, BrainCoNodeConfig
    >>> config = BrainCoNodeConfig()
    >>> node = BrainCoControlNode(node_id="brainco_left", config=config, subscriber=...)
    >>> node.run()
"""

from dexim.core.collaborators import (
    DataPlanePublisher,
    JointFilter,
    MotionController,
    PipelineProfiler,
    SkeletonReceiver,
)

from .config import (
    BrainCoConfig,
    BrainCoHwConfig,
    BrainCoNodeConfig,
    BrainCoRS485Config,
    ControlConfig,
    FeatureExtractionConfig,
    FilterConfig,
    HardwareCoreConfig,
    InterfaceConfig,
    SubscriberConfig,
    SubscriberProtocol,
    VizConfig,
    WaitableSubscriberProtocol,
    load_config,
)
from .extractor import FeatureExtractor
from .node import BrainCoControlNode
from .retargeter import Retargeter

__all__ = [
    # Node
    "BrainCoControlNode",
    # Config
    "BrainCoNodeConfig",
    "BrainCoConfig",
    "BrainCoHwConfig",
    "BrainCoRS485Config",
    "SubscriberConfig",
    "SubscriberProtocol",
    "WaitableSubscriberProtocol",
    "InterfaceConfig",
    "ControlConfig",
    "HardwareCoreConfig",
    "FeatureExtractionConfig",
    "FilterConfig",
    "VizConfig",
    "load_config",
    # Collaborators (useful for testing)
    "SkeletonReceiver",
    "FeatureExtractor",
    "Retargeter",
    "JointFilter",
    "MotionController",
    "DataPlanePublisher",
    "PipelineProfiler",
]
