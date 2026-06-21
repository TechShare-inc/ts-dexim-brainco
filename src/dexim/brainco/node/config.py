"""Configuration dataclasses for brainco-node.

This module provides configuration classes for BrainCoControlNode.
Shared configs (SubscriberConfig, ControlConfig, etc.) are imported from core-config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

# Import shared configs from core-config
from dexim.core.collaborators.protocols import (
    WaitableSubscriberProtocol as WaitableSubscriberProtocol,
)
from dexim.core.config import (
    ControlConfig,
    SubscriberConfig,
)

# Import port utilities for centralized port management
from dexim.core.utility import DEFAULT_HOST, NODE_DATA_PORTS, build_endpoint

# Interface-layer configs live in interface/ and are re-exported here for
# callers that build a complete BrainCoNodeConfig.
from dexim.brainco.interface.config import (
    BrainCoHwConfig as BrainCoHwConfig,
)
from dexim.brainco.interface.config import (
    BrainCoRS485Config as BrainCoRS485Config,
)
from dexim.brainco.interface.config import (
    HardwareCoreConfig as HardwareCoreConfig,
)
from dexim.brainco.interface.config import (
    InterfaceConfig as InterfaceConfig,
)


@dataclass
class VizConfig:
    """Configuration for the Viser 3D visualiser.

    Attributes:
        enabled: Whether to start the visualiser.
        host: Viser server bind address.
        port: Viser server port.
        show_frames: Display coordinate frames.
        show_geometry: Display mesh geometry.
        show_ee_spheres: Display end-effector spheres.
        fps: Visualizer frame rate.
    """

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    show_frames: bool = False
    show_geometry: bool = True
    show_ee_spheres: bool = True
    fps: float = 30.0


# Re-export shared configs for convenience
__all__ = [
    "SubscriberProtocol",
    "WaitableSubscriberProtocol",
    "SubscriberConfig",
    "ControlConfig",
    "BrainCoHwConfig",
    "BrainCoRS485Config",
    "InterfaceConfig",
    "HardwareCoreConfig",
    "FeatureExtractionConfig",
    "FilterConfig",
    "BrainCoConfig",
    "BrainCoNodeConfig",
    "VizConfig",
    "load_config",
    # Port utilities
    "get_default_manus_address",
    "NODE_DATA_PORTS",
]


@runtime_checkable
class SubscriberProtocol(Protocol):
    """Structural protocol for any data subscriber used by BrainCoControlNode.

    Defines the methods actually used by :class:`SkeletonReceiver`.
    Any object implementing all three methods satisfies this protocol,
    regardless of its concrete type (ManusSubscriber, mock, etc.).
    """

    def read(self) -> Any:
        """Read and return the next data payload."""
        ...

    def read_latest_batch(self) -> list[Any]:
        """Drain and return all buffered data payloads since the last call."""
        ...

    def close(self) -> None:
        """Release any resources held by the subscriber."""
        ...


def get_default_manus_address() -> str:
    """Get default ZMQ address for subscribing to Manus skeleton data.

    Returns:
        ZMQ endpoint string (e.g., "tcp://localhost:5555")
    """
    return build_endpoint(DEFAULT_HOST, NODE_DATA_PORTS["manus"])


@dataclass
class FeatureExtractionConfig:
    """Configuration for skeleton finger-vector feature extraction.

    Attributes:
        src_indices: Joint indices for vector origins (one per finger).
        dst_indices: Joint indices for vector tips (one per finger).
        apply_rotation: Whether to rotate vectors into hand-local space
            using the root joint orientation.
    """

    src_indices: list[int] = field(default_factory=lambda: [1, 6, 11, 16, 21])
    dst_indices: list[int] = field(default_factory=lambda: [4, 9, 14, 19, 24])
    apply_rotation: bool = True

    def __post_init__(self):
        if len(self.src_indices) != len(self.dst_indices):
            raise ValueError(
                f"src_indices and dst_indices must have the same length, "
                f"got {len(self.src_indices)} vs {len(self.dst_indices)}"
            )
        if not self.src_indices:
            raise ValueError("src_indices must not be empty")


@dataclass
class FilterConfig:
    """Configuration for joint-angle smoothing filters.

    Three filter types are available, selected via ``type``:

    * ``"wma"`` (default) -- Weighted Moving Average (FIR).  Set ``weights``
      (must sum to 1.0).  Smooths well but introduces lag proportional to the
      window length.

    * ``"ema"`` -- Exponential Moving Average (first-order IIR).  Set
      ``alpha`` in ``(0, 1]``.  Lower alpha -> more smoothing, more lag.
      Zero warm-up; responds in one sample.

    * ``"one_euro"`` -- One Euro Filter (adaptive IIR).  Automatically
      reduces smoothing during fast motion and increases it at rest,
      minimising lag while suppressing jitter.  Best choice for interactive
      teleoperation.  Set ``freq`` (Hz), ``min_cutoff`` (Hz), ``beta``
      (speed coefficient), and ``d_cutoff`` (Hz).

    Attributes:
        type: Filter algorithm -- ``"wma"``, ``"ema"``, or ``"one_euro"``.
        weights: WMA weights (must sum to 1.0).  Used when ``type="wma"``.
        alpha: EMA smoothing factor in ``(0, 1]``.  Used when ``type="ema"``.
        freq: Sampling frequency in Hz.  Used when ``type="one_euro"``.
        min_cutoff: Minimum cutoff frequency in Hz.  Used when
            ``type="one_euro"``.
        beta: Speed coefficient >= 0.  Used when ``type="one_euro"``.
        d_cutoff: Derivative cutoff frequency in Hz.  Used when
            ``type="one_euro"``.
    """

    type: str = "wma"

    # WMA parameters
    weights: list[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])

    # EMA parameters
    alpha: float = 0.3

    # One Euro Filter parameters
    freq: float = 30.0
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        """Validate filter configuration."""
        import numpy as np

        allowed = {"wma", "ema", "one_euro"}
        if self.type not in allowed:
            raise ValueError(f"type must be one of {allowed}, got {self.type!r}")

        if self.type == "wma":
            if not isinstance(self.weights, list) or len(self.weights) == 0:
                raise ValueError("weights must be a non-empty list")
            if not np.isclose(sum(self.weights), 1.0):
                raise ValueError(
                    f"Filter weights must sum to 1.0, got {sum(self.weights)}"
                )
        elif self.type == "ema":
            if not (0.0 < self.alpha <= 1.0):
                raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        elif self.type == "one_euro":
            if self.freq <= 0:
                raise ValueError(f"freq must be positive, got {self.freq}")
            if self.min_cutoff <= 0:
                raise ValueError(f"min_cutoff must be positive, got {self.min_cutoff}")
            if self.beta < 0:
                raise ValueError(f"beta must be non-negative, got {self.beta}")
            if self.d_cutoff <= 0:
                raise ValueError(f"d_cutoff must be positive, got {self.d_cutoff}")


@dataclass
class BrainCoConfig:
    """BrainCo Revo2 hand-specific configuration.

    Attributes:
        side: Hand side -- ``"left"`` or ``"right"``.
        feature_extraction: Skeleton finger-vector extraction parameters.
        filter: Optional joint-angle smoothing filter configuration.
    """

    side: str = "left"  # "left" or "right"

    feature_extraction: FeatureExtractionConfig = field(
        default_factory=FeatureExtractionConfig
    )

    # Optional smoothing filter configuration
    filter: FilterConfig | None = None

    def __post_init__(self):
        if self.side not in ["left", "right"]:
            raise ValueError(
                f"Invalid side: {self.side}. Must be 'left' or 'right'"
            )


@dataclass
class BrainCoNodeConfig:
    """Complete configuration for BrainCoControlNode.

    Attributes:
        subscriber: ZMQ subscriber configuration.
        interface: Hardware interface configuration (mock or hw).
        control: Control loop tuning parameters.
        brainco: BrainCo Revo2 hand-specific parameters.
        viz: 3D visualizer configuration.
        data_endpoint: ZMQ endpoint for publishing action/observation data.
        bind_data: Whether to bind the data endpoint.
        observation_rate_hz: Observation publish rate (None = use control rate).
    """

    subscriber: SubscriberConfig = field(default_factory=SubscriberConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    brainco: BrainCoConfig = field(default_factory=BrainCoConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    # Publishing configuration
    data_endpoint: str = "tcp://*:5556"
    bind_data: bool = True
    observation_rate_hz: float | None = None  # None = use control rate

    # Convenience property for robot_type compatibility
    @property
    def robot_type(self) -> str:
        return "brainco"


def load_config(config_path: str) -> BrainCoNodeConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        BrainCoNodeConfig instance
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Build nested config objects
    subscriber = SubscriberConfig(**data.get("subscriber", {}))

    # Build interface config
    interface_data = data.get("interface", {})
    interface_mode = interface_data.get("mode", "mock")

    if interface_mode == "hw":
        hw_data = interface_data.get("hw", {})

        hw_core_data = hw_data.get("hardware_core", {})
        hw_core = (
            HardwareCoreConfig(**hw_core_data)
            if hw_core_data
            else HardwareCoreConfig()
        )

        rs485_data = hw_data.get("rs485")
        rs485 = BrainCoRS485Config(**rs485_data) if rs485_data else BrainCoRS485Config()

        hw = BrainCoHwConfig(
            backend=hw_data.get("backend", "modbus_rs485"),
            rs485=rs485,
            slave_id=hw_data.get("slave_id"),
            auto_detect=hw_data.get("auto_detect", False),
            auto_detect_quick=hw_data.get("auto_detect_quick", True),
            auto_calibrate=hw_data.get("auto_calibrate", False),
            hardware_core=hw_core,
        )
        interface = InterfaceConfig(mode="hw", hw=hw)
    else:
        interface = InterfaceConfig(mode="mock")

    control = ControlConfig(**data.get("control", {}))

    # Build brainco config with optional filter and feature_extraction
    brainco_data = data.get("brainco", {})
    filter_data = brainco_data.get("filter")
    filter_cfg = FilterConfig(**filter_data) if filter_data else None

    # Build FeatureExtractionConfig from dict if present in YAML
    feature_extraction_data = brainco_data.get("feature_extraction")
    feature_extraction_cfg = (
        FeatureExtractionConfig(**feature_extraction_data)
        if isinstance(feature_extraction_data, dict)
        else FeatureExtractionConfig()
    )

    # Remove handled keys from brainco_data to avoid passing them twice
    brainco_data_copy = brainco_data.copy()
    brainco_data_copy.pop("filter", None)
    brainco_data_copy.pop("feature_extraction", None)

    brainco = BrainCoConfig(
        **brainco_data_copy,
        filter=filter_cfg,
        feature_extraction=feature_extraction_cfg,
    )

    # Build viz config
    viz_data = data.get("viz", {})
    viz = VizConfig(**viz_data) if isinstance(viz_data, dict) else VizConfig()

    # Get publishing configuration (with defaults)
    data_endpoint = data.get("data_endpoint", "tcp://*:5556")
    bind_data = data.get("bind_data", True)
    observation_rate_hz = data.get("observation_rate_hz")

    return BrainCoNodeConfig(
        subscriber=subscriber,
        interface=interface,
        control=control,
        brainco=brainco,
        viz=viz,
        data_endpoint=data_endpoint,
        bind_data=bind_data,
        observation_rate_hz=observation_rate_hz,
    )
