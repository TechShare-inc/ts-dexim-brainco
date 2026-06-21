"""Interface-layer configuration dataclasses for dexim.brainco.interface.

These dataclasses configure how the BrainCo Revo2 hand interface is built
(transport mode, hardware process tuning).  They are defined here (in the
interface layer) so that :mod:`dexim.brainco.interface.factory` can import
them without reaching up into the node layer.
:mod:`dexim.brainco.node.config` re-exports them for callers that build a
complete ``BrainCoNodeConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Default slave IDs per hand side ───────────────────────────────────────────
DEFAULT_SLAVE_ID_LEFT: int = 126  # 0x7E
DEFAULT_SLAVE_ID_RIGHT: int = 127  # 0x7F


@dataclass
class BrainCoRS485Config:
    """RS-485 transport parameters for BrainCo Revo2.

    Attributes:
        port: Serial port name (e.g. ``"COM5"`` on Windows, ``"/dev/ttyUSB0"``
            on Linux).  Optional when ``auto_detect`` is enabled.
        baud: Baud rate (default ``460800``).
    """

    port: str | None = None
    baud: int = 460800


@dataclass
class HardwareCoreConfig:
    """Configuration for the Hardware Core child process (P1).

    Attributes:
        rate_hz: Hardware loop rate in Hz (default 30 -> ~33 ms period).
        shm_prefix: Prefix for shared-memory block names.  Should be unique
            per node instance to avoid name collisions when running multiple
            nodes on the same machine (e.g. ``"brainco_left"``).  The
            :class:`BrainCoHwConfig` builder appends a side tag automatically,
            so this value only needs to be human-readable.
        carrot_lookahead_cycles: Carrot extrapolation look-ahead expressed in
            control cycles (may be fractional).  Set to ``0.0`` (default) to
            disable extrapolation.  Typical values are 1-3.
    """

    rate_hz: float = 30.0
    shm_prefix: str = "brainco"
    carrot_lookahead_cycles: float = 0.0


@dataclass
class BrainCoHwConfig:
    """Hardware-specific configuration for the BrainCo Revo2 hand interface.

    Groups all hardware transport parameters for ``mode='hw'`` in
    :class:`InterfaceConfig`.  The ``backend`` field selects the transport
    used inside the Hardware Core child process (P1).

    Attributes:
        backend: Hardware backend -- currently ``'modbus_rs485'``.
        rs485: RS-485 serial connection parameters.
        slave_id: Modbus slave ID.  Defaults from *hand_side* if omitted
            (left=126, right=127).
        auto_detect: Whether to use BrainCo SDK auto-detection to find the
            device instead of explicit port/baud/slave_id.
        auto_detect_quick: When ``auto_detect=True``, use fast detection
            (scans common baud rates only).  Default ``True``.
        auto_calibrate: Whether to call position calibration on connect.
            **Default ``False``** -- enable only when hardware safety is
            confirmed and the hand has completed power-on calibration.
        hardware_core: Tuning parameters for the Hardware Core process (P1).
    """

    backend: str = "modbus_rs485"
    rs485: BrainCoRS485Config = field(default_factory=BrainCoRS485Config)
    slave_id: int | None = None
    auto_detect: bool = False
    auto_detect_quick: bool = True
    auto_calibrate: bool = False
    hardware_core: HardwareCoreConfig = field(default_factory=HardwareCoreConfig)

    def __post_init__(self) -> None:
        if self.backend == "modbus_rs485":
            if not self.auto_detect:
                if self.rs485 is None:
                    raise ValueError(
                        "backend='modbus_rs485' requires 'rs485' field "
                        "when auto_detect=False"
                    )
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def effective_slave_id(self, hand_side: str | None = None) -> int:
        """Return the slave ID, defaulting from side if not explicitly set."""
        if self.slave_id is not None:
            return self.slave_id
        if hand_side == "right":
            return DEFAULT_SLAVE_ID_RIGHT
        return DEFAULT_SLAVE_ID_LEFT


@dataclass
class InterfaceConfig:
    """Top-level interface config (mock vs hw) for BrainCo Revo2.

    Attributes:
        mode: Interface mode -- ``'mock'`` (in-process loopback, no hardware)
            or ``'hw'`` (hardware interface via Hardware Core child process).
        hw: Hardware config.  Required when ``mode='hw'``.
    """

    mode: str = "mock"  # 'mock' or 'hw'
    hw: BrainCoHwConfig | None = None

    def __post_init__(self) -> None:
        if self.mode == "mock":
            pass  # no extra config needed
        elif self.mode == "hw":
            if self.hw is None:
                raise ValueError("mode='hw' requires 'hw' field")
        else:
            raise ValueError("mode must be 'mock' or 'hw'")

    def is_mock(self) -> bool:
        """Return True if running in mock (loopback) mode."""
        return self.mode == "mock"

    def is_hw(self) -> bool:
        """Return True if running against hardware."""
        return self.mode == "hw"

    def get_hardware_config(self) -> BrainCoHwConfig | None:
        """Return hardware-specific config for hw mode, or None for mock."""
        if not self.is_hw() or self.hw is None:
            return None
        return self.hw


__all__ = [
    "BrainCoHwConfig",
    "BrainCoRS485Config",
    "HardwareCoreConfig",
    "InterfaceConfig",
    "DEFAULT_SLAVE_ID_LEFT",
    "DEFAULT_SLAVE_ID_RIGHT",
]
