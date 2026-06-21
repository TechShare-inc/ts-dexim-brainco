"""Factory function for constructing BrainCo RobotInterface instances from InterfaceConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from loguru import logger

from dexim.brainco.interface.config import InterfaceConfig
from dexim.brainco.interface.mock_interface import MockInterface

if TYPE_CHECKING:
    from dexim.brainco.interface.brainco_interface import BrainCoInterface
    from dexim.brainco.model.model import BrainCoModel


def build_interface(
    config: InterfaceConfig,
    model: BrainCoModel | None = None,
    hand_side: Literal["left", "right"] | None = None,
) -> MockInterface | BrainCoInterface:
    """Instantiate the correct RobotInterface for the given InterfaceConfig.

    For ``mode='hw'`` the returned interface is a
    :class:`~dexim.brainco.interface.brainco_interface.BrainCoInterface`
    that spawns the Hardware Core child process (P1) on :meth:`connect`.

    Args:
        config: InterfaceConfig specifying mode and hardware parameters.
        model: BrainCoModel instance (used by MockInterface).
        hand_side: Hand side (``'left'`` or ``'right'``). Required for hw
            mode; defaults to ``'left'`` for mock.

    Returns:
        Configured RobotInterface.

    Raises:
        ValueError: If required fields are missing for the selected mode.
    """
    if config.is_mock():
        logger.debug("Creating MockInterface (loopback)")
        return MockInterface(
            brainco_model=model,
            hand_side=hand_side or "left",
        )

    if hand_side is None:
        raise ValueError("hand_side is required for hw mode")

    hw_config = config.get_hardware_config()
    if hw_config is None:
        raise ValueError("mode='hw' requires a populated 'hw' config field")

    hw_core = hw_config.hardware_core
    shm_prefix = f"{hw_core.shm_prefix}_{hand_side.lower()}"

    from dexim.brainco.interface.brainco_interface import BrainCoInterface  # lazy

    # Build backend config dict from hw_config fields.
    rs485 = hw_config.rs485
    backend_config: dict = {
        "slave_id": hw_config.effective_slave_id(hand_side),
        "hand_side": hand_side,
        "auto_detect": hw_config.auto_detect,
        "auto_detect_quick": hw_config.auto_detect_quick,
        "auto_calibrate": hw_config.auto_calibrate,
    }
    if rs485 is not None:
        if rs485.port:
            backend_config["port"] = rs485.port
        backend_config["baud"] = rs485.baud

    logger.debug(
        f"Creating BrainCoInterface (backend=modbus_rs485, config={backend_config}, "
        f"rate_hz={hw_core.rate_hz}, shm_prefix={shm_prefix!r})"
    )
    return BrainCoInterface(
        backend_type="modbus_rs485",
        backend_config=backend_config,
        num_joints=6,
        rate_hz=hw_core.rate_hz,
        shm_prefix=shm_prefix,
        carrot_lookahead_cycles=hw_core.carrot_lookahead_cycles,
    )
