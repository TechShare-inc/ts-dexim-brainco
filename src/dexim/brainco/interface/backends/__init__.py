"""
dexim.brainco.interface.backends -- internal hardware backends for BrainCo Revo2.

Classes:
    BackendProtocol:          Structural protocol for connect/disconnect/read/write.

Functions:
    create_backend:           Factory to instantiate the correct backend inside P1.

Note:
    These backends run exclusively inside the Hardware Core child process (P1)
    and are never imported by the node or the main (Brain & Gateway) process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dexim.core.robot_interface import JointCommand, JointState


@runtime_checkable
class BackendProtocol(Protocol):
    """Structural protocol for hardware backend classes.

    Any class implementing ``connect``, ``disconnect``, ``read``, and ``write``
    satisfies this protocol via structural subtyping.
    """

    def connect(self) -> None:
        """Open hardware connection."""
        ...

    def disconnect(self) -> None:
        """Close hardware connection."""
        ...

    def read(self) -> JointState:
        """Read current joint state from hardware."""
        ...

    def write(self, cmd: JointCommand) -> None:
        """Write joint command to hardware."""
        ...


def create_backend(
    backend_type: str, backend_config: dict[str, Any]
) -> BackendProtocol:
    """Instantiate the correct hardware backend inside the child process.

    This factory is called *inside* the Hardware Core child process (P1) so
    that transport objects are created and owned by the correct OS process.
    Transport objects must not cross process boundaries.

    Args:
        backend_type: ``"modbus_rs485"`` (currently the only supported backend).
        backend_config: Constructor kwargs for the chosen backend class.

    Returns:
        Configured backend instance.

    Raises:
        ValueError: If ``backend_type`` is not recognised.
    """
    if backend_type == "modbus_rs485":
        from dexim.brainco.interface.backends.modbus_rs485 import (
            BrainCoModbusRS485Backend,
        )

        _RS485_KEYS = (
            "port",
            "baud",
            "slave_id",
            "hand_side",
            "auto_detect",
            "auto_detect_quick",
            "auto_calibrate",
        )
        kwargs: dict[str, Any] = {
            k: backend_config[k] for k in _RS485_KEYS if k in backend_config
        }
        return BrainCoModbusRS485Backend(**kwargs)

    raise ValueError(
        f"Unknown backend_type: {backend_type!r}. Expected 'modbus_rs485'."
    )


__all__ = ["BackendProtocol", "create_backend"]
