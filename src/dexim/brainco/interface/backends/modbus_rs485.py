"""BrainCo Modbus RS-485 backend.

Wraps BrainCo SDK (``bc_stark_sdk``) async APIs behind a synchronous
``BackendProtocol`` boundary for use inside the Hardware Core child process
(P1).  This module is the **only** place that imports ``bc_stark_sdk``.

Implementation notes:
    - The BrainCo SDK uses async APIs.  This backend owns a persistent
      ``asyncio`` event loop and runs coroutines via
      ``loop.run_until_complete(...)``.
    - Position conversion: internal radians <-> BrainCo normalized 0-1000.
    - ``bc_stark_sdk`` is an optional ``[hardware]`` dependency and is only
      imported at runtime (guarded by a ``HAS_SDK`` flag).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from dexim.core.robot_interface import JointCommand, JointState
from loguru import logger

from dexim.brainco.interface._conversion import (
    NUM_JOINTS,
    api_to_radians,
    radians_to_api,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Optional SDK import guard
# ---------------------------------------------------------------------------

HAS_SDK = False
try:
    import bc_stark_sdk.main_mod as _libstark  # noqa: F401

    HAS_SDK = True
except ImportError:
    logger.warning(
        "bc-stark-sdk not installed.  Hardware mode is unavailable.  "
        "Install with: pixi run pip install bc-stark-sdk"
    )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class BrainCoModbusRS485Backend:
    """Synchronous wrapper around BrainCo SDK Modbus RS-485 async APIs.

    This backend owns a persistent ``asyncio`` event loop and the SDK
    ``DeviceContext``.  It runs inside the Hardware Core child process (P1)
    so that blocking serial I/O never stalls the high-level teleoperation
    loop.

    Args:
        port: Serial port name (e.g. ``"COM5"``).  Optional when
            ``auto_detect`` is enabled.
        baud: Baud rate (default ``460800``).
        slave_id: Modbus slave ID (default ``126`` for left, ``127`` for
            right).
        hand_side: ``"left"`` or ``"right"``.
        auto_detect: Use BrainCo SDK auto-detection to find the device.
        auto_detect_quick: Fast detection mode (default ``True``).
        auto_calibrate: Run position calibration on connect
            (default ``False``).
    """

    def __init__(
        self,
        port: str | None = None,
        baud: int = 460800,
        slave_id: int = 126,
        hand_side: str = "left",
        auto_detect: bool = False,
        auto_detect_quick: bool = True,
        auto_calibrate: bool = False,
    ) -> None:
        if not HAS_SDK:
            raise ImportError(
                "bc-stark-sdk is required for hardware mode.  "
                "Install with: pixi run pip install bc-stark-sdk"
            )

        self._port = port
        self._baud = baud
        self._slave_id = slave_id
        self._hand_side = hand_side
        self._auto_detect = auto_detect
        self._auto_detect_quick = auto_detect_quick
        self._auto_calibrate = auto_calibrate

        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any = None

        # Fallback state for when read fails.
        self._last_known_q = np.zeros(NUM_JOINTS, dtype=np.float64)
        self._connected = False

    # ------------------------------------------------------------------
    # BackendProtocol
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open Modbus RS-485 connection to the BrainCo Revo2 hand.

        Creates a persistent asyncio event loop, then:
        1. Either auto-detects or uses explicit port/baud/slave_id.
        2. Opens the Modbus connection.
        3. Sets the hardware type to Revo2Basic.
        4. Optionally auto-calibrates.
        """
        import bc_stark_sdk.main_mod as libstark

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._connect_async(libstark))
        except Exception:
            if self._loop is not None:
                self._loop.close()
                self._loop = None
            raise

        self._connected = True
        logger.info(
            f"BrainCoModbusRS485Backend connected: slave_id={self._slave_id}, "
            f"port={self._port}, baud={self._baud}"
        )

    @staticmethod
    def _baudrate_to_enum(baud: int, libstark: Any) -> Any:
        """Convert an integer baud rate to a ``Baudrate`` enum value.

        The SDK's ``modbus_open`` requires a ``Baudrate`` enum, not a plain
        ``int``.  This helper performs the conversion.  If the value is
        already a ``Baudrate`` instance it is returned unchanged (e.g. when
        it came from ``auto_detect_modbus_revo2``).
        """
        if isinstance(baud, libstark.Baudrate):
            return baud
        try:
            return libstark.Baudrate(baud)
        except ValueError:
            raise ValueError(
                f"Unsupported baud rate: {baud}. "
                f"Supported values: 115200, 57600, 19200, 460800, "
                f"1000000, 2000000, 3000000, 5000000"
            ) from None

    async def _connect_async(self, libstark: Any) -> None:
        """Async connect logic."""
        if self._auto_detect:
            (
                _protocol,
                self._port,
                self._baud,
                self._slave_id,
            ) = await libstark.auto_detect_modbus_revo2(
                port_name=self._port,
                quick=self._auto_detect_quick,
            )
            logger.info(
                f"Auto-detected Revo2: port={self._port}, "
                f"baud={self._baud}, slave_id={self._slave_id}"
            )

        if self._port is None:
            raise ValueError(
                "Serial port is required.  Set 'port' in config or enable auto_detect."
            )

        baudrate = self._baudrate_to_enum(self._baud, libstark)
        self._client = await libstark.modbus_open(self._port, baudrate)

        await self._client.set_hardware_type(
            self._slave_id, libstark.StarkHardwareType.Revo2Basic
        )

        if self._auto_calibrate:
            logger.info("Running auto-calibration...")
            try:
                await self._client.set_finger_unit_mode(
                    self._slave_id, libstark.StarkUnitMode.Radians
                )
                await self._client.calibrate_finger_position(self._slave_id)
            except AttributeError:
                logger.warning(
                    "auto_calibrate: calibrate_finger_position not available "
                    "in this SDK version -- skipping calibration"
                )
            except Exception as exc:
                logger.warning(
                    f"auto_calibrate failed (hand may already be calibrated): {exc}"
                )

    def disconnect(self) -> None:
        """Close the Modbus connection and event loop."""
        if not self._connected:
            return

        if self._client is not None and self._loop is not None:
            try:
                import bc_stark_sdk.main_mod as libstark

                self._loop.run_until_complete(libstark.modbus_close(self._client))
            except RuntimeError:
                # Event loop already stopped (e.g. by a prior read/write
                # error); skip the async close — the serial port will be
                # released when the loop itself is closed below.
                pass
            except Exception:
                logger.exception("Error closing Modbus connection")

        if self._loop is not None:
            self._loop.close()
            self._loop = None

        self._client = None
        self._connected = False
        logger.info("BrainCoModbusRS485Backend disconnected")

    def write(self, cmd: JointCommand) -> None:
        """Send joint position targets to the Revo2 hand.

        Converts internal radians to BrainCo normalized 0-1000 values,
        then calls ``set_finger_positions``.

        Args:
            cmd: Joint command with ``q`` in internal radians order.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or self._loop is None or self._client is None:
            raise RuntimeError("BrainCoModbusRS485Backend is not connected")

        if cmd.q is None:
            return

        api_positions = radians_to_api(cmd.q)
        self._loop.run_until_complete(
            self._client.set_finger_positions(self._slave_id, api_positions.tolist())
        )

    def read(self) -> JointState:
        """Read current motor status from the Revo2 hand.

        Calls ``get_motor_status`` and converts API positions to internal
        radians.  Falls back to the last known position on error.

        Returns:
            JointState with positions in internal radians order.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or self._loop is None or self._client is None:
            raise RuntimeError("BrainCoModbusRS485Backend is not connected")

        try:
            status = self._loop.run_until_complete(
                self._client.get_motor_status(self._slave_id)
            )
            # status.positions is in API order (list of 6 ints).
            q = api_to_radians(status.positions)
            self._last_known_q = q
        except Exception:
            logger.warning(
                "[BrainCoModbusRS485Backend] get_motor_status failed, "
                "returning last known state"
            )
            q = self._last_known_q

        return JointState(
            q=q,
            qd=np.zeros(NUM_JOINTS),
            tau=np.zeros(NUM_JOINTS),
            stamp=time.time(),
        )
