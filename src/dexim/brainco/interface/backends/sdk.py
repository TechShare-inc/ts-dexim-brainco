"""Bounded SDK fake seams; physical setup remains unconditionally gated.

Validated against bc-stark-sdk 2.0.9 and bc-revo3-sdk 2.3.5 stubs/examples.
No calibration API is invoked. Timeout/close never confirms mechanical stop.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any

import numpy as np
from dexim.core.robot_interface import JointCommand, JointState

from dexim.brainco.offline.model import describe_model


class SDKBackend:
    """Injected fake SDK owner; callers must supply only fakes in offline tests."""

    model = ""
    _owners: set[str] = set()

    def __init__(
        self,
        *,
        port: str | None = None,
        hand_side: str = "left",
        baud: int = 460800,
        slave_id: int = 126,
        auto_detect: bool = False,
        auto_detect_quick: bool = True,
        auto_calibrate: bool = False,
        fake_sdk: Any = None,
        timeout_s: float = 0.1,
    ) -> None:
        self.descriptor = describe_model(self.model, hand_side)
        if auto_detect or auto_calibrate:
            raise ValueError("Scanning/calibration disabled pending qualification")
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Finite positive timeout required")
        self.port, self.baud, self.slave_id = port, baud, slave_id
        self.sdk, self.timeout_s = fake_sdk, timeout_s
        self.loop: asyncio.AbstractEventLoop | None = None
        self.client: Any = None
        self.manager: Any = None
        self.connected = False
        self._owns_bus = False

    def _call(self, result: Any) -> Any:
        if inspect.isawaitable(result):
            return self.loop.run_until_complete(
                asyncio.wait_for(result, self.timeout_s)
            )
        return result

    def connect(self) -> None:
        """Open only an explicitly injected fake; real profiles are unqualified."""
        if self.sdk is None:
            raise RuntimeError(
                "Hardware activation blocked: qualify model, firmware, "
                "SDK, transport, axes/units/limits and stop/readback"
            )
        if not self.port or self.port in self._owners:
            raise RuntimeError("Bus requires explicit identity and a single owner")
        self._owners.add(self.port)
        self._owns_bus = True
        self.loop = asyncio.new_event_loop()
        try:
            if self.model == "revo2":
                self.client = self._call(
                    self.sdk.modbus_open(self.port, self.sdk.Baudrate(self.baud))
                )
                # Stubs use Any: accept sync or awaitable results explicitly.
                self._call(
                    self.client.set_hardware_type(
                        self.slave_id, self.sdk.StarkHardwareType.Revo2Basic
                    )
                )
                self._call(
                    self.client.set_finger_unit_mode(
                        self.slave_id, self.sdk.FingerUnitMode.Normalized
                    )
                )
            else:
                self.manager = self.sdk.Manager()
                self.client = self._call(
                    self.manager.connect_auto(port=self.port, slave_id=self.slave_id)
                )
            self.connected = True
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        """Close transport, propagate failure and release local ownership."""
        try:
            if self.client is not None:
                if self.model == "revo2":
                    self._call(self.sdk.modbus_close(self.client))
                else:
                    self._call(self.client.close())
            if self.manager is not None:
                self._call(self.manager.close())
        finally:
            if self.loop is not None:
                self.loop.close()
            if self._owns_bus:
                self._owners.discard(self.port)
                self._owns_bus = False
            self.loop = self.client = self.manager = None
            self.connected = False

    def write(self, cmd: JointCommand) -> None:
        """Issue bounded position send; propagate timeout/failure unchanged."""
        if not self.connected:
            raise RuntimeError("SDK disconnected")
        if cmd.q is None or cmd.mode != "position":
            raise ValueError("Position command required")
        values = self.descriptor.to_sdk(cmd.q)
        if self.model == "revo2":
            result = self.client.set_finger_positions(
                self.slave_id, np.rint(values).astype(int).tolist()
            )
        else:
            result = self.client.motion.set_position(values.tolist())
        self._call(result)

    def read(self) -> JointState:
        """Read fresh state; never return cached data or invented measured torque."""
        if not self.connected:
            raise RuntimeError("SDK disconnected")
        if self.model == "revo2":
            status = self._call(self.client.get_motor_status(self.slave_id))
            values = status.positions
        else:
            health = self._call(self.client.health.snapshot())
            if (
                health.error_code
                or health.faulted_motor_count
                or health.safety_state != self.sdk.SafetyState.Operational
            ):
                raise RuntimeError("Revo3 unhealthy readback")
            values = self._call(self.client.state.positions_deg())
        q = self.descriptor.from_sdk(values)
        unavailable = np.full(len(q), np.nan)
        return JointState(q.copy(), unavailable.copy(), unavailable.copy(), time.time())
