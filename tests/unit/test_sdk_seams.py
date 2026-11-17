"""Injected fake API semantics from pinned SDK stubs; never import SDK binaries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from dexim.core.robot_interface import JointCommand

from dexim.brainco.interface.backends.modbus_rs485 import BrainCoModbusRS485Backend
from dexim.brainco.interface.backends.revo3 import Revo3SDKBackend


def revo2_sdk():
    client = SimpleNamespace(
        set_hardware_type=Mock(),
        set_finger_unit_mode=AsyncMock(),
        set_finger_positions=AsyncMock(),
        get_motor_status=AsyncMock(return_value=SimpleNamespace(positions=[0] * 6)),
    )
    sdk = SimpleNamespace(
        modbus_open=AsyncMock(return_value=client),
        modbus_close=Mock(),
        Baudrate=int,
        StarkHardwareType=SimpleNamespace(Revo2Basic=1),
        FingerUnitMode=SimpleNamespace(Normalized=0),
    )
    return sdk, client


def revo3_sdk():
    client = SimpleNamespace(
        state=SimpleNamespace(positions_deg=AsyncMock(return_value=[0] * 21)),
        health=SimpleNamespace(
            snapshot=AsyncMock(
                return_value=SimpleNamespace(
                    error_code=0, system_state=0, faulted_motor_count=0, safety_state=1
                )
            )
        ),
        motion=SimpleNamespace(set_position=AsyncMock()),
        close=AsyncMock(),
    )
    manager = SimpleNamespace(
        connect_auto=AsyncMock(return_value=client), close=AsyncMock()
    )
    return SimpleNamespace(
        Manager=Mock(return_value=manager), SafetyState=SimpleNamespace(Operational=1)
    ), client


@pytest.mark.parametrize(
    "cls,make,n",
    [(BrainCoModbusRS485Backend, revo2_sdk, 6), (Revo3SDKBackend, revo3_sdk, 21)],
)
def test_fake_lifecycle_read_write_failures_and_bus_owner(cls, make, n):
    sdk, client = make()
    backend = cls(port="fake-only", fake_sdk=sdk)
    backend.connect()
    try:
        state = backend.read()
        assert len(state.q) == n and np.isnan(state.tau).all()
        backend.write(JointCommand(q=np.zeros(n)))
        second = cls(port="fake-only", fake_sdk=sdk)
        with pytest.raises(RuntimeError, match="single owner"):
            second.connect()
        # A failed claimant must not release the first owner's reservation.
        second.disconnect()
        third = cls(port="fake-only", fake_sdk=sdk)
        with pytest.raises(RuntimeError, match="single owner"):
            third.connect()
        if n == 6:
            client.get_motor_status.side_effect = TimeoutError("lost")
        else:
            client.state.positions_deg.side_effect = TimeoutError("lost")
        with pytest.raises(TimeoutError):
            backend.read()
    finally:
        backend.disconnect()
    if n == 6:
        sdk.modbus_close.assert_called_once_with(client)
        client.set_finger_unit_mode.assert_awaited_once_with(126, 0)
    else:
        client.close.assert_awaited_once()


@pytest.mark.parametrize("cls", [BrainCoModbusRS485Backend, Revo3SDKBackend])
def test_hardware_is_fail_closed(cls):
    with pytest.raises(RuntimeError, match="qualification|qualify"):
        cls(port="never-open").connect()
    with pytest.raises(ValueError, match="calibration"):
        cls(auto_calibrate=True)


def test_timeout_and_malformed_readback_not_cached():
    sdk, client = revo2_sdk()
    backend = BrainCoModbusRS485Backend(
        port="fake-timeout", fake_sdk=sdk, timeout_s=0.01
    )
    backend.connect()
    try:
        client.get_motor_status.return_value = SimpleNamespace(positions=[0])
        with pytest.raises(ValueError):
            backend.read()

        async def stall(*args):
            await asyncio.sleep(10)

        client.set_finger_positions.side_effect = stall
        with pytest.raises(asyncio.TimeoutError):
            backend.write(JointCommand(q=np.zeros(6)))
    finally:
        backend.disconnect()


def test_revo3_one_shot_write_and_health_fail_closed():
    sdk, client = revo3_sdk()
    backend = Revo3SDKBackend(port="fake-health", fake_sdk=sdk)
    backend.connect()
    try:
        backend.write(JointCommand(q=np.zeros(21)))
        client.motion.set_position.assert_awaited_once()
        for safety in (None, 0, 2, 3):
            client.health.snapshot.return_value.safety_state = safety
            with pytest.raises(RuntimeError, match="unhealthy"):
                backend.read()
    finally:
        backend.disconnect()
