"""Parent-side BrainCoInterface behaviour over the shared Hardware Core."""

from __future__ import annotations

import numpy as np
import pytest
from dexim.core.ipc import StateBufferWriter, StateFault, TargetBufferReader
from dexim.core.robot_interface import JointCommand

from dexim.brainco.interface import hardware_core
from dexim.brainco.interface._conversion import NUM_JOINTS
from dexim.brainco.interface.brainco_interface import BrainCoInterface


class FakeProcess:
    """Stand-in for the hw-core process; the core never actually runs."""

    def __init__(self, *args, **kwargs):
        self.alive = True
        self.pid = 1

    def start(self):
        pass

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.alive = False

    def terminate(self):
        self.alive = False


@pytest.fixture
def interface(monkeypatch):
    monkeypatch.setattr(
        "dexim.brainco.interface.brainco_interface.multiprocessing.Process",
        FakeProcess,
    )
    iface = BrainCoInterface("modbus_rs485", {"port": "COM_FAKE"})
    iface.connect()
    yield iface
    iface.disconnect()


def _latch(iface, fault):
    shm = iface._state_shm
    writer = StateBufferWriter(shm, num_joints=NUM_JOINTS)
    writer.set_fault(fault)
    writer.close()


def test_connect_publishes_no_target(interface):
    reader = TargetBufferReader(interface._target_shm, num_joints=NUM_JOINTS)
    _, timestamp = reader.read()
    reader.close()
    assert timestamp == 0.0


def test_write_publishes_a_target(interface):
    interface.write(JointCommand(q=np.full(NUM_JOINTS, 0.2), mode="position"))
    reader = TargetBufferReader(interface._target_shm, num_joints=NUM_JOINTS)
    q, timestamp = reader.read()
    reader.close()
    assert timestamp > 0.0
    np.testing.assert_allclose(q, 0.2, atol=1e-6)


def test_healthy_core_is_connected_and_readable(interface):
    assert interface.fault() is StateFault.NONE
    assert interface.is_connected()
    interface.read()


@pytest.mark.parametrize(
    "fault",
    [StateFault.READ_FAILURES, StateFault.WRITE_FAILURES, StateFault.LOOP_EXITED],
)
def test_latched_fault_disconnects_and_fails_read(interface, fault):
    _latch(interface, fault)
    assert interface.fault() is fault
    assert not interface.is_connected()
    with pytest.raises(RuntimeError, match=fault.name):
        interface.read()


def test_dead_core_fails_read_and_is_disconnected(interface):
    interface._hw_process.alive = False
    assert not interface.is_connected()
    with pytest.raises(RuntimeError, match="not running"):
        interface.read()


def test_fault_is_none_before_connect():
    assert BrainCoInterface("modbus_rs485", {}).fault() is StateFault.NONE


def test_interface_spawns_the_hardware_core_entry_point(monkeypatch):
    seen = {}

    class Spy(FakeProcess):
        def __init__(self, *args, **kwargs):
            super().__init__()
            seen["target"] = kwargs["target"]

    monkeypatch.setattr(
        "dexim.brainco.interface.brainco_interface.multiprocessing.Process", Spy
    )
    iface = BrainCoInterface("modbus_rs485", {"port": "COM_FAKE"})
    iface.connect()
    iface.disconnect()
    assert seen["target"] is hardware_core.run_hardware_core
