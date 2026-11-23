"""Focused tests for the BrainCo Hardware Core spec over the shared loop."""

from __future__ import annotations

import threading

import numpy as np
from dexim.core import legacy_hardware_core
from dexim.core.ipc import (
    StateBufferReader,
    TargetBufferWriter,
    create_state_shm,
    create_target_shm,
)
from dexim.core.robot_interface import JointCommand, JointState

from dexim.brainco.interface import backends, hardware_core
from dexim.brainco.interface._conversion import JOINT_LIMITS, NUM_JOINTS


class FakeBackend:
    """In-memory backend that stops the loop after a fixed number of reads."""

    instances: list[FakeBackend] = []

    def __init__(self, stop: threading.Event, ticks: int, fail_read: bool = False):
        self.stop = stop
        self.ticks = ticks
        self.fail_read = fail_read
        self.commands: list[np.ndarray] = []
        self.reads = 0
        self.connected = False
        self.disconnected = False
        FakeBackend.instances.append(self)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def write(self, command: JointCommand) -> None:
        self.commands.append(np.array(command.q, copy=True))

    def read(self) -> JointState:
        self.reads += 1
        if self.reads >= self.ticks:
            self.stop.set()
        if self.fail_read:
            raise RuntimeError("bus down")
        q = np.full(NUM_JOINTS, 0.25)
        return JointState(
            q=q, qd=np.full(NUM_JOINTS, 0.5), tau=np.full(NUM_JOINTS, 0.75), stamp=0.0
        )


def _run(monkeypatch, tmp_path, *, ticks=3, fail_read=False, lookahead=0.0):
    stop = threading.Event()
    # The shared logger setup removes every loguru sink, including the one the
    # suite's caplog bridge owns; it is covered by dexim-core's own tests.
    monkeypatch.setattr(
        legacy_hardware_core, "configure_process_logger", lambda *a, **k: None
    )
    FakeBackend.instances.clear()
    monkeypatch.setattr(
        backends,
        "create_backend",
        lambda t, c: FakeBackend(stop, ticks, fail_read),
    )
    # signal.signal needs the main thread; the test runs there.
    tgt_name, sta_name = f"bc_tgt_{id(stop)}", f"bc_sta_{id(stop)}"
    tgt, sta = create_target_shm(tgt_name), create_state_shm(sta_name)
    try:
        TargetBufferWriter(tgt, NUM_JOINTS).write(np.full(NUM_JOINTS, 0.1, np.float32))
        hardware_core.run_hardware_core(
            "modbus_rs485",
            {"port": "COM_FAKE"},
            tgt_name,
            sta_name,
            stop,
            rate_hz=1000.0,
            num_joints=NUM_JOINTS,
            log_file=str(tmp_path / "hw.log"),
            carrot_lookahead_cycles=lookahead,
        )
        reader = StateBufferReader(sta, NUM_JOINTS)
        q, qd, tau, _ = reader.read()
        reader.close()
    finally:
        for shm in (tgt, sta):
            shm.close()
            shm.unlink()
    return FakeBackend.instances[0], q, qd, tau


def test_loop_writes_targets_and_publishes_state(monkeypatch, tmp_path):
    backend, q, qd, tau = _run(monkeypatch, tmp_path)
    assert backend.connected and backend.disconnected
    assert len(backend.commands) == 3
    np.testing.assert_allclose(backend.commands[0], 0.1, atol=1e-6)
    np.testing.assert_allclose(q, 0.25)
    np.testing.assert_allclose(qd, 0.5)
    np.testing.assert_allclose(tau, 0.75)


def test_read_failure_republishes_last_known_positions(monkeypatch, tmp_path):
    _, q, qd, tau = _run(monkeypatch, tmp_path, fail_read=True)
    np.testing.assert_array_equal(q, 0.0)
    np.testing.assert_array_equal(qd, 0.0)
    np.testing.assert_array_equal(tau, 0.0)


def test_carrot_extrapolation_clips_to_brainco_limits(monkeypatch, tmp_path):
    backend, *_ = _run(monkeypatch, tmp_path, lookahead=2.0)
    upper = np.array([hi for _, hi in JOINT_LIMITS])
    lower = np.array([lo for lo, _ in JOINT_LIMITS])
    for cmd in backend.commands:
        assert np.all(cmd <= upper + 1e-6) and np.all(cmd >= lower - 1e-6)


def test_joint_limits_helper_matches_conversion_table():
    lo, hi = hardware_core._joint_limits()
    assert lo.dtype == hi.dtype == np.float64
    assert lo.shape == hi.shape == (NUM_JOINTS,)
