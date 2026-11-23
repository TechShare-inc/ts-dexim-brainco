"""Focused tests for the BrainCo Hardware Core spec over the shared loop."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from dexim.core import legacy_hardware_core
from dexim.core.ipc import (
    StateBufferReader,
    StateFault,
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

    def __init__(
        self,
        stop: threading.Event,
        ticks: int,
        fail_read: bool = False,
        fail_write: bool = False,
        nan_dynamics: bool = False,
    ):
        self.stop = stop
        self.ticks = ticks
        self.fail_read = fail_read
        self.fail_write = fail_write
        self.nan_dynamics = nan_dynamics
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
        if self.fail_write:
            raise RuntimeError("write timeout")
        self.commands.append(np.array(command.q, copy=True))

    def read(self) -> JointState:
        self.reads += 1
        if self.reads >= self.ticks:
            self.stop.set()
        if self.fail_read:
            raise RuntimeError("bus down")
        q = np.full(NUM_JOINTS, 0.25)
        if self.nan_dynamics:
            # Mirrors the SDK backend, which has no velocity or torque.
            nan = np.full(NUM_JOINTS, np.nan)
            return JointState(q=q, qd=nan.copy(), tau=nan.copy(), stamp=0.0)
        return JointState(
            q=q, qd=np.full(NUM_JOINTS, 0.5), tau=np.full(NUM_JOINTS, 0.75), stamp=0.0
        )


def _run(
    monkeypatch,
    tmp_path,
    *,
    ticks=3,
    fail_read=False,
    fail_write=False,
    nan_dynamics=False,
    lookahead=0.0,
    publish_target=True,
):
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
        lambda t, c: FakeBackend(stop, ticks, fail_read, fail_write, nan_dynamics),
    )
    # signal.signal needs the main thread; the test runs there.
    tgt_name, sta_name = f"bc_tgt_{id(stop)}", f"bc_sta_{id(stop)}"
    tgt, sta = create_target_shm(tgt_name), create_state_shm(sta_name)
    try:
        if publish_target:
            TargetBufferWriter(tgt, NUM_JOINTS).write(
                np.full(NUM_JOINTS, 0.1, np.float32)
            )
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
        q, qd, tau, stamp = reader.read()
        fault = reader.fault
        reader.close()
    finally:
        for shm in (tgt, sta):
            shm.close()
            shm.unlink()
    return SimpleNamespace(
        backend=FakeBackend.instances[0], q=q, qd=qd, tau=tau, stamp=stamp, fault=fault
    )


def test_loop_writes_targets_and_publishes_state(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path)
    assert run.backend.connected and run.backend.disconnected
    assert len(run.backend.commands) == 3
    np.testing.assert_allclose(run.backend.commands[0], 0.1, atol=1e-6)
    np.testing.assert_allclose(run.q, 0.25)
    np.testing.assert_allclose(run.qd, 0.5)
    np.testing.assert_allclose(run.tau, 0.75)
    assert run.fault is StateFault.NONE


def test_no_write_before_the_first_target(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, publish_target=False)
    assert run.backend.commands == []
    assert run.backend.reads == 3
    np.testing.assert_allclose(run.q, 0.25)
    assert run.fault is StateFault.NONE


def test_few_read_failures_publish_nothing_and_do_not_fault(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, ticks=3, fail_read=True)
    assert run.stamp == 0.0
    np.testing.assert_array_equal(run.q, 0.0)
    assert run.fault is StateFault.NONE


def test_persistent_read_failures_latch_a_fault(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, ticks=100, fail_read=True)
    assert run.backend.reads == hardware_core.MAX_CONSECUTIVE_READ_FAILURES
    assert run.fault is StateFault.READ_FAILURES
    assert run.backend.disconnected


def test_persistent_write_failures_latch_a_fault(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, ticks=100, fail_write=True)
    assert run.backend.reads == hardware_core.MAX_CONSECUTIVE_WRITE_FAILURES - 1
    assert run.fault is StateFault.WRITE_FAILURES
    assert run.backend.disconnected


def test_unavailable_velocity_and_torque_publish_as_zero(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, nan_dynamics=True)
    np.testing.assert_allclose(run.q, 0.25)
    np.testing.assert_array_equal(run.qd, 0.0)
    np.testing.assert_array_equal(run.tau, 0.0)
    assert run.fault is StateFault.NONE


def test_spec_wires_the_new_policies(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        hardware_core,
        "run_legacy_hardware_core",
        lambda spec, *a, **k: captured.update(spec=spec),
    )
    hardware_core.run_hardware_core(
        "modbus_rs485", {"port": "COM_FAKE"}, "t", "s", threading.Event()
    )
    spec = captured["spec"]
    assert spec.write_target is legacy_hardware_core.write_position_target
    assert spec.max_consecutive_write_failures == 5
    assert spec.max_consecutive_read_failures == 5
    assert not hasattr(legacy_hardware_core, "LastKnownStateReader")
    assert not hasattr(legacy_hardware_core, "write_ignoring_errors")


def test_state_reader_rejects_non_finite_positions():
    class Bad:
        def read(self):
            q = np.full(NUM_JOINTS, np.nan)
            return JointState(q=q, qd=q.copy(), tau=q.copy(), stamp=0.0)

    with pytest.raises(ValueError, match="non-finite"):
        hardware_core.make_brainco_state_reader(NUM_JOINTS)(Bad())


def test_carrot_extrapolation_clips_to_brainco_limits(monkeypatch, tmp_path):
    backend = _run(monkeypatch, tmp_path, lookahead=2.0).backend
    upper = np.array([hi for _, hi in JOINT_LIMITS])
    lower = np.array([lo for lo, _ in JOINT_LIMITS])
    for cmd in backend.commands:
        assert np.all(cmd <= upper + 1e-6) and np.all(cmd >= lower - 1e-6)


def test_joint_limits_helper_matches_conversion_table():
    lo, hi = hardware_core._joint_limits()
    assert lo.dtype == hi.dtype == np.float64
    assert lo.shape == hi.shape == (NUM_JOINTS,)
