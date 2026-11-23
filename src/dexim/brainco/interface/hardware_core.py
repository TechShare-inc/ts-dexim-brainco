"""Hardware Core process -- strict hardware control loop for BrainCo Revo2.

This module provides :func:`run_hardware_core`, the entry point for the
*Hardware Core* OS process.  It is the only process that touches the hardware
backend (Modbus RS-485) and it does exactly one thing:

1. Read the latest target joint angles from the *Target* shared-memory buffer
   (written by the Brain & Gateway process).
2. Send those targets to the hardware backend.
3. Read the actual joint state (telemetry) from the hardware backend.
4. Write the actual state into the *State* shared-memory buffer (read by the
   Brain & Gateway process).
5. Sleep until the next tick.

**No complex math, no ZMQ calls happen here.**  The process logger is
configured by :func:`~dexim.core.process_logger.configure_process_logger`:
stderr at INFO and a rotating file sink (TRACE+, ``enqueue=True``) so that
file I/O never blocks the real-time RS-485 loop.

The loop itself is :func:`~dexim.core.legacy_hardware_core.run_legacy_hardware_core`;
this module only supplies the BrainCo backend, joint limits and policies.
"""

from __future__ import annotations

from multiprocessing.synchronize import Event as MPEvent
from typing import Any

import numpy as np
from dexim.core.legacy_hardware_core import (
    LegacyHardwareCoreSpec,
    ReadState,
    StateSample,
    make_state_reader,
    run_legacy_hardware_core,
)
from dexim.core.robot_interface import RobotInterface

# Consecutive failed writes before the core latches StateFault.WRITE_FAILURES.
# The shared default (5, about 170 ms at 30 Hz) is kept deliberately: the SDK
# backend bounds each send with a timeout and propagates it as an exception, so
# one timeout on a shared RS-485 bus is a transient fault, not proof of a dead
# link.  Stopping on the first one would fault the hand on a single glitch.
# Persistent failure still faults within a few ticks.
MAX_CONSECUTIVE_WRITE_FAILURES = 5
MAX_CONSECUTIVE_READ_FAILURES = 5


def make_brainco_state_reader(num_joints: int) -> ReadState:
    """Build the BrainCo read policy.

    The SDK backend reports velocity and torque as NaN because the hand does
    not measure them.  The State buffer documents zeros for unavailable
    channels, so non-finite ``qd`` and ``tau`` become zero.  Joint positions
    stay strictly validated: a bad shape or a non-finite ``q`` is a failed read.
    """
    validate = make_state_reader(num_joints)

    def read_state(interface: RobotInterface) -> StateSample:
        state = interface.read()
        sample = StateSample(
            q=state.q.astype(np.float32),
            qd=np.nan_to_num(
                state.qd.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            ),
            tau=np.nan_to_num(
                state.tau.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            ),
        )
        # Reuse the shared shape/finite checks on the sanitised sample.
        return validate(_Sampled(sample))

    return read_state


class _Sampled:
    """Adapter that replays one already-read sample through a shared reader."""

    def __init__(self, sample: StateSample) -> None:
        self._sample = sample

    def read(self) -> StateSample:
        return self._sample


def _joint_limits() -> tuple[np.ndarray, np.ndarray]:
    from dexim.brainco.interface._conversion import JOINT_LIMITS

    return (
        np.array([lo for lo, _ in JOINT_LIMITS], dtype=np.float64),
        np.array([hi for _, hi in JOINT_LIMITS], dtype=np.float64),
    )


def run_hardware_core(
    backend_type: str,
    backend_config: dict[str, Any],
    target_shm_name: str,
    state_shm_name: str,
    stop_event: MPEvent,
    rate_hz: float = 30.0,
    num_joints: int = 6,
    log_file: str | None = None,
    carrot_lookahead_cycles: float = 0.0,
) -> None:
    """Entry point for the Hardware Core process.

    Creates the hardware backend interface **inside** this process (transport
    objects must not be shared across process boundaries), then runs the tight
    read/write loop until *stop_event* is set.

    Args:
        backend_type: Backend transport identifier -- ``"modbus_rs485"``.
        backend_config: Dict of kwargs forwarded to the backend interface
            constructor.  Keys: ``port``, ``baud``, ``slave_id``,
            ``hand_side``, ``auto_detect``, ``auto_detect_quick``,
            ``auto_calibrate``.
        target_shm_name: Name of the target shared-memory block (created by
            the parent process via :func:`~dexim.core.ipc.create_target_shm`).
        state_shm_name: Name of the state shared-memory block (created by
            the parent process via :func:`~dexim.core.ipc.create_state_shm`).
        stop_event: :class:`multiprocessing.Event` set by the parent process
            to request a clean shutdown.
        rate_hz: Target loop rate in Hz (default 30 -> ~33 ms period).
        num_joints: Number of active joints (default 6 for Revo2).
        log_file: Path for the process-local log file, or ``None`` to log to
            stderr only.  Passed by the parent via
            :class:`~dexim.brainco.interface.brainco_interface.BrainCoInterface`.
        carrot_lookahead_cycles: Carrot extrapolation look-ahead in control
            cycles (may be fractional).  ``0.0`` (default) disables
            extrapolation.
    """
    from dexim.brainco.interface.backends import create_backend

    spec = LegacyHardwareCoreSpec(
        backend_factory=lambda: create_backend(backend_type, backend_config),
        backend_label=backend_type,
        connection_hint=str(backend_config.get("port", "?")),
        read_state=make_brainco_state_reader(num_joints),
        max_consecutive_write_failures=MAX_CONSECUTIVE_WRITE_FAILURES,
        max_consecutive_read_failures=MAX_CONSECUTIVE_READ_FAILURES,
        joint_limits=_joint_limits,
    )
    run_legacy_hardware_core(
        spec,
        target_shm_name,
        state_shm_name,
        stop_event,
        rate_hz=rate_hz,
        num_joints=num_joints,
        log_file=log_file,
        carrot_lookahead_cycles=carrot_lookahead_cycles,
    )
