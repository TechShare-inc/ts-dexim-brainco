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
configured by :func:`~dexim.brainco.interface._process_logger.configure_process_logger`:
stderr at INFO and a rotating file sink (TRACE+, ``enqueue=True``) so that
file I/O never blocks the real-time RS-485 loop.
"""

from __future__ import annotations

import atexit
import signal
from multiprocessing.synchronize import Event as MPEvent
from typing import Any

import numpy as np
from dexim.core.ipc import (
    StateBufferWriter,
    TargetBufferReader,
    open_state_shm,
    open_target_shm,
)
from dexim.core.nodes.utils import RateLimiter
from loguru import logger

from dexim.brainco.interface._process_logger import configure_process_logger


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
    configure_process_logger(log_file, process_name="hw-core")

    # -----------------------------------------------------------------------
    # Signal handling -- treat SIGTERM/SIGINT as a clean stop signal.
    # -----------------------------------------------------------------------
    def _handle_signal(signum: int, frame: Any) -> None:  # noqa: ANN001
        logger.debug(f"[hw-core] received signal {signum}, stopping")
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # -----------------------------------------------------------------------
    # Build the backend interface in this process.
    # -----------------------------------------------------------------------
    from dexim.brainco.interface.backends import create_backend

    interface = create_backend(backend_type, backend_config)

    try:
        interface.connect()
    except Exception as exc:
        logger.error(f"[hw-core] Failed to connect {backend_type} interface: {exc}")
        return

    # -----------------------------------------------------------------------
    # Attach to shared-memory blocks (created by the parent).
    # -----------------------------------------------------------------------
    try:
        target_shm = open_target_shm(target_shm_name)
        state_shm = open_state_shm(state_shm_name)
    except Exception as exc:
        logger.error(f"[hw-core] Failed to open shared memory: {exc}")
        interface.disconnect()
        return

    target_reader = TargetBufferReader(target_shm, num_joints=num_joints)
    state_writer = StateBufferWriter(state_shm, num_joints=num_joints)

    # Register cleanup so SHM is released even on ungraceful exit.
    def _cleanup() -> None:
        try:
            target_shm.close()
        except Exception:
            pass
        try:
            state_shm.close()
        except Exception:
            pass

    atexit.register(_cleanup)

    # -----------------------------------------------------------------------
    # Main RS-485 loop.
    # -----------------------------------------------------------------------
    from dexim.core.nodes.utils import TargetExtrapolator
    from dexim.core.robot_interface import JointCommand

    rate_limiter = RateLimiter(rate_hz=rate_hz)
    last_known_q = np.zeros(num_joints, dtype=np.float32)

    # Build carrot extrapolator (optional).
    extrapolator: TargetExtrapolator | None = None
    if carrot_lookahead_cycles > 0.0:
        from dexim.brainco.interface._conversion import JOINT_LIMITS

        joint_min = np.array([lo for lo, _ in JOINT_LIMITS], dtype=np.float64)
        joint_max = np.array([hi for _, hi in JOINT_LIMITS], dtype=np.float64)
        extrapolator = TargetExtrapolator(
            num_joints=num_joints,
            dt=1.0 / rate_hz,
            joint_min=joint_min,
            joint_max=joint_max,
            lookahead_cycles=carrot_lookahead_cycles,
        )

    logger.info(
        f"[hw-core] Entering control loop -- "
        f"backend={backend_type}, rate_hz={rate_hz}, nq={num_joints}"
    )

    try:
        while not stop_event.is_set():
            # 1. Read the latest target from the parent process.
            target_angles, _, _, _ = target_reader.read()

            # 2. Optionally extrapolate for smoothness.
            if extrapolator is not None:
                target_angles = extrapolator.push(
                    target_angles.astype(np.float64)
                ).astype(np.float32)

            # 3. Send target to hardware.
            interface.write(JointCommand(mode="position", q=target_angles))

            # 4. Read hardware state.
            try:
                state = interface.read()
                last_known_q = state.q.astype(np.float32)
            except Exception:
                logger.exception("[hw-core] read failed, using last known state")

            # 5. Write state to shared memory for parent.
            state_writer.write(
                last_known_q,
                np.zeros(num_joints, dtype=np.float32),
                np.zeros(num_joints, dtype=np.float32),
            )

            # 6. Sleep until next tick.
            rate_limiter.sleep()

    except Exception:
        logger.exception("[hw-core] Fatal error in control loop")
    finally:
        logger.info("[hw-core] Control loop exited -- cleaning up")
        try:
            interface.disconnect()
        except Exception:
            logger.exception("[hw-core] Error during disconnect")
        _cleanup()
