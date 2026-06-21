"""BrainCoInterface -- RobotInterface adapter for multi-process architecture.

:class:`BrainCoInterface` implements the
:class:`~dexim.core.robot_interface.RobotInterface` protocol but routes all
I/O through SeqLock-protected shared-memory buffers instead of calling
hardware directly.

On :meth:`connect` it:

1. Creates the *Target* and *State* shared-memory blocks.
2. Spawns the :func:`~dexim.brainco.interface.hardware_core.run_hardware_core`
   child process (P1), which owns the hardware backend (Modbus RS-485) and
   runs the strict control loop.

Callers (the Brain & Gateway process) then use :meth:`write` and :meth:`read`
as usual -- those calls are now microsecond-fast shared-memory operations rather
than blocking hardware I/O.

The Hardware Core process stops automatically when the parent calls
:meth:`disconnect` (or when the parent process exits, since it is a daemon
process).
"""

from __future__ import annotations

import multiprocessing
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np
from dexim.core.ipc import (
    StateBufferReader,
    TargetBufferWriter,
    create_state_shm,
    create_target_shm,
    make_shm_name,
)
from dexim.core.robot_interface import JointCommand, JointState, RobotInterface
from loguru import logger


class BrainCoInterface(RobotInterface):
    """RobotInterface that communicates with the Hardware Core via shared memory.

    This is the RobotInterface the Brain & Gateway process (and the node) uses
    for all real hardware interaction.  The actual hardware communication is
    delegated to a child OS process (P1) spawned during :meth:`connect`.

    Args:
        backend_type: Transport backend -- ``"modbus_rs485"``.
        backend_config: Dict of kwargs forwarded to the backend inside the
            Hardware Core process.  Keys: ``port``, ``baud``, ``slave_id``,
            ``hand_side``, ``auto_detect``, ``auto_detect_quick``,
            ``auto_calibrate``.
        num_joints: Number of active joints (default 6 for Revo2).
        rate_hz: Hardware Core loop rate in Hz (default 30).
        shm_prefix: Prefix for shared-memory block names (used to create
            collision-resistant identifiers, e.g. ``"brainco_left"``).
        carrot_lookahead_cycles: Carrot extrapolation look-ahead in control
            cycles (may be fractional).  Forwarded to the Hardware Core
            process.  ``0.0`` (default) disables extrapolation.
    """

    def __init__(
        self,
        backend_type: str,
        backend_config: dict[str, Any],
        num_joints: int = 6,
        rate_hz: float = 30.0,
        shm_prefix: str = "brainco",
        carrot_lookahead_cycles: float = 0.0,
    ) -> None:
        self._backend_type = backend_type
        self._backend_config = backend_config
        self._nq = num_joints
        self._rate_hz = rate_hz
        self._shm_prefix = shm_prefix
        self._carrot_lookahead_cycles = carrot_lookahead_cycles

        # Populated by connect().
        self._target_shm: SharedMemory | None = None
        self._state_shm: SharedMemory | None = None
        self._target_writer: TargetBufferWriter | None = None
        self._state_reader: StateBufferReader | None = None
        self._target_shm_name: str = ""
        self._state_shm_name: str = ""

        self._stop_event: multiprocessing.Event | None = None  # type: ignore[type-arg]
        self._hw_process: multiprocessing.Process | None = None
        self._connected: bool = False
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # RobotInterface lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Create shared memory and spawn the Hardware Core process.

        Raises:
            RuntimeError: If already connected.
        """
        if self._connected:
            raise RuntimeError("BrainCoInterface already connected")

        # Create unique SHM names for this node instance.
        self._target_shm_name = make_shm_name(self._shm_prefix, "tgt")
        self._state_shm_name = make_shm_name(self._shm_prefix, "sta")

        # Create the shared-memory blocks (zero-initialised by helpers).
        self._target_shm = create_target_shm(self._target_shm_name)
        self._state_shm = create_state_shm(self._state_shm_name)

        # Build typed wrappers for the parent process (Brain side).
        self._target_writer = TargetBufferWriter(self._target_shm, self._nq)
        self._state_reader = StateBufferReader(self._state_shm, self._nq)

        # Write the safe position (zeros = open hand) into the target buffer so
        # the Hardware Core starts from a known state.
        self._target_writer.write(np.zeros(self._nq, dtype=np.float32))

        # Spawn the Hardware Core process.
        from dexim.brainco.interface.hardware_core import run_hardware_core

        self._stop_event = multiprocessing.Event()
        self._hw_process = multiprocessing.Process(
            target=run_hardware_core,
            args=(
                self._backend_type,
                self._backend_config,
                self._target_shm_name,
                self._state_shm_name,
                self._stop_event,
                self._rate_hz,
                self._nq,
            ),
            kwargs={
                "log_file": f"logs/hw-core-{self._shm_prefix}.log",
                "carrot_lookahead_cycles": self._carrot_lookahead_cycles,
            },
            daemon=True,
            name="brainco-hw-core",
        )
        self._hw_process.start()
        self._connected = True
        self._start_time = time.time()

        logger.info(
            f"BrainCoInterface connected -- backend={self._backend_type}, "
            f"hw-core PID {self._hw_process.pid}, "
            f"target={self._target_shm_name}, state={self._state_shm_name}"
        )

    def disconnect(self) -> None:
        """Signal the Hardware Core process to stop, join it, and release shared memory."""
        if not self._connected:
            return

        logger.info("BrainCoInterface disconnecting -- stopping hw-core")

        if self._stop_event is not None:
            self._stop_event.set()

        if self._hw_process is not None:
            self._hw_process.join(timeout=5.0)
            if self._hw_process.is_alive():
                logger.warning("[hw-core] did not exit within 5 s -- terminating")
                self._hw_process.terminate()
                self._hw_process.join(timeout=2.0)

        self._release_shm()
        self._connected = False
        logger.info("BrainCoInterface disconnected")

    def is_connected(self) -> bool:
        """Return whether the Hardware Core process is alive."""
        if not self._connected:
            return False
        if self._hw_process is None:
            return False
        return self._hw_process.is_alive()

    # ------------------------------------------------------------------
    # RobotInterface read / write
    # ------------------------------------------------------------------

    def write(self, cmd: JointCommand) -> None:
        """Write joint targets into the shared-memory target buffer.

        The Hardware Core process reads this buffer on its next 30 Hz tick
        (<= 33 ms latency).

        Args:
            cmd: Joint command.  Only ``cmd.q`` (position) is forwarded;
                 velocity and torque entries are ignored (Revo2 is
                 position-only).

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or self._target_writer is None:
            raise RuntimeError("BrainCoInterface is not connected")
        if cmd.q is None:
            return
        self._target_writer.write(cmd.q, time.time())

    def read(self) -> JointState:
        """Read the latest joint state from the shared-memory state buffer.

        The Hardware Core process updates this buffer every 33 ms.  The read
        is lock-free and completes in microseconds.

        Returns:
            :class:`~dexim.core.robot_interface.JointState` with the
            most-recent telemetry (velocities and torques are zeros for the
            Revo2 hand, which does not report them via Modbus).

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or self._state_reader is None:
            raise RuntimeError("BrainCoInterface is not connected")
        angles, vels, taus, timestamp = self._state_reader.read()
        return JointState(
            q=angles.astype(np.float64),
            qd=vels.astype(np.float64),
            tau=taus.astype(np.float64),
            stamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _release_shm(self) -> None:
        """Close and unlink shared memory."""
        if self._target_shm is not None:
            try:
                self._target_shm.close()
            except Exception:
                pass
            try:
                self._target_shm.unlink()
            except Exception:
                pass
            self._target_shm = None
        if self._state_shm is not None:
            try:
                self._state_shm.close()
            except Exception:
                pass
            try:
                self._state_shm.unlink()
            except Exception:
                pass
            self._state_shm = None
