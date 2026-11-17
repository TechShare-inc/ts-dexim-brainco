"""Epoch-fenced synchronous offline control with independently clocked input."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from dexim.core.messages.types import HandState
from dexim.core.robot_interface import JointCommand

from .config import OfflineConfig
from .hand import normalize_hand, retarget_hand
from .interface import OfflineMockInterface
from .model import describe_model


@dataclass(frozen=True)
class ControlReport:
    """Software outcome; safety never certifies a physical stop."""

    outcome: str
    epoch_id: str | None
    fault: str | None
    safety: str


@dataclass(frozen=True)
class ActionRecord:
    """Issued action after write success; source/host timestamps are Unix."""

    model: str
    side: str
    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    source_timestamp: float
    host_timestamp: float
    epoch_id: str


@dataclass(frozen=True)
class ObservationRecord:
    """Independent mock readback; timestamps are Unix, mimic values derived."""

    model: str
    side: str
    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    full_joint_names: tuple[str, ...]
    full_positions: tuple[float, ...]
    source_timestamp: float
    host_timestamp: float
    feedback: str
    derived_full: bool


@dataclass(frozen=True)
class StepResult:
    """Separate streams and current control report."""

    action: ActionRecord | None
    observation: ObservationRecord | None
    report: ControlReport


class OfflineController:
    """Single-hand bounded authority core; faults latch until a new instance.

    One local RLock serializes lifecycle/write admission; no other locks acquired.
    Retarget work runs outside it, then rechecks epoch and all freshness gates.
    """

    def __init__(
        self,
        config: OfflineConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.clock, self.wall_clock = clock, wall_clock
        self.descriptor = describe_model(config.model, config.side)
        self.interface = OfflineMockInterface(self.descriptor, clock=wall_clock)
        self._lock = threading.RLock()
        self.active_epoch: str | None = None
        self.fault: str | None = None
        self._prepared = self._closed = self._maintenance = False
        self._used: set[str] = set()
        self._start = self._watchdog = self._input = self._last_write = 0.0
        self._sequence = -1
        self._running = False
        self._source = -math.inf
        self._local: np.ndarray | None = None
        self._q = np.clip(
            np.zeros(len(self.descriptor.joint_names)),
            self.descriptor.lower,
            self.descriptor.upper,
        )

    def _report(self, outcome: str) -> ControlReport:
        return ControlReport(
            "fault" if self.fault else outcome,
            self.active_epoch,
            self.fault,
            "authority_revoked" if self.fault else "mock",
        )

    def _fail(self, reason: str) -> None:
        self.fault = self.fault or reason
        self.active_epoch = None
        self._running = False

    def prepare(self) -> ControlReport:
        """Connect mock and acquire exclusive session ownership."""
        with self._lock:
            if self._closed or self._maintenance or self.fault:
                return self._report("rejected")
            self.interface.connect()
            self._prepared = True
            return self._report("prepared")

    def activate(self, epoch_id: str, start_at: float) -> ControlReport:
        """Schedule one never-reused epoch in monotonic seconds."""
        with self._lock:
            if (
                not self._prepared
                or self._closed
                or self.fault
                or self.active_epoch is not None
                or not isinstance(epoch_id, str)
                or not epoch_id
                or epoch_id in self._used
                or isinstance(start_at, bool)
                or not math.isfinite(start_at)
                or start_at < self.clock()
            ):
                return self._report("rejected")
            self.active_epoch = epoch_id
            self._used.add(epoch_id)
            self._start = start_at
            self._watchdog = self.clock()
            self._sequence = -1
            self._running = False
            self._source = -math.inf
            self._local = None
            return self._report("scheduled")

    def watchdog(self, epoch_id: str, sequence: int) -> ControlReport:
        """Only a strictly progressing current-epoch heartbeat refreshes TTL."""
        with self._lock:
            self._check()
            if (
                epoch_id != self.active_epoch
                or type(sequence) is not int
                or sequence <= self._sequence
            ):
                return self._report("rejected")
            self._sequence = sequence
            self._watchdog = self.clock()
            return self._report("active" if self._running else "scheduled")

    def end(self, epoch_id: str) -> ControlReport:
        """Revoke matching authority; old end cannot affect a newer epoch."""
        with self._lock:
            if epoch_id == self.active_epoch:
                self.active_epoch = None
                self._running = False
                return self._report("ended")
            return self._report(
                "ended"
                if epoch_id in self._used and self.active_epoch is None
                else "rejected"
            )

    def shutdown(self) -> ControlReport:
        """Revoke all authority and disconnect the mock."""
        with self._lock:
            self.active_epoch = None
            self._prepared = False
            self._closed = True
            self.interface.disconnect()
            return self._report("shutdown")

    @contextmanager
    def maintenance(self) -> Iterator[None]:
        """Exclude maintenance and prepared/session ownership."""
        with self._lock:
            if self._prepared or self._maintenance or self._closed:
                raise RuntimeError("Session/maintenance ownership conflict")
            self._maintenance = True
        try:
            yield
        finally:
            with self._lock:
                self._maintenance = False

    def _check(self) -> None:
        now = self.clock()
        if self.active_epoch is None:
            return
        if now - self._watchdog > self.config.watchdog_timeout_s:
            self._fail("watchdog_expired")
        elif now >= self._start:
            if not self._running:
                self._running = True
                self._input = self._last_write = now
            elif now - self._input > self.config.input_timeout_s:
                self._fail("input_stale")

    def _observe(self) -> ObservationRecord | None:
        if not self._prepared or self._closed:
            return None
        try:
            state = self.interface.read()
            q = self.descriptor.vector(state.q)
            age = self.wall_clock() - state.stamp
            if not math.isfinite(age) or age < 0 or age > self.config.state_timeout_s:
                raise ValueError("state_stale")
            if (q < self.descriptor.lower).any() or (q > self.descriptor.upper).any():
                raise ValueError("state_bounds")
            return ObservationRecord(
                self.config.model,
                self.config.side,
                self.descriptor.joint_names,
                tuple(q),
                self.descriptor.full_joint_names,
                self.descriptor.expand(q),
                state.stamp,
                self.wall_clock(),
                "mock",
                self.config.model == "revo2",
            )
        except Exception as exc:
            self._fail(f"read_failed:{exc}")
            return None

    def step(
        self, hand: HandState | None = None, *, epoch_id: str | None = None
    ) -> StepResult:
        """Observe independently, solve fresh input, then fence before write."""
        with self._lock:
            self._check()
            observation = self._observe()
            admitted = (
                self.active_epoch is not None
                and self._running
                and epoch_id == self.active_epoch
                and not self.fault
            )
            token = self.active_epoch
            if not admitted or hand is None:
                return StepResult(
                    None,
                    observation,
                    self._report("active" if self._running else "prepared"),
                )
        started = self.clock()
        try:
            local = normalize_hand(hand, self.config)
            q = retarget_hand(hand, self.config)
            with self._lock:
                self._check()
                if self.active_epoch != token or self.fault or self._closed:
                    return StepResult(None, observation, self._report("rejected"))
                age = self.wall_clock() - hand.timestamp
                if (
                    age < 0
                    or age > self.config.input_timeout_s
                    or self.clock() - started > self.config.solve_budget_s
                ):
                    raise ValueError("input_stale_or_solve_deadline")
                if hand.timestamp <= self._source:
                    raise ValueError("input_timestamp_not_progressing")
                if (
                    self._local is not None
                    and np.max(np.linalg.norm(local - self._local, axis=1)) > 0.15
                ):
                    raise ValueError("input_discontinuity")
                state_age = self.wall_clock() - observation.source_timestamp
                if state_age < 0 or state_age > self.config.state_timeout_s:
                    raise ValueError("state_stale_after_solve")
                dt = max(0.0, self.clock() - self._last_write)
                delta = self.config.max_velocity * dt
                q = np.clip(0.5 * q + 0.5 * self._q, self._q - delta, self._q + delta)
                q = np.clip(q, self.descriptor.lower, self.descriptor.upper)
                self.interface.write(JointCommand(q=q.copy()))
                self._q = q.copy()
                self._input = self._last_write = self.clock()
                self._source, self._local = hand.timestamp, local.copy()
                action = ActionRecord(
                    self.config.model,
                    self.config.side,
                    self.descriptor.joint_names,
                    tuple(q),
                    hand.timestamp,
                    self.wall_clock(),
                    token,
                )
                return StepResult(action, observation, self._report("active"))
        except Exception as exc:
            with self._lock:
                if token == self.active_epoch:
                    self._fail(f"input_or_write_failed:{exc}")
                return StepResult(None, observation, self._report("rejected"))
