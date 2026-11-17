"""Validated configuration for the bounded offline control path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .model import describe_model


@dataclass(frozen=True)
class OfflineConfig:
    """One explicit synthetic/replay hand; hardware activation fails closed."""

    model: str = "revo2"
    side: str = "left"
    glove_id: int = 1
    root_id: int = 0
    joint_ids: tuple[int, ...] = tuple(range(25))
    mode: str = "mock"
    input_timeout_s: float = 0.25
    watchdog_timeout_s: float = 0.5
    state_timeout_s: float = 0.25
    max_velocity: float = 2.0
    solve_budget_s: float = 0.05

    def __post_init__(self) -> None:
        describe_model(self.model, self.side)
        if self.mode != "mock":
            raise ValueError(
                "Hardware unqualified: require model/firmware/SDK/transport "
                "identity, axis/unit/limit and stop/readback qualification"
            )
        if type(self.glove_id) is not int or self.glove_id < 0:
            raise ValueError("glove_id must be a nonnegative integer")
        ids = tuple(self.joint_ids)
        if (
            len(ids) != 25
            or len(set(ids)) != 25
            or any(type(i) is not int or i < 0 for i in ids)
            or type(self.root_id) is not int
            or ids[0] != self.root_id
        ):
            raise ValueError(
                "Require 25 unique ordered IDs: root, five 4/5-point fingers"
            )
        object.__setattr__(self, "joint_ids", ids)
        for name in (
            "input_timeout_s",
            "watchdog_timeout_s",
            "state_timeout_s",
            "max_velocity",
            "solve_budget_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OfflineConfig:
        """Reject unknown configuration fields at the integration boundary."""
        return cls(**data)
