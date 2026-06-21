"""Rich display helpers for the dexim-brainco CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from dexim.cli.common import make_table
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

if TYPE_CHECKING:
    from dexim.core.robot_interface import JointState

    from dexim.brainco.model.model import BrainCoModel
    from dexim.brainco.node.config import BrainCoNodeConfig

# Ordered joint labels matching BrainCo Revo2 active joints (thumb_flex, thumb_aux, index, middle, ring, pinky)
_BRAINCO_JOINT_NAMES = [
    "thumb_flex",
    "thumb_aux",
    "index",
    "middle",
    "ring",
    "pinky",
]


def render_config(cfg: BrainCoNodeConfig, console: Console) -> None:
    """Render a BrainCoNodeConfig as a Rich table.

    Args:
        cfg: Resolved BrainCoNodeConfig instance.
        console: Rich Console to print to.
    """
    console.print(Rule("BrainCo Configuration", style="brand.dim"))

    # Connection info depends on mode
    connection_rows: list[list[str]] = []
    mode = cfg.interface.mode
    hw = cfg.interface.get_hardware_config()
    if hw is not None and hw.backend == "modbus_rs485" and hw.rs485:
        connection_rows.append(["serial port", hw.rs485.port or "(auto-detect)"])
        connection_rows.append(["baud rate", str(hw.rs485.baud)])
        if not hw.auto_detect:
            connection_rows.append(
                ["slave_id", str(hw.effective_slave_id())]
            )

    rows = [
        ["mode", mode],
        ["hand (side)", cfg.brainco.side],
        *connection_rows,
        ["subscriber address", cfg.subscriber.address],
        ["control rate Hz", str(cfg.control.rate_hz)],
        ["data endpoint", cfg.data_endpoint],
    ]

    if hw is not None:
        rows.append(["auto_detect", str(hw.auto_detect)])
        rows.append(["auto_calibrate", str(hw.auto_calibrate)])

    if cfg.brainco.filter:
        rows.append(["filter type", cfg.brainco.filter.type])

    console.print(
        make_table(
            title="",
            columns=[("Key", "key"), ("Value", "value")],
            rows=rows,
        )
    )


def render_model_info(model: BrainCoModel, console: Console) -> None:
    """Render BrainCoModel joint information as Rich tables.

    Args:
        model: Initialized BrainCoModel instance.
        console: Rich Console to print to.
    """
    console.print(Rule("BrainCo Revo2 Hand Model", style="brand.dim"))

    summary_rows = [
        ["DOF (active joints)", str(model.nq)],
        [
            "Tip frames",
            ", ".join(model.tip_frame_names) if model.tip_frame_names else "--",
        ],
    ]
    console.print(
        make_table(
            title="Model Summary",
            columns=[("Property", "key"), ("Value", "value")],
            rows=summary_rows,
        )
    )

    lower_rad = model.lower_joint_limits
    upper_rad = model.upper_joint_limits
    lower_deg = np.rad2deg(lower_rad)
    upper_deg = np.rad2deg(upper_rad)

    joint_count = len(lower_rad)
    if joint_count <= len(_BRAINCO_JOINT_NAMES):
        names = _BRAINCO_JOINT_NAMES[:joint_count]
    else:
        names = _BRAINCO_JOINT_NAMES + [
            f"joint_{i}" for i in range(len(_BRAINCO_JOINT_NAMES), joint_count)
        ]

    joint_rows = [
        [name, f"{lo_r:.3f}", f"{hi_r:.3f}", f"{lo_d:.1f}", f"{hi_d:.1f}"]
        for name, lo_r, hi_r, lo_d, hi_d in zip(
            names, lower_rad, upper_rad, lower_deg, upper_deg
        )
    ]
    console.print(
        make_table(
            title="Joint Limits",
            columns=[
                ("Joint", "key"),
                ("Lower (rad)", "value"),
                ("Upper (rad)", "value"),
                ("Lower ( deg)", "value"),
                ("Upper ( deg)", "value"),
            ],
            rows=joint_rows,
        )
    )


def make_joint_state_table(state: JointState) -> Table:
    """Build a joint-state table for snapshots and live views."""
    joint_rows = [
        [name, f"{value:.3f}", f"{np.rad2deg(value):.1f}"]
        for name, value in zip(_BRAINCO_JOINT_NAMES, state.q)
    ]
    return make_table(
        title="Joint Snapshot",
        columns=[("Joint", "key"), ("Radians", "value"), ("Degrees", "value")],
        rows=joint_rows,
    )


def render_joint_state(state: JointState, console: Console, title: str) -> None:
    """Render a single joint-state snapshot."""
    console.print(Rule(title, style="brand.dim"))
    console.print(make_joint_state_table(state))
    console.print(f"[muted]Timestamp:[/] {state.stamp:.3f}")


def render_health_checks(
    rows: list[tuple[str, str, str]],
    console: Console,
    title: str = "Health Check",
) -> None:
    """Render ordered health-check results."""
    console.print(Rule(title, style="brand.dim"))
    console.print(
        make_table(
            title="",
            columns=[("Check", "key"), ("Status", "value"), ("Details", "value")],
            rows=[[name, status, details] for name, status, details in rows],
        )
    )


def make_watch_panel(state: JointState, hz: float, port: str) -> Panel:
    """Build the live panel used by the direct hardware joint watch command."""
    header = f"{port} @ {hz:.1f} Hz"
    return Panel(make_joint_state_table(state), title=f"Joint Watch * {header}")
