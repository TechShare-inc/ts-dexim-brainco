"""Direct hardware joint commands for dexim-brainco."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rich_click as click
from dexim.cli.common import get_console, handle_cli_error
from dexim.core.robot_interface import JointCommand
from rich.live import Live

from dexim.brainco.interface._conversion import JOINT_LIMITS

from .config_utils import default_serial_port, resolve_device_config
from .display import make_watch_panel, render_joint_state

_WATCH_DEFAULT_HZ = 10.0
_DEFAULT_BAUD = 460800


@click.group(name="joint")
def joint_group() -> None:
    """Send one-shot joint commands or watch live hardware state."""


def _hardware_target_options(func: callable) -> callable:
    func = click.option(
        "--config-dir",
        type=click.Path(path_type=Path, file_okay=False),
        default=None,
        help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
    )(func)
    func = click.option(
        "--device",
        default=None,
        help="Named device from config/dexim/devices.yaml.",
    )(func)
    func = click.option(
        "--port",
        default=None,
        help="Serial port name, e.g. COM5 or /dev/ttyUSB0.",
    )(func)
    func = click.option("--baud", type=int, default=None, help="RS-485 baud rate.")(
        func
    )
    func = click.option(
        "--hand",
        type=click.Choice(["left", "right"]),
        default=None,
        help="Hand side. Defaults to the device registry entry or left.",
    )(func)
    func = click.option(
        "--slave-id",
        type=int,
        default=None,
        help="Modbus slave ID (default: 126=left, 127=right).",
    )(func)
    func = click.option(
        "--auto-detect",
        is_flag=True,
        default=False,
        help="Use BrainCo SDK auto-detection to find the device.",
    )(func)
    return func


def _open_interface(
    *,
    device: str | None,
    port: str | None,
    baud: int | None,
    hand: str | None,
    slave_id: int | None,
    auto_detect: bool,
    config_dir: Path | None,
):
    try:
        from dexim.brainco.interface.backends.modbus_rs485 import (
            BrainCoModbusRS485Backend,
        )
    except ImportError as exc:
        raise click.ClickException(
            "BrainCo hardware support is not installed. "
            "Use: pip install 'dexim-brainco[hardware]'"
        ) from exc

    if device:
        from dexim.brainco.node import load_config as _load_cfg

        _dev_cfg = _load_cfg(
            str(resolve_device_config(device, _path_value(config_dir)))
        )
        _hw = _dev_cfg.interface.get_hardware_config()
        if _hw is None or _hw.backend != "modbus_rs485":
            raise click.ClickException(
                f"Device config is in '{_dev_cfg.interface.mode}' mode, not 'hw/modbus_rs485'.\n"
                "The joint commands only support Modbus RS-485 connections."
            )
        rs485 = _hw.rs485
        resolved_port = rs485.port if rs485 and rs485.port else None
        resolved_baud = rs485.baud if rs485 else _DEFAULT_BAUD
        resolved_hand = _dev_cfg.brainco.side
        resolved_slave_id = _hw.effective_slave_id(resolved_hand)
        resolved_auto_detect = _hw.auto_detect
    else:
        resolved_port = port
        resolved_baud = baud or _DEFAULT_BAUD
        resolved_hand = hand or "left"
        resolved_slave_id = slave_id or (126 if resolved_hand == "left" else 127)
        resolved_auto_detect = auto_detect

    interface = BrainCoModbusRS485Backend(
        port=resolved_port,
        baud=resolved_baud,
        slave_id=resolved_slave_id,
        hand_side=resolved_hand,
        auto_detect=resolved_auto_detect,
        auto_calibrate=False,
    )
    return interface, resolved_port, resolved_baud, resolved_hand


def _limits() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([lo for lo, _ in JOINT_LIMITS], dtype=float)
    upper = np.array([hi for _, hi in JOINT_LIMITS], dtype=float)
    return lower, upper


def _command_and_render(
    *,
    label: str,
    target_q: np.ndarray,
    wait_s: float,
    device: str | None,
    port: str | None,
    baud: int | None,
    hand: str | None,
    slave_id: int | None,
    auto_detect: bool,
    config_dir: Path | None,
) -> None:
    console = get_console()
    interface, resolved_port, resolved_baud, resolved_hand = _open_interface(
        device=device,
        port=port,
        baud=baud,
        hand=hand,
        slave_id=slave_id,
        auto_detect=auto_detect,
        config_dir=config_dir,
    )
    console.print(
        f"[info]Sending {label} command to [key]{resolved_hand}[/key] hand on "
        f"[key]{resolved_port or '(auto-detect)'}[/key] @ [key]{resolved_baud}[/key]...[/]"
    )
    interface.connect()
    try:
        interface.write(JointCommand(q=target_q, mode="position"))
        if wait_s > 0:
            time.sleep(wait_s)
        state = interface.read()
    finally:
        interface.disconnect()
    render_joint_state(state, console, title=f"{label.title()} Snapshot")


@joint_group.command(name="open")
@_hardware_target_options
@click.option(
    "--wait",
    "wait_s",
    type=float,
    default=0.8,
    show_default=True,
    help="Seconds to wait before reading back state.",
)
@handle_cli_error
def joint_open(
    device: str | None,
    port: str | None,
    baud: int | None,
    hand: str | None,
    slave_id: int | None,
    auto_detect: bool,
    config_dir: Path | None,
    wait_s: float,
) -> None:
    """Open the hand to its lower joint limits."""
    lower, _upper = _limits()
    _command_and_render(
        label="open",
        target_q=lower,
        wait_s=wait_s,
        device=device,
        port=port,
        baud=baud,
        hand=hand,
        slave_id=slave_id,
        auto_detect=auto_detect,
        config_dir=config_dir,
    )


@joint_group.command(name="close")
@_hardware_target_options
@click.option(
    "--wait",
    "wait_s",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds to wait before reading back state.",
)
@handle_cli_error
def joint_close(
    device: str | None,
    port: str | None,
    baud: int | None,
    hand: str | None,
    slave_id: int | None,
    auto_detect: bool,
    config_dir: Path | None,
    wait_s: float,
) -> None:
    """Close the hand to its upper joint limits."""
    _lower, upper = _limits()
    _command_and_render(
        label="close",
        target_q=upper,
        wait_s=wait_s,
        device=device,
        port=port,
        baud=baud,
        hand=hand,
        slave_id=slave_id,
        auto_detect=auto_detect,
        config_dir=config_dir,
    )


@joint_group.command(name="home")
@_hardware_target_options
@click.option(
    "--wait",
    "wait_s",
    type=float,
    default=0.8,
    show_default=True,
    help="Seconds to wait before reading back state.",
)
@handle_cli_error
def joint_home(
    device: str | None,
    port: str | None,
    baud: int | None,
    hand: str | None,
    slave_id: int | None,
    auto_detect: bool,
    config_dir: Path | None,
    wait_s: float,
) -> None:
    """Move the hand to its zero/open home pose."""
    lower, _upper = _limits()
    _command_and_render(
        label="home",
        target_q=lower,
        wait_s=wait_s,
        device=device,
        port=port,
        baud=baud,
        hand=hand,
        slave_id=slave_id,
        auto_detect=auto_detect,
        config_dir=config_dir,
    )


@joint_group.command(name="set")
@click.argument("positions", nargs=6, type=float)
@_hardware_target_options
@click.option(
    "--wait",
    "wait_s",
    type=float,
    default=0.8,
    show_default=True,
    help="Seconds to wait before reading back state.",
)
@handle_cli_error
def joint_set(
    positions: tuple[float, float, float, float, float, float],
    device: str | None,
    port: str | None,
    baud: int | None,
    hand: str | None,
    slave_id: int | None,
    auto_detect: bool,
    config_dir: Path | None,
    wait_s: float,
) -> None:
    """Send a direct six-joint position command in radians."""
    _command_and_render(
        label="set",
        target_q=np.asarray(positions, dtype=float),
        wait_s=wait_s,
        device=device,
        port=port,
        baud=baud,
        hand=hand,
        slave_id=slave_id,
        auto_detect=auto_detect,
        config_dir=config_dir,
    )


@joint_group.command(name="watch")
@_hardware_target_options
@click.option(
    "--hz",
    type=float,
    default=_WATCH_DEFAULT_HZ,
    show_default=True,
    help="Polling rate in Hertz.",
)
@click.option(
    "--samples",
    type=int,
    default=0,
    show_default=True,
    help="Number of samples to read. Zero means run until Ctrl+C.",
)
@handle_cli_error
def joint_watch(
    device: str | None,
    port: str | None,
    baud: int | None,
    hand: str | None,
    slave_id: int | None,
    auto_detect: bool,
    config_dir: Path | None,
    hz: float,
    samples: int,
) -> None:
    """Poll hardware joint state directly and render a live table."""
    console = get_console()
    if hz <= 0:
        raise click.ClickException("--hz must be > 0")

    interface, resolved_port, _resolved_baud, _resolved_hand = _open_interface(
        device=device,
        port=port,
        baud=baud,
        hand=hand,
        slave_id=slave_id,
        auto_detect=auto_detect,
        config_dir=config_dir,
    )
    delay_s = 1.0 / hz
    sample_count = 0

    interface.connect()
    try:
        state = interface.read()
        with Live(
            make_watch_panel(state, hz, resolved_port or "(auto-detect)"),
            console=console,
            refresh_per_second=max(1, int(hz)),
        ) as live:
            try:
                while samples == 0 or sample_count < samples:
                    state = interface.read()
                    live.update(
                        make_watch_panel(
                            state, hz, resolved_port or "(auto-detect)"
                        )
                    )
                    sample_count += 1
                    time.sleep(delay_s)
            except KeyboardInterrupt:
                console.print("\n[warning]Stopping joint watch.[/]")
    finally:
        interface.disconnect()


def _path_value(path: Path | None) -> str | None:
    return str(path) if path is not None else None
