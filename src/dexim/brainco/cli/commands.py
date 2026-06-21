"""Click commands for the dexim-brainco CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import rich_click as click
from dexim.cli.common import get_console, handle_cli_error, make_table, status_badge

from .config_utils import (
    create_config_yaml,
    default_serial_port,
    edit_config_yaml,
    get_config_yaml_path,
    list_named_configs,
    remove_config_yaml,
    resolve_config_path,
    resolve_device_config,
    scan_serial_ports,
)

_IMPORT_HINT = "[muted]Install dependencies with: pip install 'dexim-brainco[viz]'[/]"
_HARDWARE_HINT = (
    "[muted]Install hardware support with: pip install 'dexim-brainco[hardware]'[/]"
)
_DEFAULT_BAUD = 460800


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--config",
    "-c",
    default=None,
    help="YAML config file or named config. Overrides other options if provided.",
)
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@click.option(
    "--mode",
    type=click.Choice(["mock", "hw"]),
    default="hw",
    show_default=True,
    help="Interface mode.",
)
@click.option(
    "--hand",
    type=click.Choice(["left", "right"]),
    default="left",
    show_default=True,
    help="Hand side.",
)
@click.option(
    "--address",
    default=None,
    help="ZMQ address for Manus skeleton subscriber (default: tcp://localhost:5555).",
)
@click.option(
    "--device",
    default=None,
    help="Named device from config/dexim/devices.yaml.",
)
@click.option(
    "--port",
    default=None,
    help="RS-485 serial port (e.g. COM5).",
)
@click.option(
    "--baud",
    type=int,
    default=None,
    help="RS-485 baud rate (default: 460800).",
)
@click.option(
    "--slave-id",
    type=int,
    default=None,
    help="Modbus slave ID (default: 126=left, 127=right).",
)
@click.option(
    "--auto-detect",
    is_flag=True,
    default=False,
    help="Use BrainCo SDK auto-detection to find the device.",
)
@click.option(
    "--auto-start",
    is_flag=True,
    default=False,
    help="Activate teleoperation immediately without waiting for an orchestrator START command.",
)
@handle_cli_error
def run(
    config: str | None,
    config_dir: Path | None,
    mode: str,
    hand: str,
    address: str | None,
    device: str | None,
    port: str | None,
    baud: int | None,
    slave_id: int | None,
    auto_detect: bool,
    auto_start: bool,
) -> None:
    """Start the BrainCo Revo2 hand control node."""
    console = get_console()

    try:
        from dexim.brainco.interface.config import BrainCoRS485Config

        from dexim.brainco.node import (
            BrainCoConfig,
            BrainCoControlNode,
            BrainCoNodeConfig,
            SubscriberConfig,
            load_config,
        )
        from dexim.brainco.node.config import (
            BrainCoHwConfig,
            InterfaceConfig,
        )
    except ImportError as exc:
        console.print(f"[error]Missing dependency: {exc}[/]")
        console.print(_IMPORT_HINT)
        raise SystemExit(1) from exc

    if device and config:
        raise click.UsageError("--device and --config are mutually exclusive.")

    if device:
        cfg = load_config(str(resolve_device_config(device, _path_value(config_dir))))
    elif config:
        cfg = load_config(str(resolve_config_path(config, _path_value(config_dir))))
    else:
        if mode == "mock":
            interface = InterfaceConfig(mode="mock")
        elif mode == "hw":
            rs485 = BrainCoRS485Config(
                port=port,
                baud=baud or _DEFAULT_BAUD,
            )
            hw = BrainCoHwConfig(
                backend="modbus_rs485",
                rs485=rs485,
                slave_id=slave_id,
                auto_detect=auto_detect,
            )
            interface = InterfaceConfig(mode="hw", hw=hw)
        else:
            interface = InterfaceConfig(mode="mock")

        subscriber_address = address or SubscriberConfig().address
        cfg = BrainCoNodeConfig(
            subscriber=SubscriberConfig(address=subscriber_address),
            interface=interface,
            brainco=BrainCoConfig(side=hand),
        )

    node_id = f"brainco_{cfg.brainco.side}"

    console.print(
        f"[info]Starting BrainCo node in [key]{cfg.interface.mode}[/key] mode...[/]"
    )
    console.print(f"  Hand:    [key]{cfg.brainco.side}[/key]")
    console.print(f"  Address: [key]{cfg.subscriber.address}[/key]")
    hw = cfg.interface.get_hardware_config()
    if hw is not None and hw.backend == "modbus_rs485":
        rs485 = hw.rs485
        if rs485 and rs485.port:
            console.print(f"  Port:    [key]{rs485.port}[/key]")
        if rs485:
            console.print(f"  Baud:    [key]{rs485.baud}[/key]")
        if not hw.auto_detect:
            console.print(f"  Slave ID: [key]{hw.effective_slave_id(hand)}[/key]")
        if hw.auto_detect:
            console.print("  Detection: [key]auto[/key]")

    try:
        from dexim.core.messages import HandState, TopicBuilder
        from dexim.core.nodes import TopicSubscriber
    except ImportError as exc:
        console.print(f"[error]Missing dependency: {exc}[/]")
        console.print("[muted]Install dependencies with: pip install dexim-core[/]")
        raise SystemExit(1) from exc

    topic = TopicBuilder().observation.hand_state(cfg.subscriber.source_node_id)
    subscriber = TopicSubscriber(
        address=cfg.subscriber.address,
        topic=topic,
        msg_type=HandState,
        timeout_ms=cfg.subscriber.timeout_ms,
    )

    try:
        node = BrainCoControlNode(
            node_id=node_id,
            config=cfg,
            subscriber=subscriber,
        )
        if auto_start:
            node._teleop_active = True
            console.print("[info]Auto-start: teleoperation active immediately.[/]")
        console.print("[success]* BrainCo node is running. Press Ctrl+C to stop.[/]")
        node.run()
    except KeyboardInterrupt:
        console.print("\n[warning]Stopping.[/]")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["mock", "hw"]),
    default="hw",
    show_default=True,
    help="Interface mode.",
)
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@click.option(
    "--device",
    default=None,
    help="Named device from config/dexim/devices.yaml.",
)
@click.option(
    "--hand",
    type=click.Choice(["left", "right"]),
    default="left",
    show_default=True,
    help="Hand side for direct RS-485 reads.",
)
@click.option(
    "--port",
    default=None,
    help="RS-485 serial port.",
)
@click.option(
    "--baud",
    type=int,
    default=None,
    help="RS-485 baud rate.",
)
@click.option(
    "--slave-id",
    type=int,
    default=None,
    help="Modbus slave ID (default: 126=left, 127=right).",
)
@click.option(
    "--auto-detect",
    is_flag=True,
    default=False,
    help="Use BrainCo SDK auto-detection to find the device.",
)
@handle_cli_error
def status(
    mode: str,
    config_dir: Path | None,
    device: str | None,
    hand: str,
    port: str | None,
    baud: int | None,
    slave_id: int | None,
    auto_detect: bool,
) -> None:
    """Show BrainCo Revo2 hand connection status."""
    console = get_console()

    if mode == "mock":
        rows = [
            ["Mode", "mock"],
            ["Status", status_badge(True)],
            ["Interface", "MockInterface (no hardware required)"],
        ]
    else:
        if device:
            from dexim.brainco.node import load_config as _load_cfg

            _dev_cfg = _load_cfg(
                str(resolve_device_config(device, _path_value(config_dir)))
            )
            resolved_port, resolved_baud, resolved_hand, resolved_slave_id, resolved_auto_detect = _modbus_params_from_config(_dev_cfg)
        else:
            resolved_port = port or default_serial_port()
            resolved_baud = baud or _DEFAULT_BAUD
            resolved_hand = hand
            resolved_slave_id = slave_id or (126 if resolved_hand == "left" else 127)
            resolved_auto_detect = auto_detect

        console.print(
            "[info]Reading BrainCo Revo2 hand on "
            f"[key]{resolved_port}[/key] @ [key]{resolved_baud}[/key]...[/]"
        )
        state = _modbus_read_state(
            port=resolved_port if not resolved_auto_detect else None,
            baud=resolved_baud,
            hand_side=resolved_hand,
            slave_id=resolved_slave_id,
            auto_detect=resolved_auto_detect,
        )
        rows = [
            ["Mode", "hw"],
            ["Backend", "modbus_rs485"],
            ["Port", resolved_port or "(auto-detect)"],
            ["Baud", str(resolved_baud)],
            ["Slave ID", str(resolved_slave_id)],
            ["Hand", resolved_hand],
            ["Status", status_badge(state is not None)],
        ]
        if state is not None:
            rows.append(["Joint Count", str(len(state.q))])

    console.print(
        make_table(
            title="BrainCo Revo2 Hand Status",
            columns=[("Property", "key"), ("Value", "value")],
            rows=rows,
        )
    )


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@click.option(
    "--device",
    default=None,
    help="Named device from config/dexim/devices.yaml.",
)
@click.option(
    "--port",
    default=None,
    help="RS-485 serial port. If omitted, the CLI will auto-scan.",
)
@click.option("--baud", type=int, default=None, help="RS-485 baud rate.")
@click.option(
    "--hand",
    type=click.Choice(["left", "right"]),
    default="left",
    show_default=True,
    help="Hand side.",
)
@click.option(
    "--slave-id",
    type=int,
    default=None,
    help="Modbus slave ID (default: 126=left, 127=right).",
)
@click.option(
    "--auto-detect",
    is_flag=True,
    default=False,
    help="Use BrainCo SDK auto-detection to find the device.",
)
@handle_cli_error
def probe(
    config_dir: Path | None,
    device: str | None,
    port: str | None,
    baud: int | None,
    hand: str,
    slave_id: int | None,
    auto_detect: bool,
) -> None:
    """Connect once to hardware and print a joint snapshot."""
    console = get_console()
    for candidate_port, candidate_baud, candidate_hand, candidate_slave_id, candidate_auto_detect in _modbus_probe_candidates(
        config_dir=config_dir,
        device=device,
        port=port,
        baud=baud,
        hand=hand,
        slave_id=slave_id,
        auto_detect=auto_detect,
    ):
        console.print(
            "[info]Trying "
            f"[key]{candidate_port or '(auto-detect)'}[/key] @ [key]{candidate_baud}[/key] "
            f"slave_id=[key]{candidate_slave_id}[/key] for [key]{candidate_hand}[/key] hand...[/]"
        )
        state = _modbus_read_state(
            port=candidate_port if not candidate_auto_detect else None,
            baud=candidate_baud,
            hand_side=candidate_hand,
            slave_id=candidate_slave_id,
            auto_detect=candidate_auto_detect,
        )
        if state is not None:
            from .display import render_joint_state

            render_joint_state(
                state, console, title=f"Probe Result * {candidate_port or '(auto-detect)'}"
            )
            return

    raise click.ClickException(
        "Failed to connect to BrainCo hardware on any candidate serial port"
    )


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@click.option(
    "--device",
    default=None,
    help="Named device from config/dexim/devices.yaml.",
)
@click.option(
    "--port",
    default=None,
    help="RS-485 serial port. If omitted, the CLI will auto-scan.",
)
@click.option("--baud", type=int, default=None, help="RS-485 baud rate.")
@click.option(
    "--hand",
    type=click.Choice(["left", "right"]),
    default="left",
    show_default=True,
    help="Hand side.",
)
@click.option(
    "--slave-id",
    type=int,
    default=None,
    help="Modbus slave ID (default: 126=left, 127=right).",
)
@click.option(
    "--auto-detect",
    is_flag=True,
    default=False,
    help="Use BrainCo SDK auto-detection to find the device.",
)
@handle_cli_error
def check(
    config_dir: Path | None,
    device: str | None,
    port: str | None,
    baud: int | None,
    hand: str,
    slave_id: int | None,
    auto_detect: bool,
) -> None:
    """Run an ordered RS-485 hardware health checklist."""
    console = get_console()
    from .display import render_health_checks

    exit_code = 0
    discovered_ports = scan_serial_ports()
    rows: list[tuple[str, str, str]] = []
    if discovered_ports:
        rows.append(
            (
                "Serial ports discovered",
                "[success]PASS[/]",
                ", ".join(discovered_ports[:6]),
            )
        )
    else:
        rows.append(
            (
                "Serial ports discovered",
                "[warning]WARN[/]",
                "No ports discovered; using explicit/default target only",
            )
        )
        exit_code = 1

    candidates = _modbus_probe_candidates(
        config_dir=config_dir,
        device=device,
        port=port,
        baud=baud,
        hand=hand,
        slave_id=slave_id,
        auto_detect=auto_detect,
    )
    first_port, first_baud, first_hand, first_slave_id, first_auto_detect = candidates[0]
    rows.append(
        (
            "Target selection",
            "[success]PASS[/]",
            f"{first_port or '(auto-detect)'} @ {first_baud} slave_id={first_slave_id} ({first_hand})",
        )
    )

    connected_state = None
    connected_port = None
    for candidate_port, candidate_baud, candidate_hand, candidate_slave_id, candidate_auto_detect in candidates:
        connected_state = _modbus_read_state(
            port=candidate_port if not candidate_auto_detect else None,
            baud=candidate_baud,
            hand_side=candidate_hand,
            slave_id=candidate_slave_id,
            auto_detect=candidate_auto_detect,
        )
        if connected_state is not None:
            connected_port = candidate_port
            rows.append(
                (
                    "Hardware connect",
                    "[success]PASS[/]",
                    f"Connected on {candidate_port or '(auto-detect)'}",
                )
            )
            break

    if connected_state is None:
        rows.append(
            (
                "Hardware connect",
                "[error]FAIL[/]",
                "Unable to connect to any candidate serial port",
            )
        )
        render_health_checks(rows, console)
        raise SystemExit(2)

    if bool(np.isfinite(connected_state.q).all()) and len(connected_state.q) == 6:
        rows.append(
            (
                "Joint read",
                "[success]PASS[/]",
                f"Received 6 joints from {connected_port or '(auto-detect)'}",
            )
        )
    else:
        rows.append(
            (
                "Joint read",
                "[error]FAIL[/]",
                "Invalid joint payload returned by hardware",
            )
        )
        render_health_checks(rows, console)
        raise SystemExit(2)

    render_health_checks(rows, console)
    raise SystemExit(exit_code)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@click.group(name="config")
def config_group() -> None:
    """Manage named BrainCo configuration files (CRUD)."""


@config_group.command(name="show")
@click.argument("config_name")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@handle_cli_error
def config_show(config_name: str, config_dir: Path | None) -> None:
    """Pretty-print a resolved BrainCo configuration file."""
    console = get_console()
    try:
        from dexim.brainco.node import load_config
    except ImportError as exc:
        console.print(f"[error]Missing dependency: {exc}[/]")
        console.print(_IMPORT_HINT)
        raise SystemExit(1) from exc

    cfg = load_config(str(resolve_config_path(config_name, _path_value(config_dir))))
    from .display import render_config

    render_config(cfg, console)


@config_group.command(name="validate")
@click.argument("config_name")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@handle_cli_error
def config_validate(config_name: str, config_dir: Path | None) -> None:
    """Validate a BrainCo configuration file without starting the node."""
    console = get_console()
    try:
        from dexim.brainco.node import load_config
    except ImportError as exc:
        console.print(f"[error]Missing dependency: {exc}[/]")
        console.print(_IMPORT_HINT)
        raise SystemExit(1) from exc

    resolved = resolve_config_path(config_name, _path_value(config_dir))
    load_config(str(resolved))
    console.print(f"[success][OK] Config is valid:[/] {resolved}")


@config_group.command(name="list")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@handle_cli_error
def config_list(config_dir: Path | None) -> None:
    """List all named BrainCo configuration files."""
    console = get_console()
    names = list_named_configs(_path_value(config_dir))
    if not names:
        console.print("[muted]No configs found.[/]")
        return
    rows = [
        [name, str(get_config_yaml_path(name, _path_value(config_dir)))]
        for name in names
    ]
    console.print(
        make_table(
            title="Named BrainCo Configs",
            columns=[("Name", "key"), ("Path", "value")],
            rows=rows,
        )
    )


def _config_dir_option(func):
    return click.option(
        "--config-dir",
        type=click.Path(path_type=Path, file_okay=False),
        default=None,
        help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
    )(func)


def _config_field_options(func):
    """Shared CLI options for config new / config edit."""
    func = click.option(
        "--mode",
        type=click.Choice(["mock", "hw"]),
        default=None,
        help="Interface mode.",
    )(func)
    func = click.option("--port", default=None, help="RS-485 serial port.")(func)
    func = click.option("--baud", type=int, default=None, help="RS-485 baud rate.")(func)
    func = click.option(
        "--slave-id", type=int, default=None, help="Modbus slave ID (126=left, 127=right)."
    )(func)
    func = click.option(
        "--hand",
        type=click.Choice(["left", "right"]),
        default=None,
        help="Hand side.",
    )(func)
    func = click.option(
        "--auto-detect",
        is_flag=True,
        default=None,
        help="Use BrainCo SDK auto-detection.",
    )(func)
    func = click.option("--address", default=None, help="ZMQ subscriber address.")(func)
    func = click.option(
        "--rate-hz", type=float, default=None, help="Control loop rate in Hz."
    )(func)
    return func


def _build_config_dict(
    *,
    mode: str | None,
    port: str | None,
    baud: int | None,
    slave_id: int | None,
    hand: str | None,
    auto_detect: bool | None,
    address: str | None,
    rate_hz: float | None,
) -> dict:
    """Build a config dict from CLI option values, omitting None fields."""
    data: dict = {}

    # Interface section
    interface: dict = {}
    resolved_mode = mode or "mock"
    interface["mode"] = resolved_mode

    if resolved_mode == "hw":
        interface["backend"] = "modbus_rs485"
        rs485: dict = {}
        if port is not None:
            rs485["port"] = port
        if baud is not None:
            rs485["baud"] = baud
        if rs485:
            interface["rs485"] = rs485
        if slave_id is not None:
            interface["slave_id"] = slave_id
        if auto_detect is not None:
            interface["auto_detect"] = auto_detect

    data["interface"] = interface

    # BrainCo section
    if hand is not None:
        data.setdefault("brainco", {})["side"] = hand

    # Subscriber section
    if address is not None:
        data.setdefault("subscriber", {})["address"] = address

    # Control section
    if rate_hz is not None:
        data.setdefault("control", {})["rate_hz"] = rate_hz

    return data


def _collect_updates(
    *,
    mode: str | None,
    port: str | None,
    baud: int | None,
    slave_id: int | None,
    hand: str | None,
    auto_detect: bool | None,
    address: str | None,
    rate_hz: float | None,
) -> dict:
    """Build a partial update dict from non-None CLI option values."""
    updates: dict = {}

    if mode is not None:
        updates.setdefault("interface", {})["mode"] = mode
    if port is not None:
        updates.setdefault("interface", {}).setdefault("rs485", {})["port"] = port
    if baud is not None:
        updates.setdefault("interface", {}).setdefault("rs485", {})["baud"] = baud
    if slave_id is not None:
        updates.setdefault("interface", {})["slave_id"] = slave_id
    if auto_detect is not None:
        updates.setdefault("interface", {})["auto_detect"] = auto_detect

    if hand is not None:
        updates.setdefault("brainco", {})["side"] = hand

    if address is not None:
        updates.setdefault("subscriber", {})["address"] = address

    if rate_hz is not None:
        updates.setdefault("control", {})["rate_hz"] = rate_hz

    return updates


@config_group.command(name="new")
@click.argument("config_name")
@_config_field_options
@_config_dir_option
@handle_cli_error
def config_new(
    config_name: str,
    config_dir: Path | None,
    mode: str | None,
    port: str | None,
    baud: int | None,
    slave_id: int | None,
    hand: str | None,
    auto_detect: bool | None,
    address: str | None,
    rate_hz: float | None,
) -> None:
    """Create a new named BrainCo configuration file."""
    console = get_console()
    data = _build_config_dict(
        mode=mode,
        port=port,
        baud=baud,
        slave_id=slave_id,
        hand=hand,
        auto_detect=auto_detect,
        address=address,
        rate_hz=rate_hz,
    )
    path = create_config_yaml(config_name, data, _path_value(config_dir))
    console.print(f"[success][OK] Created config '{config_name}':[/] {path}")


@config_group.command(name="edit")
@click.argument("config_name")
@_config_field_options
@_config_dir_option
@handle_cli_error
def config_edit(
    config_name: str,
    config_dir: Path | None,
    mode: str | None,
    port: str | None,
    baud: int | None,
    slave_id: int | None,
    hand: str | None,
    auto_detect: bool | None,
    address: str | None,
    rate_hz: float | None,
) -> None:
    """Update fields in an existing named BrainCo configuration file."""
    console = get_console()
    updates = _collect_updates(
        mode=mode,
        port=port,
        baud=baud,
        slave_id=slave_id,
        hand=hand,
        auto_detect=auto_detect,
        address=address,
        rate_hz=rate_hz,
    )
    if not updates:
        raise click.UsageError("Specify at least one field to change.")
    path = edit_config_yaml(config_name, updates, _path_value(config_dir))
    console.print(f"[success][OK] Updated config '{config_name}':[/] {path}")


@config_group.command(name="remove")
@click.argument("config_name")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Confirm deletion without prompting.",
)
@_config_dir_option
@handle_cli_error
def config_remove(
    config_name: str,
    yes: bool,
    config_dir: Path | None,
) -> None:
    """Delete a named BrainCo configuration file."""
    console = get_console()
    if not yes:
        raise click.UsageError(
            f"This will permanently delete config '{config_name}'. "
            f"Pass --yes to confirm."
        )
    path = remove_config_yaml(config_name, _path_value(config_dir))
    console.print(f"[success][OK] Removed config '{config_name}':[/] {path}")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


@click.group(name="model")
def model_group() -> None:
    """Inspect and visualize the BrainCo Revo2 hand kinematic model."""


@model_group.command(name="info")
@click.option(
    "--hand",
    type=click.Choice(["left", "right"]),
    default="left",
    show_default=True,
    help="Hand side.",
)
@handle_cli_error
def model_info(hand: Literal["left", "right"]) -> None:
    """Show joint names, DOF, and limits for the BrainCo Revo2 hand model."""
    console = get_console()
    try:
        from dexim.brainco.model.factory import create_model
    except ImportError as exc:
        console.print(f"[error]Missing dependency: {exc}[/]")
        console.print(_IMPORT_HINT)
        raise SystemExit(1) from exc

    console.print(f"[info]Loading BrainCo Revo2 [key]{hand}[/key] hand model...[/]")
    model = create_model(hand_side=hand)
    from .display import render_model_info

    render_model_info(model, console)


@model_group.command(name="viz")
@click.option(
    "--hand",
    type=click.Choice(["left", "right"]),
    default="left",
    show_default=True,
    help="Hand side.",
)
@click.option(
    "--port",
    default=8080,
    show_default=True,
    help="Viser server port.",
)
@handle_cli_error
def model_viz(hand: Literal["left", "right"], port: int) -> None:
    """Launch the Viser 3D visualizer for the BrainCo Revo2 hand model."""
    console = get_console()
    try:
        from dexim.brainco.model.factory import create_model
        from dexim.brainco.viz import BrainCoRenderer
    except ImportError as exc:
        console.print(f"[error]Missing dependency: {exc}[/]")
        console.print(_IMPORT_HINT)
        raise SystemExit(1) from exc

    console.print(f"[info]Loading BrainCo Revo2 [key]{hand}[/key] hand model...[/]")
    model = create_model(hand_side=hand)

    console.print(f"[info]Starting Viser server on port [key]{port}[/key]...[/]")
    _viz = BrainCoRenderer(model=model, port=port)

    console.print(f"[success]* Viser running at [key]http://localhost:{port}[/key][/]")
    console.print("[muted]Open your browser to interact. Press Ctrl+C to stop.[/]")
    try:
        import time

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        console.print("\n[warning]Stopping.[/]")


# ---------------------------------------------------------------------------
# viz
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--node-id",
    default="brainco_left",
    show_default=True,
    help="Node identifier to subscribe to (matches the running node's node_id).",
)
@click.option(
    "--endpoint",
    default="tcp://localhost:5556",
    show_default=True,
    help="ZMQ PUB endpoint of the running node's data plane.",
)
@click.option(
    "--hand",
    type=click.Choice(["left", "right"]),
    default="left",
    show_default=True,
    help="Hand side (used to load the kinematic model).",
)
@click.option(
    "--port",
    default=8080,
    show_default=True,
    help="Viser web server port.",
)
@click.option(
    "--fps",
    type=float,
    default=60.0,
    show_default=True,
    help="Render loop frame rate; interpolates between received keyframes.",
)
@handle_cli_error
def viz(
    node_id: str,
    endpoint: str,
    hand: Literal["left", "right"],
    port: int,
    fps: float,
) -> None:
    """Live 3D visualizer \u2014 subscribes to a running node's data plane.

    Connects to the ZMQ PUB socket of a running ``dexim-brainco run`` node,
    receives joint commands and renders them in a Viser web UI.

    Start the node first, then run this command in a separate terminal::

        dexim-brainco viz --node-id brainco_left --endpoint tcp://localhost:5556
    """
    console = get_console()
    try:
        from dexim.brainco.model.factory import create_model
        from dexim.brainco.viz import BrainCoRenderer
        from dexim.brainco.viz.subscriber import VizSubscriber
    except ImportError as exc:
        console.print(f"[error]Missing dependency: {exc}[/]")
        console.print(_IMPORT_HINT)
        raise SystemExit(1) from exc

    console.print(f"[info]Loading BrainCo Revo2 [key]{hand}[/key] hand model\u2026[/]")
    model = create_model(hand_side=hand)

    console.print(f"[info]Starting Viser server on port [key]{port}[/key]\u2026[/]")
    renderer = BrainCoRenderer(model=model, port=port)

    console.print(
        f"[success]\u25cf Viser running at [key]http://localhost:{port}[/key][/]\n"
        f"[info]Subscribing to [key]{node_id}[/key] at [key]{endpoint}[/key][/]"
    )
    console.print("[muted]Open your browser to interact. Press Ctrl+C to stop.[/]")

    viewer = VizSubscriber(
        renderer=renderer, endpoint=endpoint, node_id=node_id, fps=fps
    )
    try:
        viewer.run()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    console.print("\n[warning]Stopping.[/]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_value(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _modbus_read_state(
    port: str | None,
    baud: int,
    hand_side: str,
    slave_id: int,
    auto_detect: bool = False,
):
    try:
        from dexim.brainco.interface.backends.modbus_rs485 import (
            BrainCoModbusRS485Backend,
        )
    except ImportError:
        console = get_console()
        console.print("[error]BrainCo hardware support is not installed.[/]")
        console.print(_HARDWARE_HINT)
        raise SystemExit(1)

    try:
        backend = BrainCoModbusRS485Backend(
            port=port,
            baud=baud,
            slave_id=slave_id,
            hand_side=hand_side,
            auto_detect=auto_detect,
            auto_calibrate=False,
        )
        backend.connect()
        try:
            return backend.read()
        finally:
            backend.disconnect()
    except Exception:
        return None


def _modbus_params_from_config(cfg) -> tuple[str | None, int, str, int, bool]:
    """Extract Modbus RS-485 connection params from a loaded BrainCoNodeConfig.

    Args:
        cfg: BrainCoNodeConfig instance.

    Returns:
        Tuple of (port, baud, hand, slave_id, auto_detect).

    Raises:
        click.ClickException: If the config is not in hw/modbus_rs485 mode.
    """
    hw = cfg.interface.get_hardware_config()
    if hw is None or hw.backend != "modbus_rs485":
        raise click.ClickException(
            f"Device config is in '{cfg.interface.mode}' mode, not 'hw/modbus_rs485'.\n"
            "This command only supports Modbus RS-485 connections."
        )
    rs485 = hw.rs485
    port = rs485.port if rs485 else None
    baud = rs485.baud if rs485 else _DEFAULT_BAUD
    return (
        port,
        baud,
        cfg.brainco.side,
        hw.effective_slave_id(cfg.brainco.side),
        hw.auto_detect,
    )


def _modbus_probe_candidates(
    *,
    config_dir: Path | None,
    device: str | None,
    port: str | None,
    baud: int | None,
    hand: str,
    slave_id: int | None,
    auto_detect: bool,
) -> list[tuple[str | None, int, str, int, bool]]:
    if device:
        from dexim.brainco.node import load_config as _load_cfg

        _dev_cfg = _load_cfg(
            str(resolve_device_config(device, _path_value(config_dir)))
        )
        return [_modbus_params_from_config(_dev_cfg)]

    if port:
        resolved_slave_id = slave_id or (126 if hand == "left" else 127)
        return [(port, baud or _DEFAULT_BAUD, hand, resolved_slave_id, auto_detect)]

    if auto_detect:
        return [(None, baud or _DEFAULT_BAUD, hand, slave_id or (126 if hand == "left" else 127), True)]

    discovered_ports = scan_serial_ports()
    preferred_port = default_serial_port()
    ordered_ports = [preferred_port] + [
        candidate for candidate in discovered_ports if candidate != preferred_port
    ]
    resolved_slave_id = slave_id or (126 if hand == "left" else 127)
    return [
        (candidate_port, baud or _DEFAULT_BAUD, hand, resolved_slave_id, False)
        for candidate_port in dict.fromkeys(ordered_ports)
    ]
