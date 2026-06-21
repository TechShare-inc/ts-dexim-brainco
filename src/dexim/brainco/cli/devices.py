"""Device registry and serial port discovery commands for dexim-brainco."""

from __future__ import annotations

from pathlib import Path

import rich_click as click
from dexim.cli.common import get_console, handle_cli_error, make_table

from .config_utils import (
    DeviceRegistryEntry,
    default_serial_port,
    get_config_yaml_path,
    get_devices_path,
    list_named_configs,
    load_device_registry,
    save_device_registry,
    scan_serial_ports,
)


@click.group(name="devices")
def devices_group() -> None:
    """Manage named BrainCo devices and discover serial ports."""


@devices_group.command(name="scan")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@handle_cli_error
def devices_scan(config_dir: Path | None) -> None:
    """Scan for serial ports on the current host."""
    console = get_console()
    default_port = default_serial_port()
    rows = [
        [port, "default" if port == default_port else ""]
        for port in scan_serial_ports()
    ] or [["(none found)", ""]]
    console.print(
        make_table(
            title="Discovered Serial Ports",
            columns=[("Port", "key"), ("Note", "value")],
            rows=rows,
        )
    )
    console.print(
        f"[muted]Registry path:[/] {get_devices_path(_path_value(config_dir))}"
    )


@devices_group.command(name="list")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@handle_cli_error
def devices_list(config_dir: Path | None) -> None:
    """List named BrainCo devices from the shared registry."""
    console = get_console()
    entries = load_device_registry(_path_value(config_dir))
    rows = [
        [entry.name, entry.package, entry.config] for entry in entries.values()
    ] or [["(empty)", "", ""]]
    console.print(
        make_table(
            title="Registered Devices",
            columns=[
                ("Name", "key"),
                ("Package", "value"),
                ("Config", "value"),
            ],
            rows=rows,
        )
    )
    console.print(
        f"[muted]Registry path:[/] {get_devices_path(_path_value(config_dir))}"
    )


@devices_group.command(name="add")
@click.argument("name")
@click.option(
    "--config",
    "config_name",
    required=True,
    help=(
        "Logical config name that this device alias points to. "
        "Must be an existing named config (see 'config list')."
    ),
)
@click.option(
    "--package",
    default="dexim-brainco",
    show_default=True,
    help="Owning package recorded in the registry.",
)
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@handle_cli_error
def devices_add(
    name: str,
    config_name: str,
    package: str,
    config_dir: Path | None,
) -> None:
    """Add or update a named device alias pointing to a config name."""
    console = get_console()
    dir_value = _path_value(config_dir)

    # Validate config name exists on disk
    config_path = get_config_yaml_path(config_name, dir_value)
    if not config_path.exists():
        available = list_named_configs(dir_value)
        hint = f"  Available configs: {', '.join(available)}\n" if available else ""
        raise click.ClickException(
            f"Config '{config_name}' does not exist: {config_path}\n"
            f"{hint}"
            f"Create it first with: dexim-brainco config new {config_name} ..."
        )

    entries = load_device_registry(dir_value)
    entries[name] = DeviceRegistryEntry(
        name=name,
        package=package,
        config=config_name,
    )
    path = save_device_registry(entries, dir_value)
    console.print(
        f"[success][OK] Saved device '{name}' -> config '{config_name}'[/] at {path}"
    )


@devices_group.command(name="show")
@click.argument("name")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@handle_cli_error
def devices_show(name: str, config_dir: Path | None) -> None:
    """Show details of a named device entry."""
    console = get_console()
    entries = load_device_registry(_path_value(config_dir))
    if name not in entries:
        raise click.ClickException(
            f"Device '{name}' not found in {get_devices_path(_path_value(config_dir))}"
        )
    entry = entries[name]
    config_path = get_config_yaml_path(entry.config, _path_value(config_dir))
    config_exists = config_path.exists()
    rows = [
        ["Name", entry.name],
        ["Package", entry.package],
        ["Config", entry.config],
        ["Config Path", str(config_path)],
        ["Config Exists", "[success][OK] yes[/]" if config_exists else "[error][FAIL] no[/]"],
    ]
    console.print(
        make_table(
            title=f"Device: {name}",
            columns=[("Property", "key"), ("Value", "value")],
            rows=rows,
        )
    )


@devices_group.command(name="remove")
@click.argument("name")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@handle_cli_error
def devices_remove(name: str, config_dir: Path | None) -> None:
    """Remove a named device entry."""
    console = get_console()
    entries = load_device_registry(_path_value(config_dir))
    if name not in entries:
        raise click.ClickException(
            f"Device '{name}' not found in {get_devices_path(_path_value(config_dir))}"
        )
    del entries[name]
    path = save_device_registry(entries, _path_value(config_dir))
    console.print(f"[success][OK] Removed device '{name}'[/] from {path}")


def _path_value(path: Path | None) -> str | None:
    return str(path) if path is not None else None
