"""Shared config and device registry helpers for the dexim-brainco CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEXIM_CONFIG_DIR_ENV = "DEXIM_CONFIG_DIR"
_DEFAULT_WINDOWS_PORT = "COM5"
_DEFAULT_POSIX_PORT = "/dev/ttyUSB0"
_DEFAULT_BAUD = 460800


@dataclass(frozen=True)
class DeviceRegistryEntry:
    """Named device entry stored in config/dexim/devices.yaml.

    A device entry maps a short alias to a logical config name.
    The config name is resolved to ``{config_dir}/dexim/brainco/{config}.yaml``
    via the standard config resolution path.
    """

    name: str
    package: str
    config: str  # logical config name (e.g. "lab-modbus"), NOT a file path

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> DeviceRegistryEntry:
        """Build a typed entry from a YAML mapping.

        Args:
            name: Device name (key in devices.yaml).
            data: Raw YAML mapping for this entry.

        Returns:
            A validated DeviceRegistryEntry.

        Raises:
            ValueError: If the entry uses a legacy format.
        """
        old_fields = {"port", "baud", "hand"} & data.keys()
        if old_fields:
            raise ValueError(
                f"Device '{name}' uses the old inline-params format "
                f"(found fields: {', '.join(sorted(old_fields))}).\n"
                f"Create a config first, then re-register:\n"
                f"  dexim-brainco config new <config-name> --mode hw ...\n"
                f"  dexim-brainco devices add {name} --config <config-name>"
            )

        package = str(data.get("package", "")).strip() or "dexim-brainco"
        config = str(data.get("config", "")).strip()

        if not config:
            raise ValueError(
                f"Device '{name}' is missing the required 'config' field.\n"
                f"Re-register it with:\n"
                f"  dexim-brainco devices add {name} --config <config-name>"
            )

        _validate_logical_config_name(config, device_name=name)
        return cls(name=name, package=package, config=config)

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry to YAML-friendly mapping."""
        return {
            "package": self.package,
            "config": self.config,
        }


def _validate_logical_config_name(config: str, device_name: str = "") -> None:
    """Ensure *config* is a logical name, not a file path.

    Raises:
        ValueError: If the value looks like a path (contains ``/``, ``\\``
            or ends with ``.yaml``/``.yml``).
    """
    if "/" in config or "\\" in config or config.endswith((".yaml", ".yml")):
        ctx = f" for device '{device_name}'" if device_name else ""
        raise ValueError(
            f"Config value '{config}'{ctx} looks like a file path.\n"
            f"Use a logical config name instead (e.g. 'lab-modbus').\n"
            f"Create the config first with:\n"
            f"  dexim-brainco config new <config-name> ..."
        )


def default_serial_port() -> str:
    """Return the platform-default RS-485 serial port."""
    return _DEFAULT_WINDOWS_PORT if os.name == "nt" else _DEFAULT_POSIX_PORT


def get_config_dir(config_dir: str | None = None) -> Path:
    """Resolve the CLI config root directory."""
    raw_path = config_dir or os.environ.get(DEXIM_CONFIG_DIR_ENV) or "./config"
    return Path(raw_path).expanduser().resolve()


def get_brainco_config_dir(config_dir: str | None = None) -> Path:
    """Return the directory that stores named BrainCo configs."""
    return get_config_dir(config_dir) / "dexim" / "brainco"


def get_devices_path(config_dir: str | None = None) -> Path:
    """Return the shared device registry path."""
    return get_config_dir(config_dir) / "dexim" / "devices.yaml"


def resolve_config_path(name_or_path: str, config_dir: str | None = None) -> Path:
    """Resolve a config name or file path into an actual YAML path.

    Resolution order:
    1. Existing explicit path.
    2. {config_dir}/dexim/brainco/{name}.yaml
    3. {config_dir}/dexim/{name}.yaml
    """
    candidate = Path(name_or_path).expanduser()
    if candidate.suffix in {".yaml", ".yml"} or candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"Config file not found: {resolved}")

    config_root = get_config_dir(config_dir)
    search_paths = [
        config_root / "dexim" / "brainco" / f"{name_or_path}.yaml",
        config_root / "dexim" / f"{name_or_path}.yaml",
    ]
    for path in search_paths:
        if path.exists():
            return path.resolve()

    searched = ", ".join(str(path) for path in search_paths)
    raise FileNotFoundError(f"Config '{name_or_path}' not found. Searched: {searched}")


def load_device_registry(
    config_dir: str | None = None,
) -> dict[str, DeviceRegistryEntry]:
    """Load the shared device registry file if it exists."""
    path = get_devices_path(config_dir)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Device registry must be a mapping: {path}")

    return {
        name: DeviceRegistryEntry.from_dict(name, entry)
        for name, entry in sorted(data.items())
    }


def save_device_registry(
    entries: dict[str, DeviceRegistryEntry],
    config_dir: str | None = None,
) -> Path:
    """Persist the shared device registry to disk."""
    path = get_devices_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: entry.to_dict()
        for name, entry in sorted(entries.items(), key=lambda item: item[0])
    }
    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False)
    return path


def get_device_entry(name: str, config_dir: str | None = None) -> DeviceRegistryEntry:
    """Look up a named device in the shared registry."""
    entries = load_device_registry(config_dir)
    try:
        return entries[name]
    except KeyError as exc:
        raise KeyError(
            f"Device '{name}' not found in {get_devices_path(config_dir)}"
        ) from exc


def resolve_device_config(name: str, config_dir: str | None = None) -> Path:
    """Resolve a named device to its YAML config path.

    The device entry stores a logical config name which is resolved to
    ``{config_dir}/dexim/brainco/{config_name}.yaml``.

    Args:
        name: Device name in the shared registry.
        config_dir: Config root directory override.

    Returns:
        Absolute path to the device's YAML config file.

    Raises:
        KeyError: If the device is not found in the registry.
        FileNotFoundError: If the resolved config file does not exist.
    """
    entry = get_device_entry(name, config_dir)
    brainco_dir = get_brainco_config_dir(config_dir)
    resolved = (brainco_dir / f"{entry.config}.yaml").resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"Device '{name}' references config '{entry.config}' "
            f"but the file does not exist: {resolved}\n"
            f"Create the config first:\n"
            f"  dexim-brainco config new {entry.config} ...\n"
            f"Or update the device entry:\n"
            f"  dexim-brainco devices add {name} --config <existing-config-name>"
        )
    return resolved


# ---------------------------------------------------------------------------
# Config CRUD helpers
# ---------------------------------------------------------------------------


def list_named_configs(config_dir: str | None = None) -> list[str]:
    """Return sorted logical names of all BrainCo config YAMLs on disk.

    Scans ``{config_dir}/dexim/brainco/*.yaml`` and returns stems.
    """
    brainco_dir = get_brainco_config_dir(config_dir)
    if not brainco_dir.is_dir():
        return []
    return sorted(p.stem for p in brainco_dir.glob("*.yaml"))


def get_config_yaml_path(name: str, config_dir: str | None = None) -> Path:
    """Return the canonical path for a named BrainCo config."""
    return get_brainco_config_dir(config_dir) / f"{name}.yaml"


def create_config_yaml(
    name: str,
    data: dict[str, Any],
    config_dir: str | None = None,
) -> Path:
    """Create a new named BrainCo config YAML.

    Args:
        name: Logical config name (e.g. ``lab-modbus``).
        data: Config dict to serialize.
        config_dir: Config root directory override.

    Returns:
        Path to the created YAML file.

    Raises:
        FileExistsError: If a config with this name already exists.
    """
    path = get_config_yaml_path(name, config_dir)
    if path.exists():
        raise FileExistsError(
            f"Config '{name}' already exists: {path}\n"
            f"Use 'config edit {name}' to modify it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    return path


def edit_config_yaml(
    name: str,
    updates: dict[str, Any],
    config_dir: str | None = None,
) -> Path:
    """Patch fields in an existing named BrainCo config YAML.

    Performs a shallow merge: top-level keys in *updates* overwrite
    corresponding keys in the existing file.  Nested dicts are merged
    one level deep.

    Args:
        name: Logical config name.
        updates: Mapping of fields to set/overwrite.
        config_dir: Config root directory override.

    Returns:
        Path to the updated YAML file.

    Raises:
        FileNotFoundError: If the config does not exist.
    """
    path = get_config_yaml_path(name, config_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Config '{name}' not found: {path}\n"
            f"Use 'config new {name}' to create it first."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value

    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    return path


def remove_config_yaml(name: str, config_dir: str | None = None) -> Path:
    """Delete a named BrainCo config YAML from disk.

    Args:
        name: Logical config name.
        config_dir: Config root directory override.

    Returns:
        Path that was deleted.

    Raises:
        FileNotFoundError: If the config does not exist.
    """
    path = get_config_yaml_path(name, config_dir)
    if not path.exists():
        raise FileNotFoundError(f"Config '{name}' not found: {path}")
    path.unlink()
    return path


# ---------------------------------------------------------------------------
# Serial port scanning
# ---------------------------------------------------------------------------


def scan_serial_ports() -> list[str]:
    """Return discovered serial ports with an OS-aware fallback."""
    try:
        from serial.tools import list_ports  # type: ignore[import-not-found]

        ports = sorted(port.device for port in list_ports.comports())
        if ports:
            return ports
    except ImportError:
        pass

    if os.name == "nt":
        return [f"COM{i}" for i in range(1, 33)]

    patterns = ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/tty.usbserial*")
    seen: set[str] = set()
    results: list[str] = []
    for pattern in patterns:
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            value = str(path)
            if value not in seen:
                seen.add(value)
                results.append(value)
    return results
