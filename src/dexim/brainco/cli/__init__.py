"""
dexim.brainco.cli -- CLI entry point for BrainCo Revo2 device management.

Subcommands:
    config      Create / list / edit BrainCo config YAML files.
    devices     Manage the device registry (devices.yaml).
    joint       Joint-level commands (open, close, home, set, watch).
    probe       Quick device connectivity check.
    check       Full device health check.
    run         Launch the teleoperation control node.
"""

from __future__ import annotations


def standalone_app() -> None:
    """CLI entry point (placeholder).

    Registered as the ``dexim-brainco`` console script in pyproject.toml.
    """
    print("dexim-brainco CLI — not yet implemented")
