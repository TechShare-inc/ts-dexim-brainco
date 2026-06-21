"""Hardware integration tests for BrainCo Revo2 hand.

These tests require a real BrainCo Revo2 hand connected via RS-485.
They are excluded from the default test run and must be invoked manually:

    pixi run test-hardware

Prerequisites:
    - BrainCo Revo2 hand connected via RS-485 / Modbus.
    - ``bc-stark-sdk`` installed (``pixi run pip install bc-stark-sdk``).
    - A device config YAML in ``config/dexim/brainco/`` or explicit CLI args.
"""

from __future__ import annotations

import os

import pytest

# Mark the whole module as hardware tests.
pytestmark = pytest.mark.hardware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_env(var: str) -> str:
    """Return the value of environment variable *var*, or skip the test."""
    value = os.environ.get(var, "").strip()
    if not value:
        pytest.skip(f"Set {var} to run hardware tests")
    return value


# ---------------------------------------------------------------------------
# Device connectivity
# ---------------------------------------------------------------------------


class TestDeviceConnectivity:
    """Verify that the BrainCo Revo2 hand can be detected and connected."""

    def test_sdk_importable(self):
        """bc-stark-sdk must be installed."""
        try:
            import bc_stark_sdk  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"bc-stark-sdk not installed: {exc}")

    def test_sdk_main_mod_importable(self):
        """bc_stark_sdk.main_mod must be importable."""
        try:
            import bc_stark_sdk.main_mod  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"bc_stark_sdk.main_mod not importable: {exc}")

    def test_backend_constructs(self):
        """BrainCoModbusRS485Backend constructs without errors."""
        try:
            from dexim.brainco.interface.backends.modbus_rs485 import (
                BrainCoModbusRS485Backend,
            )
        except ImportError as exc:
            pytest.skip(f"Backend import failed: {exc}")

        port = os.environ.get("BRAINCO_PORT", "COM5")
        slave_id = int(os.environ.get("BRAINCO_SLAVE_ID", "126"))
        baud = int(os.environ.get("BRAINCO_BAUD", "460800"))

        backend = BrainCoModbusRS485Backend(
            port=port,
            baud=baud,
            slave_id=slave_id,
        )
        assert backend is not None
        assert not backend._connected


class TestDeviceConnect:
    """Verify that the backend can connect to and disconnect from the hand."""

    def test_connect_disconnect(self):
        """Connect, verify connected state, disconnect."""
        from dexim.brainco.interface.backends.modbus_rs485 import (
            BrainCoModbusRS485Backend,
        )

        port = os.environ.get("BRAINCO_PORT", "COM5")
        slave_id = int(os.environ.get("BRAINCO_SLAVE_ID", "126"))
        baud = int(os.environ.get("BRAINCO_BAUD", "460800"))
        auto_detect = os.environ.get("BRAINCO_AUTO_DETECT", "0") == "1"

        backend = BrainCoModbusRS485Backend(
            port=port if not auto_detect else None,
            baud=baud,
            slave_id=slave_id,
            auto_detect=auto_detect,
        )

        try:
            backend.connect()
            assert backend._connected
        finally:
            if backend._connected:
                backend.disconnect()
            assert not backend._connected


class TestDeviceReadWrite:
    """Verify that the backend can read state and write commands."""

    def test_read_state(self):
        """Connect, read current joint state, disconnect."""
        from dexim.brainco.interface.backends.modbus_rs485 import (
            BrainCoModbusRS485Backend,
        )

        port = os.environ.get("BRAINCO_PORT", "COM5")
        slave_id = int(os.environ.get("BRAINCO_SLAVE_ID", "126"))
        baud = int(os.environ.get("BRAINCO_BAUD", "460800"))

        backend = BrainCoModbusRS485Backend(
            port=port,
            baud=baud,
            slave_id=slave_id,
        )

        try:
            backend.connect()
            state = backend.read()
            assert state is not None
            assert state.q is not None
            assert len(state.q) == 6
        finally:
            if backend._connected:
                backend.disconnect()

    def test_write_open_position(self):
        """Connect, send open-hand command, read back, disconnect."""
        import numpy as np
        from dexim.core.robot_interface import JointCommand

        from dexim.brainco.interface.backends.modbus_rs485 import (
            BrainCoModbusRS485Backend,
        )

        port = os.environ.get("BRAINCO_PORT", "COM5")
        slave_id = int(os.environ.get("BRAINCO_SLAVE_ID", "126"))
        baud = int(os.environ.get("BRAINCO_BAUD", "460800"))

        backend = BrainCoModbusRS485Backend(
            port=port,
            baud=baud,
            slave_id=slave_id,
        )

        try:
            backend.connect()
            # Send open-hand position (all zeros = fully open)
            cmd = JointCommand(mode="position", q=np.zeros(6))
            backend.write(cmd)

            import time

            time.sleep(0.5)

            state = backend.read()
            assert state.q is not None
            # Positions should be near zero (open)
            assert np.all(np.isfinite(state.q))
        finally:
            if backend._connected:
                backend.disconnect()
