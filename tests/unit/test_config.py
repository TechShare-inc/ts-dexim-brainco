"""Unit tests for dexim.brainco.interface.config -- dataclass validation."""

from __future__ import annotations

import pytest

from dexim.brainco.interface.config import (
    DEFAULT_SLAVE_ID_LEFT,
    DEFAULT_SLAVE_ID_RIGHT,
    BrainCoHwConfig,
    BrainCoRS485Config,
    HardwareCoreConfig,
    InterfaceConfig,
)

# ---------------------------------------------------------------------------
# HardwareCoreConfig
# ---------------------------------------------------------------------------


class TestHardwareCoreConfig:
    def test_defaults(self):
        cfg = HardwareCoreConfig()
        assert cfg.rate_hz == 30.0
        assert cfg.shm_prefix == "brainco"
        assert cfg.carrot_lookahead_cycles == 0.0

    def test_custom_rate(self):
        cfg = HardwareCoreConfig(rate_hz=100.0)
        assert cfg.rate_hz == 100.0


# ---------------------------------------------------------------------------
# BrainCoRS485Config
# ---------------------------------------------------------------------------


class TestBrainCoRS485Config:
    def test_defaults(self):
        cfg = BrainCoRS485Config()
        assert cfg.port is None
        assert cfg.baud == 460800

    def test_explicit_port(self):
        cfg = BrainCoRS485Config(port="COM5")
        assert cfg.port == "COM5"


# ---------------------------------------------------------------------------
# BrainCoHwConfig
# ---------------------------------------------------------------------------


class TestBrainCoHwConfig:
    def test_defaults(self):
        cfg = BrainCoHwConfig()
        assert cfg.backend == "modbus_rs485"
        assert cfg.slave_id is None
        assert cfg.auto_detect is False
        assert cfg.auto_detect_quick is True
        assert cfg.auto_calibrate is False

    def test_requires_rs485_when_not_auto_detect(self):
        with pytest.raises(ValueError, match="rs485"):
            BrainCoHwConfig(auto_detect=False, rs485=None)

    def test_auto_detect_ok_without_rs485(self):
        cfg = BrainCoHwConfig(auto_detect=True)
        assert cfg.auto_detect is True

    def test_effective_slave_id_left_default(self):
        cfg = BrainCoHwConfig()
        assert cfg.effective_slave_id("left") == DEFAULT_SLAVE_ID_LEFT
        assert cfg.effective_slave_id() == DEFAULT_SLAVE_ID_LEFT

    def test_effective_slave_id_right_default(self):
        cfg = BrainCoHwConfig()
        assert cfg.effective_slave_id("right") == DEFAULT_SLAVE_ID_RIGHT

    def test_effective_slave_id_explicit_overrides_side(self):
        cfg = BrainCoHwConfig(slave_id=42)
        assert cfg.effective_slave_id("left") == 42
        assert cfg.effective_slave_id("right") == 42

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            BrainCoHwConfig(backend="canfd")


# ---------------------------------------------------------------------------
# InterfaceConfig
# ---------------------------------------------------------------------------


class TestInterfaceConfig:
    def test_default_is_mock(self):
        cfg = InterfaceConfig()
        assert cfg.mode == "mock"
        assert cfg.is_mock()
        assert not cfg.is_hw()

    def test_hw_mode_requires_hw_config(self):
        with pytest.raises(ValueError, match="hw"):
            InterfaceConfig(mode="hw", hw=None)

    def test_hw_mode_valid(self):
        hw = BrainCoHwConfig(auto_detect=True)
        cfg = InterfaceConfig(mode="hw", hw=hw)
        assert cfg.is_hw()
        assert not cfg.is_mock()
        assert cfg.get_hardware_config() is hw

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            InterfaceConfig(mode="invalid")

    def test_get_hardware_config_returns_none_for_mock(self):
        cfg = InterfaceConfig()
        assert cfg.get_hardware_config() is None

    def test_hw_mode_with_explicit_rs485(self):
        rs485 = BrainCoRS485Config(port="COM3", baud=115200)
        hw = BrainCoHwConfig(rs485=rs485, slave_id=126)
        cfg = InterfaceConfig(mode="hw", hw=hw)
        assert cfg.is_hw()
        hw_cfg = cfg.get_hardware_config()
        assert hw_cfg is not None
        assert hw_cfg.rs485 is not None
        assert hw_cfg.rs485.port == "COM3"
        assert hw_cfg.rs485.baud == 115200
        assert hw_cfg.effective_slave_id("left") == 126


# ---------------------------------------------------------------------------
# Default slave IDs
# ---------------------------------------------------------------------------


class TestDefaultSlaveIDs:
    def test_left_is_126(self):
        assert DEFAULT_SLAVE_ID_LEFT == 126
        assert DEFAULT_SLAVE_ID_LEFT == 0x7E

    def test_right_is_127(self):
        assert DEFAULT_SLAVE_ID_RIGHT == 127
        assert DEFAULT_SLAVE_ID_RIGHT == 0x7F
