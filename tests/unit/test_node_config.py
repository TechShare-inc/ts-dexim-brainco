"""Unit tests for dexim.brainco.node.config -- dataclass validation and YAML loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from dexim.brainco.node.config import (
    BrainCoConfig,
    BrainCoNodeConfig,
    FeatureExtractionConfig,
    FilterConfig,
    SubscriberConfig,
    VizConfig,
    load_config,
)

# ---------------------------------------------------------------------------
# FeatureExtractionConfig
# ---------------------------------------------------------------------------


class TestFeatureExtractionConfig:
    def test_defaults(self):
        cfg = FeatureExtractionConfig()
        assert cfg.src_indices == [1, 6, 11, 16, 21]
        assert cfg.dst_indices == [4, 9, 14, 19, 24]
        assert cfg.apply_rotation is True

    def test_custom_indices(self):
        cfg = FeatureExtractionConfig(
            src_indices=[1, 2, 3],
            dst_indices=[4, 5, 6],
        )
        assert len(cfg.src_indices) == 3

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            FeatureExtractionConfig(
                src_indices=[1, 2],
                dst_indices=[3],
            )

    def test_empty_src_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            FeatureExtractionConfig(src_indices=[], dst_indices=[])


# ---------------------------------------------------------------------------
# FilterConfig
# ---------------------------------------------------------------------------


class TestFilterConfig:
    def test_default_is_wma(self):
        cfg = FilterConfig()
        assert cfg.type == "wma"
        assert cfg.weights == [0.4, 0.3, 0.2, 0.1]

    def test_wma_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            FilterConfig(type="wma", weights=[0.5, 0.5, 0.5])

    def test_wma_weights_not_empty(self):
        with pytest.raises(ValueError, match="non-empty list"):
            FilterConfig(type="wma", weights=[])

    def test_ema_validates_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            FilterConfig(type="ema", alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            FilterConfig(type="ema", alpha=1.5)

    def test_ema_valid(self):
        cfg = FilterConfig(type="ema", alpha=0.5)
        assert cfg.alpha == 0.5

    def test_one_euro_validates_freq(self):
        with pytest.raises(ValueError, match="freq"):
            FilterConfig(type="one_euro", freq=0)

    def test_one_euro_validates_min_cutoff(self):
        with pytest.raises(ValueError, match="min_cutoff"):
            FilterConfig(type="one_euro", min_cutoff=0)

    def test_one_euro_validates_beta(self):
        with pytest.raises(ValueError, match="beta"):
            FilterConfig(type="one_euro", beta=-1)

    def test_one_euro_validates_d_cutoff(self):
        with pytest.raises(ValueError, match="d_cutoff"):
            FilterConfig(type="one_euro", d_cutoff=0)

    def test_one_euro_valid(self):
        cfg = FilterConfig(
            type="one_euro", freq=60.0, min_cutoff=2.0, beta=0.01, d_cutoff=2.0
        )
        assert cfg.freq == 60.0

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="type must be one of"):
            FilterConfig(type="unknown")


# ---------------------------------------------------------------------------
# BrainCoConfig
# ---------------------------------------------------------------------------


class TestBrainCoConfig:
    def test_defaults(self):
        cfg = BrainCoConfig()
        assert cfg.side == "left"
        assert isinstance(cfg.feature_extraction, FeatureExtractionConfig)
        assert cfg.filter is None

    def test_right_side(self):
        cfg = BrainCoConfig(side="right")
        assert cfg.side == "right"

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError, match="Invalid side"):
            BrainCoConfig(side="both")

    def test_with_filter(self):
        fcfg = FilterConfig(type="ema", alpha=0.5)
        cfg = BrainCoConfig(filter=fcfg)
        assert cfg.filter is fcfg


# ---------------------------------------------------------------------------
# VizConfig
# ---------------------------------------------------------------------------


class TestVizConfig:
    def test_defaults(self):
        cfg = VizConfig()
        assert cfg.enabled is True
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8080
        assert cfg.show_frames is False
        assert cfg.show_geometry is True
        assert cfg.show_ee_spheres is True
        assert cfg.fps == 30.0

    def test_disabled(self):
        cfg = VizConfig(enabled=False)
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# BrainCoNodeConfig
# ---------------------------------------------------------------------------


class TestBrainCoNodeConfig:
    def test_defaults(self):
        cfg = BrainCoNodeConfig()
        assert isinstance(cfg.subscriber, SubscriberConfig)
        assert cfg.interface.mode == "mock"
        assert cfg.brainco.side == "left"
        assert cfg.viz.enabled is True
        assert cfg.data_endpoint == "tcp://*:5556"
        assert cfg.bind_data is True
        assert cfg.observation_rate_hz is None

    def test_robot_type(self):
        cfg = BrainCoNodeConfig()
        assert cfg.robot_type == "brainco"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_mock_config(self):
        yaml_data = {
            "subscriber": {"address": "tcp://localhost:5555"},
            "interface": {"mode": "mock"},
            "brainco": {"side": "right"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(yaml_data, f)
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            assert cfg.interface.mode == "mock"
            assert cfg.brainco.side == "right"
            assert cfg.subscriber.address == "tcp://localhost:5555"
        finally:
            Path(tmp_path).unlink()

    def test_loads_hw_config(self):
        yaml_data = {
            "interface": {
                "mode": "hw",
                "hw": {
                    "backend": "modbus_rs485",
                    "rs485": {"port": "COM5", "baud": 460800},
                    "slave_id": 126,
                },
            },
            "brainco": {"side": "left"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(yaml_data, f)
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            assert cfg.interface.mode == "hw"
            hw = cfg.interface.get_hardware_config()
            assert hw is not None
            assert hw.backend == "modbus_rs485"
            assert hw.rs485.port == "COM5"
            assert hw.rs485.baud == 460800
            assert hw.effective_slave_id("left") == 126
        finally:
            Path(tmp_path).unlink()

    def test_loads_with_filter_config(self):
        yaml_data = {
            "brainco": {
                "side": "left",
                "filter": {"type": "ema", "alpha": 0.3},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(yaml_data, f)
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            assert cfg.brainco.filter is not None
            assert cfg.brainco.filter.type == "ema"
            assert cfg.brainco.filter.alpha == 0.3
        finally:
            Path(tmp_path).unlink()

    def test_loads_viz_config(self):
        yaml_data = {
            "viz": {"enabled": False, "port": 9000},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(yaml_data, f)
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            assert cfg.viz.enabled is False
            assert cfg.viz.port == 9000
        finally:
            Path(tmp_path).unlink()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")
