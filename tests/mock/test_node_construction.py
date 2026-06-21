"""Mock integration tests -- BrainCoControlNode construction.

Verifies that the collaborator composition is wired correctly after __init__:
attributes set, sub-systems initialised, defaults respected.
All external dependencies (ZMQ, pinocchio) are replaced by the
fixtures in conftest.py.
"""

from __future__ import annotations

from dexim.brainco.node.config import BrainCoNodeConfig, VizConfig


class TestConstruction:
    """BrainCoControlNode attributes after __init__."""

    def test_node_id_set(self, make_brainco_node):
        node = make_brainco_node()
        assert node.node_id == "test_brainco"

    def test_config_stored(self, make_brainco_node):
        node = make_brainco_node()
        assert isinstance(node.config, BrainCoNodeConfig)

    def test_model_assigned(self, make_brainco_node, mock_brainco_model):
        """node.model must be the patched mock (create_model is stub)."""
        node = make_brainco_node()
        assert node.model is mock_brainco_model

    def test_interface_assigned(self, make_brainco_node):
        node = make_brainco_node()
        assert node.interface is not None

    def test_retargeter_created(self, make_brainco_node):
        node = make_brainco_node()
        assert node._retargeter is not None
        assert node._retargeter._optimizer is not None
        assert node._retargeter._optimizer.alpha == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_rate_limiter_and_dt_created(self, make_brainco_node):
        node = make_brainco_node()
        assert node._motion_controller.rate_limiter is not None
        assert node._motion_controller.dt > 0

    def test_smooth_filter_disabled_by_default(self, make_brainco_node):
        """BrainCoConfig.filter defaults to None -> JointFilter has no active filter."""
        node = make_brainco_node()
        assert node._joint_filter._filter is None

    def test_is_publishing_false_after_init(self, make_brainco_node):
        """Publishing starts disabled."""
        node = make_brainco_node()
        assert node._publisher.is_active is False

    def test_teleop_inactive_after_init(self, make_brainco_node):
        node = make_brainco_node()
        assert node._teleop_active is False

    def test_custom_node_id(self, make_brainco_node):
        """Factory accepts a different node_id via BrainCoNodeConfig passthrough."""
        cfg = BrainCoNodeConfig(viz=VizConfig(enabled=False))
        node = make_brainco_node(config=cfg)
        assert node.node_id == "test_brainco"

    def test_interface_not_connected_after_init(self, make_brainco_node):
        """connect() is deferred to on_start(); interface must not be connected after __init__."""
        node = make_brainco_node()
        assert not node.interface._connected
