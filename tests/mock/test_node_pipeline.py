"""Mock integration tests -- BrainCoControlNode._run_pipeline() pipeline.

Exercises the full control loop without hardware, ZMQ, or pinocchio.
The _run_pipeline() pipeline is:
  receive_data -> extract_features -> scale_features -> retarget ->
  filter_joints -> apply_velocity_limits -> send_command ->
  publish_action (with features) -> publish_observation

Fixtures used:
  make_brainco_node  -- factory from tests/mock/conftest.py
  mock_subscriber    -- Mock with read/read_batch stubs
  make_hand_state    -- factory from tests/conftest.py
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from dexim.core.messages import HandState, SkeletonJoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_interface(nq: int = 6) -> MagicMock:
    """Return a MagicMock satisfying RobotInterface."""
    iface = MagicMock()
    iface.read.return_value = SimpleNamespace(
        q=np.ones(nq), qd=np.zeros(nq), tau=np.zeros(nq), stamp=0.0
    )
    return iface


def _make_hand_state(num_joints: int = 25) -> HandState:
    """Build a minimal HandState with position-only joints."""
    joints = []
    for i in range(num_joints):
        x = float(i) * 0.03
        joints.append(
            SkeletonJoint(
                id=i,
                position=(0.0, x, 0.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
            )
        )
    return HandState(glove_id=0, side="left", timestamp=1.0, joints=joints)


def _patch_sleep(node) -> AbstractContextManager:
    """Patch rate_limiter.sleep to a no-op for test speed."""
    return patch.object(
        node._motion_controller.rate_limiter,
        "sleep",
        return_value={"overtime": False, "elapsed": 0.0, "jitter": 0.0},
    )


# ---------------------------------------------------------------------------
# Inactive teleop path
# ---------------------------------------------------------------------------


class TestGetDataInactive:
    """When _teleop_active is False, _run_pipeline() drains subscriber and returns early."""

    def test_drain_calls_subscriber_read_latest_batch(
        self, make_brainco_node, mock_subscriber
    ):
        node = make_brainco_node()
        node._teleop_active = False
        node.interface = _mock_interface()

        with _patch_sleep(node):
            node._run_pipeline()

        mock_subscriber.read_latest_batch.assert_called()

    def test_no_command_sent_when_inactive(self, make_brainco_node, mock_subscriber):
        node = make_brainco_node()
        mock_iface = _mock_interface()
        node.interface = mock_iface
        node._motion_controller._interface = mock_iface
        node._teleop_active = False

        with _patch_sleep(node):
            node._run_pipeline()

        mock_iface.write.assert_not_called()


# ---------------------------------------------------------------------------
# Active teleop path -- valid data
# ---------------------------------------------------------------------------


class TestGetDataActive:
    """When _teleop_active is True and data is available, the full pipeline runs."""

    def test_send_command_called_with_valid_data(
        self, make_brainco_node, mock_subscriber
    ):
        node = make_brainco_node()
        mock_iface = _mock_interface()
        node.interface = mock_iface
        node._motion_controller._interface = mock_iface
        node._teleop_active = True
        node._publisher._is_active = True

        hand_state = _make_hand_state()
        # SkeletonReceiver uses read_latest_batch(), not read()
        mock_subscriber.read_latest_batch.return_value = [hand_state]
        # disable timeout checking via mock
        node._motion_controller.check_timeout = MagicMock(return_value=False)

        with _patch_sleep(node):
            node._run_pipeline()

        mock_iface.write.assert_called()

    def test_pipeline_primed_after_first_tick(self, make_brainco_node, mock_subscriber):
        node = make_brainco_node()
        node.interface = _mock_interface()
        node._motion_controller._interface = node.interface
        node._teleop_active = True

        hand_state = _make_hand_state()
        mock_subscriber.read_latest_batch.return_value = [hand_state]

        with _patch_sleep(node):
            node._run_pipeline()

        assert node._pipeline_primed is True

    def test_publishes_action_on_valid_data(self, make_brainco_node, mock_subscriber):
        node = make_brainco_node()
        node.interface = _mock_interface()
        node._motion_controller._interface = node.interface
        node._teleop_active = True
        node._publisher._is_active = True

        hand_state = _make_hand_state()
        mock_subscriber.read_latest_batch.return_value = [hand_state]

        with _patch_sleep(node):
            node._run_pipeline()

        node._publisher._pub.send_multipart.assert_called()
        # One call for action, one for observation
        assert node._publisher._pub.send_multipart.call_count >= 1

    def test_publishes_observation_in_active(self, make_brainco_node, mock_subscriber):
        node = make_brainco_node()
        node.interface = _mock_interface()
        node._motion_controller._interface = node.interface
        node._teleop_active = True
        node._publisher._is_active = True

        hand_state = _make_hand_state()
        mock_subscriber.read_latest_batch.return_value = [hand_state]

        with _patch_sleep(node):
            node._run_pipeline()

        # Check that observation was published (at least 2 send_multipart calls: action + obs)
        assert node._publisher._pub.send_multipart.call_count >= 2


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_health_check_increases_counter(self, make_brainco_node, mock_subscriber):
        node = make_brainco_node()
        node.interface = _mock_interface()
        node._teleop_active = False

        with _patch_sleep(node):
            node._run_pipeline()

        assert node._health_check_counter == 1


# ---------------------------------------------------------------------------
# Safe position
# ---------------------------------------------------------------------------


class TestSafePosition:
    def test_safe_position_is_zeros(self, make_brainco_node):
        node = make_brainco_node()
        safe = node.get_safe_position()
        np.testing.assert_array_equal(safe, np.zeros(6))
