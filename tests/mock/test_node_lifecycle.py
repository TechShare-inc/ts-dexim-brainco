"""Mock integration tests -- BrainCoControlNode lifecycle hooks.

Exercises on_start / on_pause / on_stop / on_shutdown without hardware.
All external dependencies are replaced by the fixtures in conftest.py.

Fixtures used:
  make_brainco_node  -- factory from tests/mock/conftest.py
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


def _mock_interface(nq: int = 6) -> MagicMock:
    """Return a MagicMock satisfying RobotInterface.

    Uses non-zero q so move_to_safe_position always has a non-trivial delta
    to the zero safe position and actually calls write().
    """
    iface = MagicMock()
    iface.read.return_value = SimpleNamespace(
        q=np.ones(nq), qd=np.zeros(nq), tau=np.zeros(nq), stamp=0.0
    )
    return iface


# ---------------------------------------------------------------------------
# on_start
# ---------------------------------------------------------------------------


class TestOnStart:
    def test_enables_publishing(self, make_brainco_node):
        node = make_brainco_node()
        assert node._publisher.is_active is False
        node.on_start()
        assert node._publisher.is_active is True

    def test_idempotent(self, make_brainco_node):
        """Calling on_start twice leaves publisher active."""
        node = make_brainco_node()
        node.on_start()
        node.on_start()
        assert node._publisher.is_active is True

    def test_connects_interface(self, make_brainco_node):
        """on_start() must call interface.connect() so the hardware link is opened."""
        node = make_brainco_node()
        assert not node.interface._connected
        node.on_start()
        assert node.interface._connected

    def test_initializes_motion_controller(self, make_brainco_node):
        """on_start() must call motion_controller.initialize() to seed velocity limiting."""
        node = make_brainco_node()
        import time as _time

        before = _time.time()
        node.on_start()
        assert node._motion_controller._last_data_time >= before


# ---------------------------------------------------------------------------
# on_pause
# ---------------------------------------------------------------------------


class TestOnPause:
    def test_disables_publishing(self, make_brainco_node):
        node = make_brainco_node()
        node.on_start()
        assert node._publisher.is_active is True
        node.on_pause()
        assert node._publisher.is_active is False

    def test_pause_without_prior_start(self, make_brainco_node):
        """on_pause must not raise even if called before on_start."""
        node = make_brainco_node()
        node.on_pause()  # must not raise
        assert node._publisher.is_active is False


# ---------------------------------------------------------------------------
# on_stop
# ---------------------------------------------------------------------------


class TestOnStop:
    def test_disables_publishing(self, make_brainco_node):
        node = make_brainco_node()
        node.on_start()
        node.on_stop()
        assert node._publisher.is_active is False

    def test_sends_safe_position_command(self, make_brainco_node):
        """on_stop calls move_to_safe which issues a write() to the interface."""
        node = make_brainco_node()
        mock_iface = _mock_interface()
        node.interface = mock_iface
        node._motion_controller._interface = mock_iface
        with patch.object(node._motion_controller.rate_limiter, "sleep"):
            node.on_stop()
        mock_iface.write.assert_called()

    def test_disconnects_interface(self, make_brainco_node):
        """on_stop() must call interface.disconnect() to release the hardware link."""
        node = make_brainco_node()
        node.on_start()
        assert node.interface._connected
        with patch.object(node._motion_controller.rate_limiter, "sleep"):
            node.on_stop()
        assert not node.interface._connected

    def test_stop_after_pause_is_safe(self, make_brainco_node):
        """on_stop must not raise even if called from a paused state."""
        node = make_brainco_node()
        mock_iface = _mock_interface()
        node.interface = mock_iface
        node._motion_controller._interface = mock_iface
        node.on_start()
        node.on_pause()
        with patch.object(node._motion_controller.rate_limiter, "sleep"):
            node.on_stop()  # must not raise
        assert node._publisher.is_active is False


# ---------------------------------------------------------------------------
# on_shutdown
# ---------------------------------------------------------------------------


class TestOnShutdown:
    def test_disables_publishing(self, make_brainco_node):
        node = make_brainco_node()
        node.on_start()
        node.on_shutdown()
        assert node._publisher.is_active is False

    def test_closes_pub_socket(self, make_brainco_node):
        """on_shutdown must close and nullify _publisher._pub."""
        node = make_brainco_node()
        assert node._publisher._pub is not None
        node.on_shutdown()
        assert node._publisher._pub is None

    def test_shutdown_without_start_is_safe(self, make_brainco_node):
        """on_shutdown must not raise if node was never started."""
        node = make_brainco_node()
        node.on_shutdown()  # must not raise
        assert node._publisher.is_active is False

    def test_double_shutdown_is_safe(self, make_brainco_node):
        """Calling on_shutdown twice must not raise."""
        node = make_brainco_node()
        node.on_shutdown()
        node.on_shutdown()  # must not raise
        assert node._publisher._pub is None


# ---------------------------------------------------------------------------
# Start -> stop -> shutdown sequence
# ---------------------------------------------------------------------------


class TestLifecycleSequence:
    def test_full_sequence(self, make_brainco_node):
        """start -> pause -> stop -> shutdown leaves node in a clean state."""
        node = make_brainco_node()
        node.interface = _mock_interface()

        node.on_start()
        assert node._publisher.is_active is True

        node.on_pause()
        assert node._publisher.is_active is False

        with patch.object(node._motion_controller.rate_limiter, "sleep"):
            node.on_stop()
        assert node._publisher.is_active is False

        node.on_shutdown()
        assert node._publisher.is_active is False
        assert node._publisher._pub is None
