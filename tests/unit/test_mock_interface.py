"""Unit tests for dexim.brainco.interface.mock_interface.MockInterface."""

from __future__ import annotations

import time

import numpy as np
import pytest
from dexim.core.robot_interface import JointCommand

from dexim.brainco.interface._conversion import JOINT_LIMITS, NUM_JOINTS
from dexim.brainco.interface.mock_interface import MockInterface

_LOWER = np.array([lo for lo, _ in JOINT_LIMITS])
_UPPER = np.array([hi for _, hi in JOINT_LIMITS])


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestMockInterfaceInit:
    def test_default_joint_count(self):
        m = MockInterface()
        assert m.num_joint_configurations() == NUM_JOINTS

    def test_left_joint_name_prefix(self):
        m = MockInterface(hand_side="left")
        names = m.joint_names()
        assert all(n.startswith("L_") for n in names)

    def test_right_joint_name_prefix(self):
        m = MockInterface(hand_side="right")
        names = m.joint_names()
        assert all(n.startswith("R_") for n in names)

    def test_joint_names_count(self):
        m = MockInterface()
        assert len(m.joint_names()) == NUM_JOINTS

    def test_actuated_equals_full(self):
        m = MockInterface()
        assert m.num_actuated_configurations() == m.num_full_configurations()
        assert m.actuated_joint_names() == m.full_joint_names()

    def test_initial_state_at_lower_limits(self):
        m = MockInterface()
        m.connect()
        state = m.read()
        np.testing.assert_allclose(state.q, _LOWER, atol=1e-10)
        m.disconnect()

    def test_init_with_mock_model_uses_model_metadata(self, mock_brainco_model):
        m = MockInterface(brainco_model=mock_brainco_model)
        assert m.num_joint_configurations() == 6
        names = m.joint_names()
        assert len(names) == 6
        # Names should come from the mock model's pinocchio names (L_ prefix)
        assert names[0] == "L_thumb_proximal"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestMockInterfaceLifecycle:
    def test_connect_sets_connected(self):
        m = MockInterface()
        assert not m._connected
        m.connect()
        assert m._connected
        m.disconnect()

    def test_disconnect_clears_connected(self):
        m = MockInterface()
        m.connect()
        m.disconnect()
        assert not m._connected

    def test_double_connect_logs_warning(self, caplog):
        m = MockInterface()
        m.connect()
        m.connect()  # second call -> warning
        assert any("already connected" in msg.lower() for msg in caplog.messages)
        m.disconnect()

    def test_disconnect_when_not_connected_is_noop(self):
        m = MockInterface()
        m.disconnect()  # should not raise

    def test_time_increases_after_connect(self):
        m = MockInterface()
        m.connect()
        t0 = m.time()
        time.sleep(0.05)
        t1 = m.time()
        assert t1 > t0
        m.disconnect()

    def test_time_is_zero_before_connect(self):
        m = MockInterface()
        assert m.time() == 0.0

    def test_estop_default_false(self):
        m = MockInterface()
        assert m.estop() is False

    def test_set_estop(self):
        m = MockInterface()
        m.set_estop(True)
        assert m.estop() is True
        m.set_estop(False)
        assert m.estop() is False


# ---------------------------------------------------------------------------
# Read / Write (loopback)
# ---------------------------------------------------------------------------


class TestMockInterfaceReadWrite:
    def test_write_then_read_returns_command(self):
        m = MockInterface()
        m.connect()
        q_cmd = (_LOWER + _UPPER) / 2.0  # midpoint
        m.write(JointCommand(mode="position", q=q_cmd))
        state = m.read()
        np.testing.assert_allclose(state.q, q_cmd, atol=1e-10)
        m.disconnect()

    def test_write_clips_to_limits(self):
        m = MockInterface()
        m.connect()
        q_beyond = _UPPER + 1.0  # beyond upper limit
        m.write(JointCommand(mode="position", q=q_beyond))
        state = m.read()
        np.testing.assert_allclose(state.q, _UPPER, atol=1e-10)
        m.disconnect()

    def test_write_below_lower_limit_clips(self):
        m = MockInterface()
        m.connect()
        q_below = _LOWER - 1.0  # below lower limit
        m.write(JointCommand(mode="position", q=q_below))
        state = m.read()
        np.testing.assert_allclose(state.q, _LOWER, atol=1e-10)
        m.disconnect()

    def test_write_open_hand(self):
        m = MockInterface()
        m.connect()
        m.write(JointCommand(mode="position", q=_LOWER))
        state = m.read()
        np.testing.assert_allclose(state.q, _LOWER, atol=1e-10)
        m.disconnect()

    def test_write_closed_hand(self):
        m = MockInterface()
        m.connect()
        m.write(JointCommand(mode="position", q=_UPPER))
        state = m.read()
        np.testing.assert_allclose(state.q, _UPPER, atol=1e-8)
        m.disconnect()

    def test_read_before_connect_raises(self):
        m = MockInterface()
        with pytest.raises(RuntimeError, match="not connected"):
            m.read()

    def test_write_before_connect_raises(self):
        m = MockInterface()
        with pytest.raises(RuntimeError, match="not connected"):
            m.write(JointCommand(mode="position", q=_LOWER))

    def test_write_wrong_joint_count_raises(self):
        m = MockInterface()
        m.connect()
        with pytest.raises(ValueError, match="Expected 6"):
            m.write(JointCommand(mode="position", q=np.zeros(5)))
        m.disconnect()

    def test_write_none_q_raises(self):
        m = MockInterface()
        m.connect()
        with pytest.raises(ValueError, match="required"):
            m.write(JointCommand(mode="position", q=None))
        m.disconnect()

    def test_write_non_position_mode_raises(self):
        m = MockInterface()
        m.connect()
        with pytest.raises(NotImplementedError, match="position"):
            m.write(JointCommand(mode="velocity", q=_LOWER))
        m.disconnect()

    def test_multiple_writes_preserve_last(self):
        m = MockInterface()
        m.connect()
        m.write(JointCommand(mode="position", q=_LOWER))
        m.write(JointCommand(mode="position", q=_UPPER))
        state = m.read()
        np.testing.assert_allclose(state.q, _UPPER, atol=1e-8)
        m.disconnect()
