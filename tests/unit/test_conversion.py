"""Unit tests for dexim.brainco.interface._conversion -- pure conversion functions."""

from __future__ import annotations

import numpy as np
import pytest

from dexim.brainco.interface._conversion import (
    API_TO_INTERNAL_MAP,
    API_VALUE_MAX,
    API_VALUE_MIN,
    INTERNAL_TO_API_MAP,
    JOINT_LIMITS,
    NUM_JOINTS,
    ApiJoint,
    InternalJoint,
    api_to_radians,
    radians_to_api,
    urdf_joint_names,
)

# Convenience arrays of the lower/upper limits in internal order.
_LOWER = np.array([lo for lo, _ in JOINT_LIMITS])
_UPPER = np.array([hi for _, hi in JOINT_LIMITS])


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


class TestConstants:
    def test_num_joints(self):
        assert NUM_JOINTS == 6

    def test_joint_limits_count(self):
        assert len(JOINT_LIMITS) == NUM_JOINTS

    def test_all_lower_less_than_upper(self):
        for lo, hi in JOINT_LIMITS:
            assert lo < hi, f"Expected lower < upper, got ({lo}, {hi})"

    def test_api_range(self):
        assert API_VALUE_MIN == 0
        assert API_VALUE_MAX == 1000

    def test_lower_limits_all_zero(self):
        """Revo2 all joints have lower limit 0 (open hand)."""
        for lo, _ in JOINT_LIMITS:
            assert lo == 0.0

    def test_thumb_flex_limit_is_59_deg(self):
        lo, hi = JOINT_LIMITS[InternalJoint.THUMB_FLEX]
        np.testing.assert_allclose(hi, np.deg2rad(59.0))

    def test_thumb_aux_limit_is_90_deg(self):
        lo, hi = JOINT_LIMITS[InternalJoint.THUMB_AUX]
        np.testing.assert_allclose(hi, np.deg2rad(90.0))

    def test_finger_limits_are_81_deg(self):
        for idx in [
            InternalJoint.INDEX,
            InternalJoint.MIDDLE,
            InternalJoint.RING,
            InternalJoint.PINKY,
        ]:
            lo, hi = JOINT_LIMITS[idx]
            np.testing.assert_allclose(hi, np.deg2rad(81.0))


# ---------------------------------------------------------------------------
# Internal <-> API mapping
# ---------------------------------------------------------------------------


class TestIndexMaps:
    def test_internal_to_api_map_length(self):
        assert len(INTERNAL_TO_API_MAP) == NUM_JOINTS

    def test_api_to_internal_map_length(self):
        assert len(API_TO_INTERNAL_MAP) == NUM_JOINTS

    def test_internal_to_api_is_permutation(self):
        """Every API index 0..5 appears exactly once in INTERNAL_TO_API_MAP."""
        assert sorted(INTERNAL_TO_API_MAP) == list(range(NUM_JOINTS))

    def test_api_to_internal_is_permutation(self):
        """Every internal index 0..5 appears exactly once in API_TO_INTERNAL_MAP."""
        assert sorted(API_TO_INTERNAL_MAP) == list(range(NUM_JOINTS))

    def test_maps_are_inverses(self):
        """Applying both maps in sequence should return the identity permutation."""
        for internal_idx in range(NUM_JOINTS):
            api_idx = INTERNAL_TO_API_MAP[internal_idx]
            recovered = API_TO_INTERNAL_MAP[api_idx]
            assert (
                recovered == internal_idx
            ), f"Round-trip failed: internal {internal_idx} -> API {api_idx} -> internal {recovered}"

    def test_order_is_identity(self):
        """BrainCo internal order and API order are the same."""
        for i in range(NUM_JOINTS):
            assert INTERNAL_TO_API_MAP[i] == i
            assert API_TO_INTERNAL_MAP[i] == i


# ---------------------------------------------------------------------------
# urdf_joint_names
# ---------------------------------------------------------------------------


class TestUrdfJointNames:
    def test_left_prefix(self):
        names = urdf_joint_names("left")
        assert all(n.startswith("L_") for n in names)
        assert len(names) == 6

    def test_right_prefix(self):
        names = urdf_joint_names("right")
        assert all(n.startswith("R_") for n in names)
        assert len(names) == 6

    def test_left_thumb_names(self):
        names = urdf_joint_names("left")
        assert names[0] == "L_thumb_proximal"
        assert names[1] == "L_thumb_metacarpal"

    def test_right_finger_names(self):
        names = urdf_joint_names("right")
        assert names[2] == "R_index_proximal"
        assert names[3] == "R_middle_proximal"
        assert names[4] == "R_ring_proximal"
        assert names[5] == "R_pinky_proximal"


# ---------------------------------------------------------------------------
# radians_to_api
# ---------------------------------------------------------------------------


class TestRadiansToApi:
    def test_open_hand_gives_api_min(self):
        """Lower joint limits (0 rad = open) map to API_VALUE_MIN (0)."""
        result = radians_to_api(_LOWER)
        np.testing.assert_array_equal(result, np.full(NUM_JOINTS, API_VALUE_MIN))

    def test_closed_hand_gives_api_max(self):
        """Upper joint limits (closed) map to API_VALUE_MAX (1000)."""
        result = radians_to_api(_UPPER)
        np.testing.assert_array_equal(result, np.full(NUM_JOINTS, API_VALUE_MAX))

    def test_midpoint_maps_to_500(self):
        """Midpoint of each joint range should give API ~= 500."""
        midpoints = (_LOWER + _UPPER) / 2.0
        result = radians_to_api(midpoints)
        np.testing.assert_allclose(
            result,
            np.full(NUM_JOINTS, 500),
            atol=1,  # int rounding
        )

    def test_output_shape(self):
        result = radians_to_api(_LOWER)
        assert result.shape == (NUM_JOINTS,)

    def test_output_dtype_is_int32(self):
        result = radians_to_api(_LOWER)
        assert result.dtype == np.int32

    def test_clips_below_lower_limit(self):
        """Values below lower limit (negative) clipped to lower -> API MIN."""
        q_below = _LOWER - 1.0
        result = radians_to_api(q_below)
        np.testing.assert_array_equal(result, np.full(NUM_JOINTS, API_VALUE_MIN))

    def test_clips_above_upper_limit(self):
        """Values above upper limit clipped to upper -> API MAX."""
        q_above = _UPPER + 1.0
        result = radians_to_api(q_above)
        np.testing.assert_array_equal(result, np.full(NUM_JOINTS, API_VALUE_MAX))

    def test_output_always_in_valid_range(self):
        rng = np.random.default_rng(42)
        q_random = rng.uniform(-5.0, 5.0, size=NUM_JOINTS)
        result = radians_to_api(q_random)
        assert np.all(result >= API_VALUE_MIN)
        assert np.all(result <= API_VALUE_MAX)

    def test_thumb_flex_at_limit_gives_max(self):
        """Thumb flex at its upper limit -> API MAX."""
        q = _LOWER.copy()
        q[InternalJoint.THUMB_FLEX] = JOINT_LIMITS[InternalJoint.THUMB_FLEX][1]
        result = radians_to_api(q)
        assert result[ApiJoint.THUMB_FLEX] == API_VALUE_MAX


# ---------------------------------------------------------------------------
# api_to_radians
# ---------------------------------------------------------------------------


class TestApiToRadians:
    def test_api_min_gives_open_hand(self):
        """API 0 (open) maps to lower joint limits."""
        result = api_to_radians(np.full(NUM_JOINTS, API_VALUE_MIN))
        np.testing.assert_allclose(result, _LOWER, atol=1e-10)

    def test_api_max_gives_closed_hand(self):
        """API 1000 (closed) maps to upper joint limits."""
        result = api_to_radians(np.full(NUM_JOINTS, API_VALUE_MAX))
        np.testing.assert_allclose(result, _UPPER, atol=1e-8)

    def test_api_500_gives_midpoint(self):
        """API 500 maps to midpoint of joint range."""
        result = api_to_radians(np.full(NUM_JOINTS, 500))
        expected = (_LOWER + _UPPER) / 2.0
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_output_shape(self):
        result = api_to_radians(np.full(NUM_JOINTS, 500))
        assert result.shape == (NUM_JOINTS,)

    def test_output_dtype_is_float64(self):
        result = api_to_radians(np.full(NUM_JOINTS, 500))
        assert result.dtype == np.float64

    def test_clips_below_api_min(self):
        """API values below 0 clipped to 0 -> open hand."""
        result = api_to_radians(np.full(NUM_JOINTS, -100))
        np.testing.assert_allclose(result, _LOWER, atol=1e-10)

    def test_clips_above_api_max(self):
        """API values above 1000 clipped to 1000 -> closed hand."""
        result = api_to_radians(np.full(NUM_JOINTS, 2000))
        np.testing.assert_allclose(result, _UPPER, atol=1e-8)

    def test_output_always_in_joint_limits(self):
        rng = np.random.default_rng(42)
        api_random = rng.integers(-500, 1500, size=NUM_JOINTS)
        result = api_to_radians(api_random)
        for i in range(NUM_JOINTS):
            lo, hi = JOINT_LIMITS[i]
            assert (
                lo <= result[i] <= hi + 1e-10
            ), f"Joint {i}: {result[i]} not in [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_radians_round_trip(self):
        """radians -> api -> radians should be identity for in-range values."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            q_in = rng.uniform(_LOWER, _UPPER)
            api = radians_to_api(q_in)
            q_out = api_to_radians(api)
            # Tolerance: 1 API unit worth of radians.
            max_range = max(hi - lo for lo, hi in JOINT_LIMITS)
            atol_per_joint = max_range / API_VALUE_MAX + 1e-8
            np.testing.assert_allclose(q_out, q_in, atol=atol_per_joint)

    def test_open_round_trip(self):
        """Open hand (all zeros) round-trips."""
        q_in = _LOWER.copy()
        api = radians_to_api(q_in)
        q_out = api_to_radians(api)
        np.testing.assert_allclose(q_out, q_in, atol=1e-10)

    def test_closed_round_trip(self):
        """Closed hand (all at upper limits) round-trips."""
        q_in = _UPPER.copy()
        api = radians_to_api(q_in)
        q_out = api_to_radians(api)
        np.testing.assert_allclose(q_out, q_in, atol=1e-8)
