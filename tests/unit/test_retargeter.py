"""Unit tests for dexim.brainco.node.retargeter.Retargeter (VectorOptimizer-based)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from dexim.brainco.node.retargeter import Retargeter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_optimizer(alpha=None):
    """Create a mock VectorOptimizerProtocol with configurable alpha."""
    opt = MagicMock()
    opt.alpha = alpha if alpha is not None else [1.0] * 5
    return opt


def _make_success_result(q=None):
    """Return a (q, nlopt_result) tuple representing optimisation success."""
    q = q if q is not None else np.ones(6, dtype=np.float64)
    result = SimpleNamespace(value=1, name="SUCCESS")
    return q, result


def _make_failure_result():
    """Return a (q, nlopt_result) tuple representing optimisation failure."""
    q = np.zeros(6, dtype=np.float64)
    result = SimpleNamespace(value=-1, name="FAILURE")
    return q, result


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_stores_optimizer(self):
        opt = _make_mock_optimizer()
        r = Retargeter(opt)
        assert r._optimizer is opt

    def test_sets_alpha_on_optimizer(self):
        opt = _make_mock_optimizer()
        Retargeter(opt, alpha=[0.5, 0.6, 0.7, 0.8, 0.9])
        assert opt.alpha == [0.5, 0.6, 0.7, 0.8, 0.9]

    def test_default_alpha_unchanged_when_none(self):
        opt = _make_mock_optimizer(alpha=[1.0] * 5)
        Retargeter(opt, alpha=None)
        assert opt.alpha == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_graceful_fallback_on_alpha_set_failure(self):
        """If setting alpha on the optimizer raises, defaults to [1.0]*5."""
        opt = MagicMock()
        del opt.alpha  # Remove alpha attribute to trigger AttributeError
        r = Retargeter(opt, alpha=[0.5] * 5)
        assert r._optimizer.alpha == [1.0] * 5


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


class TestScale:
    def test_identity_scale(self):
        opt = _make_mock_optimizer(alpha=[1.0] * 5)
        r = Retargeter(opt)
        features = np.ones((5, 3), dtype=np.float64)
        scaled = r.scale(features)
        np.testing.assert_allclose(scaled, features, atol=1e-10)

    def test_half_scale(self):
        opt = _make_mock_optimizer(alpha=[0.5] * 5)
        r = Retargeter(opt)
        features = np.ones((5, 3), dtype=np.float64)
        scaled = r.scale(features)
        np.testing.assert_allclose(scaled, np.full((5, 3), 0.5), atol=1e-10)

    def test_per_finger_scale(self):
        opt = _make_mock_optimizer(alpha=[1.0, 2.0, 3.0, 4.0, 5.0])
        r = Retargeter(opt)
        features = np.ones((5, 3), dtype=np.float64)
        scaled = r.scale(features)
        expected = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(scaled, expected)

    def test_scale_reads_alpha_from_optimizer(self):
        """scale() uses the optimizer's current alpha, not a cached copy."""
        opt = _make_mock_optimizer(alpha=[0.5] * 5)
        r = Retargeter(opt)
        opt.alpha = [2.0] * 5  # change alpha after construction
        features = np.ones((5, 3), dtype=np.float64)
        scaled = r.scale(features)
        np.testing.assert_allclose(scaled, np.full((5, 3), 2.0), atol=1e-10)


# ---------------------------------------------------------------------------
# Retarget
# ---------------------------------------------------------------------------


class TestRetarget:
    def test_delegates_to_optimizer(self):
        opt = _make_mock_optimizer()
        q_expected = np.arange(6, dtype=np.float64)
        opt.retarget.return_value = _make_success_result(q=q_expected)
        r = Retargeter(opt)

        features = np.ones((5, 3), dtype=np.float64)
        q = r.retarget(features)

        opt.retarget.assert_called_once()
        np.testing.assert_array_equal(q, q_expected)

    def test_returns_none_on_optimizer_failure(self):
        opt = _make_mock_optimizer()
        opt.retarget.return_value = _make_failure_result()
        r = Retargeter(opt)

        features = np.ones((5, 3), dtype=np.float64)
        q = r.retarget(features)

        assert q is None

    def test_stopval_reached_is_not_failure(self):
        """STOPVAL_REACHED (value=2) is >= 1, so it's accepted."""
        opt = _make_mock_optimizer()
        q_expected = np.ones(6, dtype=np.float64)
        result = SimpleNamespace(value=2, name="STOPVAL_REACHED")
        opt.retarget.return_value = (q_expected, result)
        r = Retargeter(opt)

        q = r.retarget(np.ones((5, 3)))
        assert q is not None

    def test_maxeval_reached_is_accepted(self):
        """MAXEVAL_REACHED (value=5) is >= 1, so it's accepted."""
        opt = _make_mock_optimizer()
        q_expected = np.ones(6, dtype=np.float64)
        result = SimpleNamespace(value=5, name="MAXEVAL_REACHED")
        opt.retarget.return_value = (q_expected, result)
        r = Retargeter(opt)

        q = r.retarget(np.ones((5, 3)))
        assert q is not None

    def test_output_shape(self):
        opt = _make_mock_optimizer()
        opt.retarget.return_value = _make_success_result()
        r = Retargeter(opt)

        q = r.retarget(np.ones((5, 3), dtype=np.float64))
        assert q is not None
        assert q.shape == (6,)
