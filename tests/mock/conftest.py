"""Shared fixtures for mock (no-hardware) integration tests.

Strategy
--------
- ``ManagedNode.__init__`` ZMQ setup is patched so no ZMQ ports are bound.
- ``DataPlanePublisher.__init__`` is patched to a no-op so no ZMQ PUB
  socket is created.
- ``create_model`` is patched to return ``mock_brainco_model``
  (from the parent conftest) so pinocchio / URDF are not required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from dexim.brainco.node.config import (
    BrainCoNodeConfig,
    VizConfig,
)

# ---------------------------------------------------------------------------
# Individual patch fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_subscriber() -> Mock:
    """A default mock data subscriber (returns no data)."""
    sub = Mock()
    sub.read.return_value = None
    sub.read_batch.return_value = []
    sub.read_latest_batch.return_value = []
    sub.close.return_value = None
    return sub


@pytest.fixture()
def patch_zmq():
    """Patch ``DataPlanePublisher.__init__`` to a no-op.

    Replaces the ZMQ PUB socket setup with a minimal stub so no port is
    bound during tests.  The ``_pub``, ``_action_topic``, and
    ``_state_topic`` attributes are set to mocks/bytes so downstream
    publish helpers don't raise ``AttributeError``.
    """

    def _noop_publisher_init(
        self,
        node_id: str,
        data_endpoint: str = "tcp://*:5556",
        bind: bool = True,
    ) -> None:
        self._pub = MagicMock()
        self._is_active = False
        self._action_topic = b"action"
        self._state_topic = b"state"

    with patch(
        "dexim.core.collaborators.data_publisher.DataPlanePublisher.__init__",
        _noop_publisher_init,
    ):
        yield


@pytest.fixture()
def patch_managed_node():
    """Patch ``ManagedNode.__init__`` to skip ZMQ context and socket setup.

    Provides the minimal set of instance attributes that
    ``BrainCoControlNode`` and its mixins expect after construction.
    """

    def _fake_managed_init(
        self,
        node_id: str,
        control_endpoint: str | None = None,
        status_endpoint: str | None = None,
        heartbeat_interval: float = 1.0,
    ) -> None:
        self.node_id = node_id
        self.heartbeat_interval = float(heartbeat_interval)
        self.is_recording = False
        self._teleop_active = False
        self._pipeline_primed = False
        self.running = False
        self._last_heartbeat_ts = 0.0
        self._ctx = MagicMock()
        self._sub_control = MagicMock()
        self._push_status = MagicMock()
        self._poller = MagicMock()
        self._control_endpoint = control_endpoint or "tcp://localhost:5557"
        self._status_endpoint = status_endpoint or "tcp://localhost:5558"

    with patch(
        "dexim.core.nodes.managed.ManagedNode.__init__",
        autospec=True,
        side_effect=_fake_managed_init,
    ):
        yield


@pytest.fixture()
def patch_create_model(mock_brainco_model: MagicMock):
    """Patch ``create_model`` in node.py to return ``mock_brainco_model``.

    Yields:
        The ``mock_brainco_model`` fixture value (for assertions).
    """
    with patch(
        "dexim.brainco.node.node.create_model",
        return_value=mock_brainco_model,
    ):
        yield mock_brainco_model


@pytest.fixture()
def mock_vector_optimizer():
    """A mock VectorOptimizer standing in for the NLopt-based optimizer.

    Returns:
        MagicMock with ``alpha`` (list of 5 floats), ``retarget()`` returning
        ``(q, nlopt_result)`` where ``nlopt_result`` is a SimpleNamespace
        with ``.value`` and ``.name`` attributes.
    """
    opt = MagicMock()
    opt.alpha = [1.0, 1.0, 1.0, 1.0, 1.0]

    def _retarget(features):
        q = np.zeros(6, dtype=np.float64)
        result = SimpleNamespace(value=1, name="SUCCESS")
        return q, result

    opt.retarget.side_effect = _retarget
    return opt


@pytest.fixture()
def patch_vector_optimizer(mock_vector_optimizer: MagicMock):
    """Patch ``VectorOptimizer`` in node.py to return ``mock_vector_optimizer``.

    Yields:
        The ``mock_vector_optimizer`` fixture value (for assertions).
    """
    with patch(
        "dexim.brainco.node.node.VectorOptimizer",
        return_value=mock_vector_optimizer,
    ):
        yield mock_vector_optimizer


# ---------------------------------------------------------------------------
# Composed factory fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_brainco_node(
    patch_zmq,
    patch_managed_node,
    patch_create_model,
    patch_vector_optimizer,
    mock_subscriber: Mock,
):
    """Factory fixture that creates a fully-patched ``BrainCoControlNode``.

    All external dependencies are mocked:

    - ZMQ PUB socket setup -> no-op stub (``patch_zmq``)
    - ``ManagedNode.__init__`` ZMQ init -> minimal attribute stub
      (``patch_managed_node``)
    - ``create_model`` -> ``mock_brainco_model`` (``patch_create_model``)
    - Visualization disabled via ``VizConfig(enabled=False)`` in default config

    Yields:
        ``make_node`` callable.  Signature::

            make_node(
                config: BrainCoNodeConfig | None = None,
                subscriber: Any | None = None,
            ) -> BrainCoControlNode

        Pass a custom ``config`` or ``subscriber`` to override defaults.
    """
    from dexim.brainco.node.node import BrainCoControlNode

    def _make(
        config: BrainCoNodeConfig | None = None,
        subscriber: object | None = None,
    ) -> BrainCoControlNode:
        cfg = config or BrainCoNodeConfig(viz=VizConfig(enabled=False))
        sub = subscriber if subscriber is not None else mock_subscriber
        return BrainCoControlNode(
            node_id="test_brainco",
            config=cfg,
            subscriber=sub,
        )

    yield _make
