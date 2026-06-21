"""VizSubscriber -- live ZMQ subscriber that drives BrainCoRenderer.

Started independently via ``dexim-brainco viz``; subscribes to the node's
data plane to receive action messages (joint positions) and renders them in
real time.

Typical usage::

    from dexim.brainco.model import create_model
    from dexim.brainco.viz import BrainCoRenderer, VizSubscriber

    model = create_model(hand_side="left")
    renderer = BrainCoRenderer(model=model, port=8080)
    viewer = VizSubscriber(
        renderer=renderer,
        endpoint="tcp://localhost:5556",
        node_id="brainco_left",
    )
    viewer.run()  # blocks until Ctrl-C
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class ActionMessage:
    """Deserialized action message from the node data plane.

    Args:
        q: Joint positions as a list of floats.
    """

    q: list[float]

    @classmethod
    def from_dict(cls, d: Any) -> ActionMessage:
        """Deserialize from a raw ZMQ payload.

        Handles both the current dict format ``{"q": [...]}`` and the
        legacy bare-list format for backwards compatibility.

        Args:
            d: Raw payload from the ZMQ message (dict or list).

        Returns:
            ActionMessage instance.
        """
        if isinstance(d, dict):
            return cls(q=d["q"])
        # Legacy: bare float list.
        return cls(q=list(d))


class VizSubscriber:
    """Subscribes to a node's ZMQ data plane and drives a BrainCoRenderer.

    Connects to the ``action/{node_id}/joint_cmd`` topic on the node's PUB
    socket and polls for fresh action messages.  Received joint positions
    are stored as interpolation keyframes, and the render loop runs at
    *fps* (default 60 FPS) with linear interpolation between keyframes
    for smooth visualisation.

    The visualizer is completely independent of the control node -- it can
    start and stop freely without affecting the control loop.

    Args:
        renderer: Fully-initialised :class:`~dexim.brainco.viz.BrainCoRenderer`
            instance to drive.
        endpoint: ZMQ endpoint of the node's data plane PUB socket
            (e.g. ``"tcp://localhost:5556"``).
        node_id: Node identifier used to build the topic filter
            (e.g. ``"brainco_left"``).
        fps: Render loop frame rate; interpolates between received keyframes.
            Default 60.
        receive_fps: Expected incoming data rate in Hz, used for logging
            only.  Default 30.
    """

    def __init__(
        self,
        renderer: Any,
        endpoint: str,
        node_id: str,
        fps: float = 60.0,
        receive_fps: float = 30.0,
    ) -> None:
        from dexim.core.messages import TopicBuilder
        from dexim.core.nodes import TopicSubscriber

        self._renderer = renderer
        self._fps = fps
        self._receive_fps = receive_fps
        self._stop = False

        # Interpolation state: two most recent keyframes with timestamps.
        self._prev_q: np.ndarray | None = None
        self._current_q: np.ndarray | None = None
        self._prev_t: float = 0.0
        self._current_t: float = 0.0

        action_topic = TopicBuilder().action.joint_cmd(node_id)
        self._action_sub: TopicSubscriber[ActionMessage] = TopicSubscriber(
            address=endpoint,
            topic=action_topic,
            msg_type=ActionMessage,
            timeout_ms=100,
        )
        logger.info(f"VizSubscriber connected to {endpoint} (topic: {action_topic!r})")

    # -- Interpolation -------------------------------------------------------

    def _push_keyframe(
        self,
        q: np.ndarray,
        t: float,
    ) -> None:
        """Push a newly received keyframe into the interpolation buffer.

        Args:
            q: Received joint positions.
            t: Monotonic timestamp of reception (``time.perf_counter()``).
        """
        if self._current_q is not None:
            self._prev_q = self._current_q
            self._prev_t = self._current_t
        self._current_q = q
        self._current_t = t

    def _interpolate_q(self, t: float) -> np.ndarray | None:
        """Interpolate joint positions for the given render timestamp.

        Linearly blends between the two most recent keyframes.  Returns the
        most recent config directly when only one keyframe is available, or
        ``None`` when no data has ever been received.

        Args:
            t: Current render timestamp (``time.perf_counter()``).

        Returns:
            Interpolated joint positions, or ``None``.
        """
        if self._current_q is None:
            return None
        if self._prev_q is None:
            return self._current_q

        dt = self._current_t - self._prev_t
        if dt <= 0.0:
            return self._current_q
        alpha = min((t - self._prev_t) / dt, 1.0)
        alpha = max(alpha, 0.0)
        return (1.0 - alpha) * self._prev_q + alpha * self._current_q

    # -- Main loop -----------------------------------------------------------

    def run(self) -> None:
        """Block in the render loop until :meth:`stop` is called or Ctrl-C.

        Polls the action topic at the configured *fps* (default 60) and
        smoothly interpolates between received keyframes (expected at
        *receive_fps*, default 30) for a jitter-free visualisation.
        """
        period = 1.0 / self._fps

        def _handle_signal(signum: int, frame: Any) -> None:
            logger.debug(f"[viz] received signal {signum}, stopping")
            self.stop()

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        logger.info(
            f"[viz] render loop started: render={self._fps:.0f} FPS, "
            f"receive={self._receive_fps:.0f} Hz"
        )
        while not self._stop:
            loop_start = time.perf_counter()

            # Poll for the latest received keyframe.
            msg = self._action_sub.read_latest()
            if msg is not None:
                self._push_keyframe(
                    np.array(msg.q, dtype=np.float64),
                    loop_start,
                )

            # Build an interpolated render update.
            q_interp = self._interpolate_q(loop_start)
            if q_interp is not None:
                update: dict[str, Any] = {"q": q_interp}
                try:
                    self._renderer.update(update)
                except Exception as exc:
                    logger.debug(f"[viz] renderer update error: {exc}")

            # Rate-limit: sleep for most of the remaining budget, then busy-wait.
            elapsed = time.perf_counter() - loop_start
            remaining = period - elapsed
            if remaining > 0.005:
                time.sleep(remaining - 0.001)
            while time.perf_counter() < loop_start + period:
                pass

    def stop(self) -> None:
        """Signal the render loop to exit on the next iteration."""
        self._stop = True

    def close(self) -> None:
        """Stop the render loop and release ZMQ resources."""
        self.stop()
        try:
            self._action_sub.close()
        except Exception as exc:
            logger.debug(f"[viz] subscriber close error: {exc}")
