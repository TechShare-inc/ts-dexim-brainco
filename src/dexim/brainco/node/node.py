"""BrainCo Revo2 hand control node -- collaborator-composed orchestrator.

``BrainCoControlNode`` owns typed collaborator objects for each pipeline
stage instead of inheriting from multiple mixins.

The data-flow pipeline each ``_run_pipeline()`` call:

1. ``_receiver.receive()``                            -> HandState
2. ``_extractor.extract(state)``                      -> (N, 3) vectors
3. ``_retargeter.scale(vectors)``                     -> scaled vectors
4. ``_retargeter.retarget(scaled)``                   -> q
5. ``_joint_filter(q)``                               -> q_smooth
6. ``_motion_controller.apply_velocity_limits(q_smooth)`` -> q_safe
7. ``_motion_controller.send(q_safe)``
8. ``_publisher.publish_action(q_safe, extra_data={"features": ...})``
9. ``_publisher.publish_observation(hw_state)``
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from dexim.core.collaborators import (
    DataPlanePublisher,
    JointFilter,
    MotionController,
    PipelineProfiler,
    SkeletonReceiver,
)
from dexim.core.nodes import PubSubDeviceNode
from dexim.core.robot_interface import RobotInterface
from loguru import logger

from dexim.brainco.model import VectorOptimizer, create_model

from .config import BrainCoNodeConfig, SubscriberProtocol
from .extractor import FeatureExtractor
from .retargeter import Retargeter

# Active-joint URDF indices for extracting the 6 commanded DOF from the
# full 11-DOF URDF model output produced by VectorOptimizer.retarget().
from dexim.brainco.interface._conversion import (
    NUM_JOINTS,
    URDF_ACTIVE_JOINT_INDICES,
)

_FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")


class BrainCoControlNode(PubSubDeviceNode):
    """Control node for BrainCo Revo2 dexterous hand (collaborator composition).

    Attributes:
        config: ``BrainCoNodeConfig`` instance.
        interface: RobotInterface (mock / modbus_rs485).
        model: ``BrainCoModel`` for kinematics.
    """

    _HEALTH_CHECK_INTERVAL: int = 30  # iterations between is_connected() polls

    config: BrainCoNodeConfig
    interface: RobotInterface
    model: "BrainCoModel"  # noqa: F821

    def __init__(
        self,
        node_id: str,
        config: BrainCoNodeConfig,
        subscriber: SubscriberProtocol,
    ) -> None:
        """Initialise the node and all collaborators.

        Args:
            node_id: Unique node identifier.
            config: Full node configuration.
            subscriber: Data subscriber (e.g. TopicSubscriber[HandState]).
        """
        cfg = config
        bcfg = cfg.brainco
        side: Literal["left", "right"] = bcfg.side  # type: ignore[assignment]

        self.config = config

        super().__init__(node_id=node_id)

        # Model + optimizer (Pinocchio-based kinematics + NLopt vector retargeting)
        self.model = create_model(hand_side=side)
        optimizer = VectorOptimizer(self.model)

        # Interface -- mock/hw modes route through BrainCoInterface
        # which spawns P1 (hw-core) on connect().
        from dexim.brainco.interface.factory import build_interface

        self.interface = build_interface(
            cfg.interface,
            model=self.model,
            hand_side=side,
        )

        # Collaborators -- constructed in data-flow pipeline order.
        self._publisher = DataPlanePublisher(
            node_id=self.node_id,
            data_endpoint=cfg.data_endpoint,
            bind=cfg.bind_data,
        )
        self._receiver = SkeletonReceiver(
            subscriber=subscriber,
            side=side,
            glove_id=None,  # BrainCo does not filter by Manus glove ID
        )
        self._motion_controller = MotionController(
            interface=self.interface,
            rate_hz=cfg.control.rate_hz,
            enable_velocity_limiting=cfg.control.enable_velocity_limiting,
            max_joint_velocity_rad_s=cfg.control.max_joint_velocity_rad_s,
            max_joint_velocity_per_joint=cfg.control.max_joint_velocity_rad_s_per_joint,
            safe_position_on_timeout=cfg.control.safe_position_on_timeout,
            timeout_sec=cfg.control.timeout_sec,
            safe_position_max_velocity_rad_s=cfg.control.safe_position_max_velocity_rad_s,
        )
        self._extractor = FeatureExtractor(bcfg.feature_extraction)
        self._retargeter = Retargeter(optimizer, alpha=bcfg.alpha)
        self._joint_filter = JointFilter(bcfg.filter, data_size=NUM_JOINTS)
        self._profiler = PipelineProfiler()
        self._health_check_counter: int = 0

        # Countdown duration from config (used by ManagedNode on CTRL_START)
        self._countdown_duration = cfg.control.start_countdown_sec

        logger.success("BrainCoControlNode ready")

    # ------------------------------------------------------------------
    # Control pipeline
    # ------------------------------------------------------------------
    def _run_pipeline(self) -> None:
        """Run one iteration of the control pipeline."""
        # Periodic hardware-core health check.
        self._health_check_counter += 1
        if self._health_check_counter % self._HEALTH_CHECK_INTERVAL == 0:
            if not self.interface.is_connected():
                logger.warning(
                    f"[{self.node_id}] Hardware Core is no longer alive "
                    f"-- interface reports disconnected"
                )

        if not self._teleop_active:
            self._pipeline_primed = False
            self._receiver.drain()
            # Still read hardware state and publish observation so robot
            # state is visible even before teleop is started.
            hw_state = self._motion_controller.read_state()
            self._publisher.publish_observation(hw_state)
            self._motion_controller.sleep()
            return

        # Prime the timeout clock on the first tick after teleop becomes
        # active so the start countdown (which may run for several seconds
        # after initialize() set _last_data_time) does not falsely trigger
        # an immediate data timeout.
        if not self._pipeline_primed:
            self._motion_controller.record_data_received()
            self._pipeline_primed = True

        # 1. Receive
        with self._profiler.stage("recv"):
            state = self._receiver.receive()

        # 2. Timeout check
        if self._motion_controller.check_timeout():
            logger.warning("Data timeout -- moving to safe position")
            self._motion_controller.move_to_safe(self.get_safe_position())
            self._motion_controller.record_data_received()
        elif state is not None:
            # 3-7. Extract -> scale -> retarget -> filter -> vel-limit -> send
            with self._profiler.stage("extract"):
                features = self._extractor.extract(state)
            if features is not None:
                with self._profiler.stage("retarget"):
                    scaled = self._retargeter.scale(features)
                    q = self._retargeter.retarget(scaled)
                if q is not None:
                    # Extract only the 6 actively-controlled joints from
                    # the full 11-DOF URDF-model output.
                    q = q[URDF_ACTIVE_JOINT_INDICES]
                    with self._profiler.stage("filter"):
                        q_smooth = self._joint_filter(q)
                    with self._profiler.stage("vel_lim"):
                        q_safe = self._motion_controller.apply_velocity_limits(q_smooth)
                    with self._profiler.stage("send_cmd"):
                        self._motion_controller.send(q_safe)
                    self._motion_controller.record_data_received()
                    # 8. Publish action (includes scaled feature vectors)
                    with self._profiler.stage("pub_action"):
                        features_dict = {
                            n: scaled[i].tolist()
                            for i, n in enumerate(_FINGER_NAMES[: len(scaled)])
                        }
                        self._publisher.publish_action(
                            q_safe, extra_data={"features": features_dict}
                        )

        # 9. Read hardware state and publish observation.
        # interface.read() is a lock-free SHM read (microsecond).
        with self._profiler.stage("hw_read"):
            hw_state = self._motion_controller.read_state()
        with self._profiler.stage("pub_obs"):
            self._publisher.publish_observation(hw_state)

        timing = self._motion_controller.sleep()
        if timing["overtime"]:
            logger.warning(
                f"[ctrl] Loop overtime: elapsed={timing['elapsed'] * 1000:.1f}ms  "
                f"target={self._motion_controller.dt * 1000:.1f}ms  "
                f"late={timing['jitter'] * 1000:.1f}ms"
            )
        self._profiler.report_if_due(
            self.node_id,
            self._motion_controller.rate_limiter,
            data_age_sec=self._receiver.last_data_age_sec,
        )

    # ------------------------------------------------------------------
    # Auto-prepare (called each tick during STANDBY)
    # ------------------------------------------------------------------

    def _auto_prepare(self) -> None:
        """During STANDBY, log readiness when skeleton data is flowing."""
        if getattr(self, "_skeleton_ready", False):
            return
        try:
            data_age = self._receiver.last_data_age_sec
        except Exception:
            data_age = float("inf")

        if data_age is not None and data_age < 5.0:
            self._skeleton_ready = True
            logger.info(
                f"{self.node_id} Skeleton data flowing "
                f"(age={data_age:.1f}s) -- ready for START"
            )

    # ------------------------------------------------------------------
    # Safe position
    # ------------------------------------------------------------------
    def get_safe_position(self) -> np.ndarray:
        """Open-hand position (zeros = fully open)."""
        return np.zeros(NUM_JOINTS)

    # ------------------------------------------------------------------
    # ManagedNode lifecycle hooks
    # ------------------------------------------------------------------
    def on_standby(self) -> None:
        """Enter STANDBY: connect interface, move to safe home, then publish.

        ``DeviceNode.on_standby()`` handles ``interface.connect()`` first,
        so that ``MotionController.initialize()`` can read the initial
        joint state without a "not connected" error.

        After initialization the hand is moved to the safe open-hand
        position (all zeros) so that the visualizer and any monitoring
        system see the safe home pose rather than whatever position the
        hand was last at.

        The publisher is then activated so observations are streamed even
        before teleop is started (e.g., for monitoring/visualization).
        """
        logger.info(f"{self.node_id} STANDBY")
        super().on_standby()
        self._motion_controller.initialize()
        # Move to safe open-hand position so the visualizer shows the hand
        # at its canonical safe pose during standby.
        self._motion_controller.move_to_safe(self.get_safe_position())
        # Activate publisher so observations are published even before
        # teleop is started (e.g., for monitoring/visualization).
        self._publisher.activate()

    def on_start(self) -> None:
        logger.info(f"{self.node_id} START")
        self.interface.connect()
        self._motion_controller.initialize()
        self._publisher.activate()
        super().on_start()

    def on_pause(self) -> None:
        logger.info(f"{self.node_id} PAUSE")
        self._publisher.deactivate()
        super().on_pause()

    def on_stop(self) -> None:
        logger.info(f"{self.node_id} STOP -- safe position")
        self._publisher.deactivate()
        try:
            self._motion_controller.move_to_safe(self.get_safe_position())
        except Exception as exc:
            logger.error(f"Error on stop: {exc}")
        try:
            self.interface.disconnect()
        except Exception as exc:
            logger.error(f"Error disconnecting on stop: {exc}")
        super().on_stop()

    def on_start_recording(self) -> None:
        """Handle start-recording command (not yet implemented)."""
        logger.info(f"{self.node_id} START_RECORDING (not implemented)")

    def on_stop_recording(self) -> None:
        """Handle stop-recording command (not yet implemented)."""
        logger.info(f"{self.node_id} STOP_RECORDING (not implemented)")

    def on_shutdown(self) -> None:
        logger.info(f"{self.node_id} SHUTDOWN")
        self._publisher.close()
        self._receiver.close()
        try:
            self.interface.disconnect()
        except Exception:
            pass
        super().on_shutdown()
