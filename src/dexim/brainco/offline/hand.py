"""Validated hand-local geometric retargeting and deterministic replay input.

These are bounded geometric approximations, not native IK or physical accuracy
claims. The explicit 25-point layout is root, thumb(4), index/middle/ring/little(5).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from dexim.core.messages.types import HandState, SkeletonJoint

from .config import OfflineConfig
from .model import describe_model

FINGERS = (slice(1, 5), slice(5, 10), slice(10, 15), slice(15, 20), slice(20, 25))


def normalize_hand(hand: HandState, config: OfflineConfig) -> np.ndarray:
    """Validate complete identity and normalize using root quaternion (wxyz)."""
    if (
        hand.side != config.side
        or hand.glove_id != config.glove_id
        or type(hand.glove_id) is not int
        or isinstance(hand.timestamp, bool)
        or not isinstance(hand.timestamp, (int, float))
        or not math.isfinite(hand.timestamp)
        or hand.timestamp < 0
    ):
        raise ValueError("Hand identity/timestamp mismatch")
    if (
        any(type(j.id) is not int for j in hand.joints)
        or tuple(j.id for j in hand.joints) != config.joint_ids
    ):
        raise ValueError("Hand joint IDs must be complete, unique and ordered")
    positions = np.array([j.position for j in hand.joints], dtype=float)
    rotations = np.array([j.rotation for j in hand.joints], dtype=float)
    if (
        positions.shape != (25, 3)
        or rotations.shape != (25, 4)
        or not np.isfinite(positions).all()
        or not np.isfinite(rotations).all()
        or not np.allclose(np.linalg.norm(rotations, axis=1), 1.0, atol=1e-3)
    ):
        raise ValueError("Require finite positions and unit wxyz quaternions")
    w, x, y, z = rotations[0] / np.linalg.norm(rotations[0])
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    local = (positions - positions[0]) @ rotation
    if config.side == "right":
        local[:, 0] *= -1
    for finger in FINGERS:
        lengths = np.linalg.norm(np.diff(local[finger], axis=0), axis=1)
        if (lengths < 1e-5).any() or (lengths > 0.15).any():
            raise ValueError("Degenerate/discontinuous finger geometry")
    return local


def retarget_hand(hand: HandState, config: OfflineConfig) -> np.ndarray:
    """Map curl, spread and thumb opposition into model-specific bounded axes."""
    local = normalize_hand(hand, config)
    descriptor = describe_model(config.model, config.side)
    curls, spreads, flexions = [], [], []
    for finger in FINGERS:
        points = local[finger]
        segments = np.diff(points, axis=0)
        unit = segments / np.linalg.norm(segments, axis=1)[:, None]
        angles = np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1))
        # Base flexion uses the root-local palm plane, then PIP/DIP turns.
        base = abs(float(np.arctan2(unit[0, 2], np.linalg.norm(unit[0, :2]))))
        # Four-finger segment 0 is the metacarpal palm reference; MCP,
        # PIP and DIP are its three consecutive turns. Thumb has 3 segments.
        flexions.append(
            tuple(float(a) for a in angles)
            if len(angles) == 3
            else (base, float(angles[0]), float(angles[-1]))
        )
        curls.append(float(np.mean(angles)))
        spreads.append(float(np.arctan2(segments[0, 0], segments[0, 1])))
    opposition = abs(float(np.arctan2(local[4, 2], local[4, 1])))
    values = []
    for name in descriptor.joint_names:
        finger = next(
            i
            for i, f in enumerate(("thumb", "index", "middle", "ring", "little"))
            if f in name or (f == "little" and "pinky" in name)
        )
        if "metacarpal" in name or "CMP" in name:
            value = opposition
        elif "CMR" in name:
            value = abs(spreads[finger])
        elif "MPR" in name:
            value = spreads[finger]
        elif config.model == "revo3":
            axis = name.rsplit("_", 2)[1]
            value = flexions[finger][("MCP", "PIP", "DIP").index(axis)]
        else:
            value = curls[finger]
        values.append(value)
    return np.clip(values, descriptor.lower, descriptor.upper)


def synthetic_hand(
    config: OfflineConfig, timestamp: float, pose: str = "open"
) -> HandState:
    """Return repeatable core HandState input with Unix capture time."""
    if pose not in ("open", "fist", "pinch", "spread", "opposition"):
        raise ValueError("Unknown synthetic pose")
    points = [np.zeros(3)]
    for finger, section in enumerate(FINGERS):
        count = section.stop - section.start
        curl = 0.9 if pose == "fist" or (pose == "pinch" and finger < 2) else 0.0
        spread = (finger - 2) * 0.12 if pose == "spread" else 0.0
        point = np.array([(finger - 2) * 0.02, 0.025, 0.0])
        points.append(point.copy())
        for j in range(count - 1):
            angle = curl * j
            point = point + 0.025 * np.array(
                [
                    np.sin(spread) * np.cos(angle),
                    np.cos(spread) * np.cos(angle),
                    -np.sin(angle),
                ]
            )
            if finger == 0 and pose == "opposition":
                point[2] += 0.015
            points.append(point.copy())
    if config.side == "right":
        points = [p * [-1, 1, 1] for p in points]
    return HandState(
        config.glove_id,
        config.side,
        timestamp,
        [
            SkeletonJoint(i, tuple(p), (1.0, 0.0, 0.0, 0.0))
            for i, p in zip(config.joint_ids, points)
        ],
    )


def _load_replay(path: str | Path, config: OfflineConfig) -> tuple[HandState, ...]:
    """Validate every record before launch; preserve recorded Unix timestamps."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("Replay must be a nonempty HandState JSON array")
    result = []
    previous = -math.inf
    for row in data:
        # Do not use permissive core from_dict coercion at this trust boundary.
        hand = HandState(
            row["glove_id"],
            row["side"],
            row["timestamp"],
            [
                SkeletonJoint(j["id"], tuple(j["position"]), tuple(j["rotation"]))
                for j in row["joints"]
            ],
        )
        normalize_hand(hand, config)
        if hand.timestamp <= previous:
            raise ValueError("Replay capture timestamps must strictly progress")
        previous = hand.timestamp
        result.append(hand)
    return tuple(result)


def load_replay(path: str | Path, config: OfflineConfig) -> tuple[HandState, ...]:
    """Read validated replay, translating structural failures at the file boundary."""
    try:
        return _load_replay(path, config)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError(f"Malformed HandState replay: {exc}") from exc
