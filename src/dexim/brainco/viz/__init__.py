"""
dexim.brainco.viz -- 3D visualization for BrainCo Revo2 dexterous hand.

Provides:
- ``BrainCoRenderer``: pure Viser rendering engine.
- ``VizSubscriber``: standalone ZMQ-subscriber-driven visualizer.
- ``ActionMessage``: deserialized action message type for the subscriber.
- ``create_session_renderer``: factory for the dexim-visualizer registry.

Depends on dexim.brainco.model for kinematics data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin
import trimesh
from loguru import logger

from dexim.brainco.viz.renderer import BrainCoRenderer
from dexim.brainco.viz.subscriber import ActionMessage, VizSubscriber

__all__ = [
    "ActionMessage",
    "BrainCoRenderer",
    "VizSubscriber",
    "create_session_renderer",
]

# ---------------------------------------------------------------------------
# Offsets for placing independent hand visuals on the ground
# ---------------------------------------------------------------------------

HAND_SIDE_Y_OFFSET_M: dict[str, float] = {"left": 0.35, "right": -0.35}


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


def create_session_renderer(
    server: Any,
    config_path: Path,
    node_id: str,
    scene_root: str,
) -> BrainCoRenderer:
    """Factory for the session-level visualizer's renderer registry.

    Reads the BrainCo config, builds the model, computes the ground-aligned
    base position, and returns a configured ``BrainCoRenderer``.

    Args:
        server: Existing ``viser.ViserServer`` instance (shared session scene).
        config_path: Absolute path to the BrainCo node YAML config.
        node_id: Node identifier, e.g. ``"brainco_left"``.
        scene_root: Viser scene path prefix, e.g. ``"/robots/brainco_left"``.

    Returns:
        Fully configured ``BrainCoRenderer`` ready for ``update()`` calls.
    """
    from dexim.brainco.model.factory import create_model
    from dexim.brainco.node.config import load_config

    cfg = load_config(str(config_path))
    model = create_model(hand_side=cfg.brainco.side)
    base_position = _ground_aligned_hand_base_position(model, cfg.brainco.side)

    return BrainCoRenderer(
        model=model,
        server=server,
        scene_root=scene_root,
        add_grid=False,
        verbose=False,
        show_frames=cfg.viz.show_frames,
        show_geometry=cfg.viz.show_geometry,
        show_ee_spheres=cfg.viz.show_ee_spheres,
        base_position=base_position,
    )


# ---------------------------------------------------------------------------
# Ground-alignment helpers
# ---------------------------------------------------------------------------


def _ground_aligned_hand_base_position(model: Any, side: str) -> np.ndarray:
    """Place independent hand visuals on the ground without changing robot configs."""
    min_z = _visual_geometry_min_z(model)
    return np.array(
        [0.0, HAND_SIDE_Y_OFFSET_M.get(side, 0.0), -min_z],
        dtype=np.float64,
    )


def _visual_geometry_min_z(model: Any) -> float:
    """Return the lowest visual-geometry Z coordinate for a neutral hand pose."""
    q = pin.neutral(model.model)
    model.compute_forward_kinematics(q)

    min_z: float | None = None
    geometry_objects = model.geometry_model.geometryObjects
    for idx, geom_obj in enumerate(geometry_objects):
        placement = model.geometry_data.oMg[idx]
        mesh_min_z = _mesh_world_min_z(geom_obj, placement)
        if mesh_min_z is None:
            mesh_min_z = float(placement.translation[2])
        min_z = mesh_min_z if min_z is None else min(min_z, mesh_min_z)

    return min_z if min_z is not None else 0.0


def _mesh_world_min_z(geom_obj: Any, placement: Any) -> float | None:
    mesh_path = getattr(geom_obj, "meshPath", "")
    if not mesh_path:
        return None
    if mesh_path.startswith("file://"):
        mesh_path = mesh_path[7:]

    try:
        mesh = trimesh.load_mesh(mesh_path, process=False)
    except Exception as exc:
        logger.debug(f"failed to load hand mesh for ground alignment: {exc}")
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        return None

    mesh_scale = getattr(geom_obj, "meshScale", None)
    if mesh_scale is not None:
        vertices = vertices * np.asarray(mesh_scale, dtype=np.float64)

    transformed = vertices @ placement.rotation.T + placement.translation
    return float(np.min(transformed[:, 2]))
