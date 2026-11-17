"""SDK-free named joint metadata, from description revision 324cd0c5.

Numeric joint facts only; assets remain external and are not redistributed.
Revo3 SDK aliases/signs/zeros are unqualified for physical output.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


@dataclass(frozen=True)
class ModelDescriptor:
    """Canonical URDF-order independent joint metadata in radians."""

    model: str
    side: str
    joint_names: tuple[str, ...]
    full_joint_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    sdk_order: tuple[int, ...]
    mimic: tuple[tuple[int, float, float], ...]

    def vector(self, values: object) -> np.ndarray:
        """Validate and copy an independent finite vector."""
        q = np.array(values, dtype=float, copy=True)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            raise ValueError("Expected finite independent joint vector")
        return q

    def expand(self, values: object) -> tuple[float, ...]:
        """Expand Revo2 mechanical coupling as derived positions."""
        q = self.vector(values)
        return tuple(float(q[i] * scale + offset) for i, scale, offset in self.mimic)

    def to_sdk(self, values: object) -> np.ndarray:
        """Convert to explicit SDK layout; Revo2 normalized, Revo3 degrees.

        Revo3 permutation is a provisional anatomical alias for fake tests only.
        Physical signs/zeros remain qualification prerequisites.
        """
        q = self.vector(values)
        if (q < self.lower).any() or (q > self.upper).any():
            raise ValueError("Joint bounds exceeded")
        if self.model == "revo2":
            q = (q - self.lower) / (np.array(self.upper) - self.lower) * 1000
        else:
            q = np.rad2deg(q)
        return q[list(self.sdk_order)]

    def from_sdk(self, values: object) -> np.ndarray:
        """Inverse SDK position conversion, rejecting malformed readback."""
        native = self.vector(values)
        q = np.empty_like(native)
        q[list(self.sdk_order)] = native
        if self.model == "revo2":
            if (q < 0).any() or (q > 1000).any():
                raise ValueError("Invalid normalized readback")
            q = np.array(self.lower) + q / 1000 * (np.array(self.upper) - self.lower)
        else:
            q = np.deg2rad(q)
        if (q < np.array(self.lower) - 1e-9).any() or (
            q > np.array(self.upper) + 1e-9
        ).any():
            raise ValueError("Readback outside model bounds")
        return q

    def validate_assets(self, path: str | Path, *, require_meshes: bool = True) -> Path:
        """Validate supplied URDF joint identity and mesh presence."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("Supply a licensed official URDF asset path")
        expected = json.loads(Path(__file__).with_name("asset_hashes.json").read_text())
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != expected[f"{self.model}_{self.side}"]
        ):
            raise ValueError("Asset SHA256 differs from description 324cd0c5")
        root = ElementTree.parse(path).getroot()
        names = tuple(
            j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"
        )
        if names != self.full_joint_names:
            raise ValueError("Asset model/side/joint layout mismatch")
        for mesh in root.findall(".//mesh") if require_meshes else ():
            filename = mesh.get("filename", "")
            if filename.startswith("package://"):
                raise ValueError("Resolve package:// meshes to supplied local assets")
            if not (path.parent / filename).is_file():
                raise FileNotFoundError(f"Missing external mesh: {filename}")
        return path


def describe_model(model: str = "revo2", side: str = "left") -> ModelDescriptor:
    """Return explicit side-specific offline metadata without native imports."""
    if model not in ("revo2", "revo3") or side not in ("left", "right"):
        raise ValueError("model must be revo2|revo3; side must be left|right")
    rows = json.loads(Path(__file__).with_name("joints.json").read_text())[
        f"{model}_{side}"
    ]
    active = [r for r in rows if r["mimic"] is None]
    names = tuple(r["name"] for r in active)
    if model == "revo2":
        suffixes = (
            "thumb_proximal",
            "thumb_metacarpal",
            "index_proximal",
            "middle_proximal",
            "ring_proximal",
            "pinky_proximal",
        )
    else:
        suffixes = tuple(
            f"{f}_{j}"
            for f in ("little", "ring", "middle", "index")
            for j in ("MPR", "MCP", "PIP", "DIP")
        ) + tuple(f"thumb_{j}" for j in ("MCP", "PIP", "DIP", "CMP", "CMR"))
    order = tuple(names.index(f"{side}_{s}_joint") for s in suffixes)
    mimic = []
    for row in rows:
        m = row["mimic"]
        mimic.append(
            (
                names.index(m["joint"] if m else row["name"]),
                float(m.get("multiplier", 1)) if m else 1.0,
                float(m.get("offset", 0)) if m else 0.0,
            )
        )
    return ModelDescriptor(
        model,
        side,
        names,
        tuple(r["name"] for r in rows),
        tuple(r["lower"] for r in active),
        tuple(r["upper"] for r in active),
        order,
        tuple(mimic),
    )
