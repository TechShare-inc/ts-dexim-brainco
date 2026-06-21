"""
BrainCoRenderer: Viser-based 3D renderer for BrainCo Revo2 dexterous hand.

This module provides a renderer class for BrainCo Revo2 hand robots
using the Viser web-based 3D visualization library.

Revo2 URDF already contains tip frames as fixed joints, so no frame
appending is needed — the renderer uses the URDF's native frame structure.
"""

import os

import numpy as np
import pinocchio as pin
import trimesh
import viser
from loguru import logger
from viser import ViserServer
from viser.transforms import SE3 as ViserSE3
from viser.transforms import SO3 as ViserSO3

from dexim.brainco.model import BrainCoModel


def pin_se3_to_viser_se3(se3: pin.SE3) -> ViserSE3:
    """Convert Pinocchio SE3 to Viser SE3 transform.

    Args:
        se3: Pinocchio SE3 transformation

    Returns:
        Viser SE3 transformation
    """
    from scipy.spatial.transform import Rotation as R

    rotation_matrix = se3.rotation
    translation = se3.translation

    rotation = R.from_matrix(rotation_matrix)
    quat_xyzw = rotation.as_quat()  # scipy returns [x, y, z, w]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    return ViserSE3.from_rotation_and_translation(ViserSO3(wxyz=quat_wxyz), translation)


class BrainCoRenderer:
    """A Viser-based visualizer for BrainCo Revo2 robot models.

    This class provides real-time 3D visualization of BrainCo Revo2 hands
    using Viser, including:
    - Robot geometry visualization with mesh loading
    - End-effector frame displays for finger tips
    - Real-time pose visualization from joint configurations
    - Visibility controls for different visualization components

    Visibility Features:
    - Individual control over frames, geometry, and end-effector spheres
    - Toggle methods for quick on/off switching
    - Properties for getting/setting visibility states
    - Batch show/hide all components

    Method sections:
    - Initialization
    - Scene setup (private)
    - Update
    - State queries
    - Visibility control
    - Lifecycle
    """

    # -- Initialization ------------------------------------------------------

    def __init__(
        self,
        model: BrainCoModel,
        server: ViserServer | None = None,
        host: str = "localhost",
        port: int = 8080,
        verbose: bool = True,
        show_frames: bool = True,
        show_geometry: bool = True,
        show_ee_spheres: bool = True,
        scale: float = 1.0,
        scene_root: str = "",
        add_grid: bool = True,
        base_position: np.ndarray | tuple[float, float, float] | None = None,
    ):
        """Initialize the BrainCo Revo2 robot visualizer.

        Args:
            model: BrainCoModel instance containing the robot model.
            server: Optional existing Viser server instance.
            host: Host for the Viser server (if creating new server).
            port: Port for the Viser server (if creating new server).
            verbose: Whether to print status messages.
            show_frames: Whether to show end-effector frame axes.
            show_geometry: Whether to show robot geometry (links/joints).
            show_ee_spheres: Whether to show end-effector spheres.
            scale: Scale factor to apply to robot visualization.
            scene_root: Optional prefix for all scene node paths.
            add_grid: Whether to add a ground grid.
            base_position: Optional base translation offset.
        """
        self.brainco_model = model
        self.verbose = verbose
        self.scale = scale
        self.scene_root = scene_root.rstrip("/")
        self.base_position = (
            np.asarray(base_position, dtype=np.float64)
            if base_position is not None
            else np.zeros(3, dtype=np.float64)
        )
        self.logger = logger.bind(name=model.__class__.__name__)

        # Visibility flags
        self._show_frames = show_frames
        self._show_geometry = show_geometry
        self._show_ee_spheres = show_ee_spheres

        # Initialize Viser server
        if server is None:
            self.server = viser.ViserServer(host=host, port=port)
            self.owns_server = True
            self._port: int | None = port
        else:
            self.server = server
            self.owns_server = False
            self._port = None

        # Visualization components
        self.joint_frames: list[viser.SceneNodeHandle] = []
        self.frame_handles: dict[str, viser.SceneNodeHandle] = {}
        self.geometry_handles: dict[str, viser.SceneNodeHandle] = {}
        self.meshes: list[viser.SceneNodeHandle] = []

        # Root frames for organized scene hierarchy
        self.visual_root_frame: viser.FrameHandle | None = None
        self.collision_root_frame: viser.FrameHandle | None = None

        # Create geometry data if geometry model exists
        assert model.geometry_model is not None
        self.geometry_model = model.geometry_model
        self.geometry_data = model.geometry_data

        if add_grid:
            self.server.scene.add_grid(name="/grid")

        # Current robot configuration
        self._current_q = pin.neutral(model.model)
        self.brainco_model.compute_forward_kinematics(self._current_q)

        # Initialize visualization components
        self._setup_robot_visualization()

        if self.verbose:
            self.logger.info(f"BrainCoRenderer initialized on port {port}")
            self.logger.info(f"View at: http://localhost:{port}")

    # -- Scene setup (private) -----------------------------------------------

    def _setup_robot_visualization(self) -> None:
        """Set up the complete robot visualization."""
        root_node_name = self._scene_path("/brainco_robot")

        if self._show_geometry:
            self.visual_root_frame = self._create_joint_frames_and_meshes(
                root_node_name, collision_geometry=False
            )

        if self._show_frames or self._show_ee_spheres:
            self._create_frame_visualizations()

    def _create_joint_frames_and_meshes(
        self, root_node_name: str, collision_geometry: bool = False
    ) -> viser.FrameHandle:
        """Create joint frames and meshes for the robot visualization.

        Adapted from the Inspire pattern but using BrainCo model geometry objects.
        """
        prefix = "collision" if collision_geometry else "visual"
        prefixed_root_node_name = f"{root_node_name}/{prefix}".replace("//", "/")
        root_frame = self.server.scene.add_frame(
            prefixed_root_node_name, show_axes=False
        )
        self._create_geometry_objects(prefixed_root_node_name)
        return root_frame

    def _scene_path(self, path: str) -> str:
        """Return a scene path under this renderer's optional root."""
        if not self.scene_root:
            return path
        return f"{self.scene_root}/{path.lstrip('/')}"

    def _scene_position(self, position: np.ndarray) -> np.ndarray:
        """Apply the visualizer-only base translation to a model position."""
        return np.asarray(position, dtype=np.float64) * self.scale + self.base_position

    def _create_trimesh_from_geometry_object(
        self, geom_obj: pin.GeometryObject, color: tuple[float, float, float]
    ) -> trimesh.Trimesh | None:
        """Create a trimesh object from a Pinocchio GeometryObject.

        Args:
            geom_obj: The GeometryObject containing geometry, placement, and metadata.
            color: RGB color tuple for the mesh.

        Returns:
            Trimesh object or None if geometry type is not supported.
        """
        if geom_obj.meshPath:
            try:
                mesh_path = geom_obj.meshPath

                # Handle file:// URLs (from URDF preprocessing)
                if mesh_path.startswith("file://"):
                    mesh_path = mesh_path[7:]

                if not os.path.exists(mesh_path):
                    if self.verbose:
                        self.logger.warning(
                            f"Mesh file not found: {mesh_path} for object: {geom_obj.name}"
                        )
                    return None

                mesh = trimesh.load_mesh(mesh_path)

                if geom_obj.meshScale is not None:
                    scale_factors = geom_obj.meshScale
                    if len(scale_factors) != 3:
                        raise ValueError(
                            f"meshScale should have 3 elements, got {len(scale_factors)}"
                        )
                    mesh.apply_scale(scale_factors)

                if self.verbose:
                    self.logger.info(
                        f"Successfully loaded mesh from {mesh_path} for object: {geom_obj.name}"
                    )

            except Exception as e:
                if self.verbose:
                    self.logger.warning(
                        f"Failed to load mesh from {geom_obj.meshPath} for object {geom_obj.name}: {e}"
                    )
                return None
        else:
            if self.verbose:
                self.logger.warning(
                    f"Unsupported geometry type: {type(geom_obj.geometry)} for object: {geom_obj.name}"
                )
            return None

        if mesh.visual is not None:
            final_color = color

            if geom_obj.overrideMaterial and geom_obj.meshColor is not None:
                mesh_color = geom_obj.meshColor
                if len(mesh_color) >= 3:
                    final_color = tuple(mesh_color[:3])

            from trimesh.visual import ColorVisuals

            mesh.visual = ColorVisuals(
                mesh=mesh, face_colors=[[*final_color, 255]] * mesh.faces.shape[0]
            )

        return mesh

    def _create_geometry_objects(self, root_node_name: str) -> None:
        """Create visual representations of robot geometry objects.

        Args:
            root_node_name: Scene node path prefix for geometry objects.
        """
        default_color = (0.7, 0.7, 0.9)
        # Per-link colors for visual distinction
        link_colors = [
            (0.8, 0.2, 0.2),  # Red
            (0.2, 0.8, 0.2),  # Green
            (0.2, 0.2, 0.8),  # Blue
            (0.8, 0.8, 0.2),  # Yellow
            (0.8, 0.2, 0.8),  # Magenta
            (0.2, 0.8, 0.8),  # Cyan
        ]

        for i, geom_obj in enumerate(self.brainco_model.geometry_model.geometryObjects):
            try:
                pin_tf = self.geometry_data.oMg[i]
                viser_tf = pin_se3_to_viser_se3(pin_tf)

                color = (
                    link_colors[i % len(link_colors)]
                    if i < len(link_colors)
                    else default_color
                )

                mesh = self._create_trimesh_from_geometry_object(geom_obj, color)

                if mesh is not None:
                    handle = self.server.scene.add_mesh_trimesh(
                        name=f"{root_node_name}/geom_{geom_obj.name}",
                        mesh=mesh,
                        position=self._scene_position(viser_tf.translation()),
                        wxyz=viser_tf.rotation().wxyz,
                        visible=True,
                    )
                else:
                    self.logger.warning(
                        f"Skipping geometry creation for {geom_obj.name} due to "
                        f"unsupported type or load failure."
                    )
                    continue

                self.geometry_handles[geom_obj.name] = handle
                self.meshes.append(handle)

                if self.verbose:
                    self.logger.info(f"Created geometry: {geom_obj.name}")

            except Exception as e:
                if self.verbose:
                    self.logger.exception(
                        f"Failed to create geometry for {geom_obj.name}: {e}"
                    )

    def _create_frame_visualizations(self) -> None:
        """Create frame visualizations for end-effectors and joints."""
        self.brainco_model.compute_forward_kinematics(self._current_q)
        self._create_ee_frame_visualizations()
        self._create_joint_frame_visualizations()

    def _create_ee_frame_visualizations(self) -> None:
        """Create frame axes and spheres for each tip (end-effector) frame."""
        ee_colors = [
            (0.8, 0.2, 0.2),  # Red    — thumb
            (0.2, 0.8, 0.2),  # Green  — index
            (0.2, 0.2, 0.8),  # Blue   — middle
            (0.8, 0.8, 0.2),  # Yellow — ring
            (0.8, 0.2, 0.8),  # Magenta— pinky
        ]
        default_color = (0.7, 0.7, 0.9)

        for i, frame_name in enumerate(self.brainco_model.tip_frames.keys()):
            try:
                pin_tf = self.brainco_model.get_frame_pose(frame_name)
                viser_tf = pin_se3_to_viser_se3(pin_tf)

                position = self._scene_position(viser_tf.translation())
                wxyz = viser_tf.rotation().wxyz
                color = (
                    ee_colors[i % len(ee_colors)]
                    if i < len(ee_colors)
                    else default_color
                )

                if self._show_frames:
                    handle = self.server.scene.add_frame(
                        name=self._scene_path(f"/frames/ee_frames/{frame_name}"),
                        axes_length=0.05 * self.scale,
                        axes_radius=0.001 * self.scale,
                        position=position,
                        wxyz=wxyz,
                        visible=True,
                    )
                    self.frame_handles[frame_name] = handle

                if self._show_ee_spheres:
                    handle = self.server.scene.add_icosphere(
                        name=self._scene_path(f"/ee_spheres/sphere_{frame_name}"),
                        radius=0.004 * self.scale,
                        opacity=0.6,
                        position=position,
                        wxyz=wxyz,
                        color=color,
                    )
                    self.frame_handles[f"sphere_{frame_name}"] = handle

                if self.verbose:
                    components = []
                    if self._show_frames:
                        components.append("frame")
                    if self._show_ee_spheres:
                        components.append("sphere")
                    if components:
                        self.logger.info(
                            f"Created {'/'.join(components)} visualization: {frame_name}"
                        )

            except Exception as e:
                if self.verbose:
                    self.logger.exception(
                        f"Failed to create frame visualization for {frame_name}: {e}"
                    )

    def _create_joint_frame_visualizations(self) -> None:
        """Create frame axes for all non-tip joint frames."""
        remaining_frames = [
            frame_name
            for frame_name in self.brainco_model.valid_frames.keys()
            if frame_name not in self.brainco_model.tip_frames.keys()
        ]

        for frame_name in remaining_frames:
            try:
                pin_tf = self.brainco_model.get_frame_pose(frame_name)
                viser_tf = pin_se3_to_viser_se3(pin_tf)

                handle = self.server.scene.add_frame(
                    name=self._scene_path(f"/frames/joint_frames/{frame_name}"),
                    axes_length=0.03 * self.scale,
                    axes_radius=0.0005 * self.scale,
                    position=self._scene_position(viser_tf.translation()),
                    wxyz=viser_tf.rotation().wxyz,
                    visible=self._show_frames,
                )
                self.frame_handles[f"joint_{frame_name}"] = handle

            except Exception as e:
                if self.verbose:
                    self.logger.exception(
                        f"Failed to create joint frame visualization for {frame_name}: {e}"
                    )

    # -- Update --------------------------------------------------------------

    def update(self, state: dict) -> None:
        """Update the visualizer from a state dictionary.

        This is the primary per-frame update entry point, compatible with
        the VizSubscriber calling convention.

        Args:
            state: Dictionary with keys:
                ``q`` (np.ndarray): Joint configuration vector (required).
        """
        self.update_configuration(state["q"])

    def update_configuration(self, q: np.ndarray) -> None:
        """Update robot configuration and refresh all scene handles.

        Args:
            q: Joint configuration vector of shape ``(model.nq,)``

        Raises:
            ValueError: If ``q`` has the wrong shape.
        """
        if q.shape != (self.brainco_model.model.nq,):
            raise ValueError(
                f"Joint configuration must have shape "
                f"({self.brainco_model.model.nq},), got {q.shape}"
            )

        self._current_q = q.copy()
        self.brainco_model.compute_forward_kinematics(q)

        # Update geometry object positions
        for name, handle in self.geometry_handles.items():
            geom_idx = None
            for idx, geom_obj in enumerate(
                self.brainco_model.geometry_model.geometryObjects
            ):
                if geom_obj.name == name:
                    geom_idx = idx
                    break

            if geom_idx is not None:
                pin_tf = self.geometry_data.oMg[geom_idx]
                viser_tf = pin_se3_to_viser_se3(pin_tf)
                handle.position = self._scene_position(viser_tf.translation())
                handle.wxyz = viser_tf.rotation().wxyz

        # Update frame positions
        for frame_name, handle in self.frame_handles.items():
            if frame_name.startswith("sphere_"):
                actual_frame_name = frame_name.replace("sphere_", "")
            elif frame_name.startswith("joint_"):
                actual_frame_name = frame_name.replace("joint_", "")
            else:
                actual_frame_name = frame_name

            if actual_frame_name in self.brainco_model.valid_frames:
                pin_tf = self.brainco_model.get_frame_pose(actual_frame_name)
                viser_tf = pin_se3_to_viser_se3(pin_tf)
                handle.position = self._scene_position(viser_tf.translation())
                handle.wxyz = viser_tf.rotation().wxyz

    def set_cfg(self, q: np.ndarray) -> None:
        """Alias for ``update_configuration``."""
        self.update_configuration(q)

    def reset_to_neutral(self) -> None:
        """Reset robot to neutral (zero) configuration."""
        self.update_configuration(pin.neutral(self.brainco_model.model))

    # -- State queries --------------------------------------------------------

    def get_current_configuration(self) -> np.ndarray:
        """Return a copy of the current joint configuration."""
        return self._current_q.copy()

    def get_current_ee_poses(self) -> dict[str, pin.SE3]:
        """Return a mapping of tip frame name to its current SE3 pose."""
        return {
            frame_name: self.brainco_model.get_frame_pose(frame_name)
            for frame_name in self.brainco_model.tip_frame_names
        }

    # -- Visibility control ---------------------------------------------------

    @property
    def show_visual(self) -> bool:
        """Whether the visual mesh root is visible."""
        return self.visual_root_frame is not None and self.visual_root_frame.visible

    @show_visual.setter
    def show_visual(self, visible: bool) -> None:
        """Set visibility of the visual mesh root (falls back to per-handle control)."""
        if self.visual_root_frame is not None:
            self.visual_root_frame.visible = visible
        else:
            self.show_geometry = visible

    @property
    def show_frames(self) -> bool:
        """Whether end-effector and joint frame axes are visible."""
        return self._show_frames

    @show_frames.setter
    def show_frames(self, visible: bool) -> None:
        """Set visibility of end-effector and joint frame axes."""
        if self._show_frames == visible:
            return
        self._show_frames = visible
        for name, handle in self.frame_handles.items():
            if not name.startswith("sphere_"):
                handle.visible = visible
        if self.verbose:
            print(f"End-effector frames are now {'visible' if visible else 'hidden'}")

    @property
    def show_geometry(self) -> bool:
        """Whether robot geometry meshes are visible."""
        return self._show_geometry

    @show_geometry.setter
    def show_geometry(self, visible: bool) -> None:
        """Set visibility of robot geometry meshes."""
        if self._show_geometry == visible:
            return
        self._show_geometry = visible
        for handle in self.geometry_handles.values():
            handle.visible = visible
        if self.verbose:
            print(f"Robot geometry is now {'visible' if visible else 'hidden'}")

    @property
    def show_ee_spheres(self) -> bool:
        """Whether end-effector spheres are visible."""
        return self._show_ee_spheres

    @show_ee_spheres.setter
    def show_ee_spheres(self, visible: bool) -> None:
        """Set visibility of end-effector spheres."""
        if self._show_ee_spheres == visible:
            return
        self._show_ee_spheres = visible
        for name, handle in self.frame_handles.items():
            if name.startswith("sphere_"):
                handle.visible = visible
        if self.verbose:
            print(f"End-effector spheres are now {'visible' if visible else 'hidden'}")

    def toggle_frames(self) -> None:
        """Toggle visibility of frame axes."""
        self.show_frames = not self.show_frames

    def toggle_geometry(self) -> None:
        """Toggle visibility of robot geometry."""
        self.show_geometry = not self.show_geometry

    def toggle_ee_spheres(self) -> None:
        """Toggle visibility of end-effector spheres."""
        self.show_ee_spheres = not self.show_ee_spheres

    def show_all(self) -> None:
        """Show all visualization components."""
        self.show_frames = True
        self.show_geometry = True
        self.show_ee_spheres = True
        if self.verbose:
            print("All visualization components are now visible")

    def hide_all(self) -> None:
        """Hide all visualization components."""
        self.show_frames = False
        self.show_geometry = False
        self.show_ee_spheres = False
        if self.verbose:
            print("All visualization components are now hidden")

    def get_visibility_status(self) -> dict[str, bool]:
        """Return current visibility state of all components."""
        return {
            "frames": self._show_frames,
            "geometry": self._show_geometry,
            "ee_spheres": self._show_ee_spheres,
        }

    # -- Lifecycle ------------------------------------------------------------

    def remove(self) -> None:
        """Remove all scene objects from the Viser scene."""
        for frame in self.joint_frames:
            frame.remove()
        for handle in self.frame_handles.values():
            handle.remove()
        for handle in self.geometry_handles.values():
            handle.remove()
        for mesh in self.meshes:
            mesh.remove()

        self.joint_frames.clear()
        self.frame_handles.clear()
        self.geometry_handles.clear()
        self.meshes.clear()

    def close(self) -> None:
        """Remove scene objects and release server ownership if applicable."""
        self.remove()
        if self.verbose:
            print("BrainCoRenderer closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        port_info = self._port if self._port is not None else "unknown"
        return (
            f"BrainCoRenderer("
            f"hand='{self.brainco_model.hand_side}', port={port_info})"
        )
