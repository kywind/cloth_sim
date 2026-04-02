from pathlib import Path
import numpy as np
import warp as wp
from omegaconf import OmegaConf
import newton
import newton.solvers
import sys
sys.path.append(str(Path(__file__).parents[2] / "newton"))

from .cloth_env_base import ClothEnvBase
from ..utils.env_utils import xyzw_to_wxyz, quat_to_vec4, resolve_asset_path


class ClothEnvARX(ClothEnvBase):

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

    def _setup_viewer_camera(self) -> None:
        table_height = float(self.cfg.env.get("table_height", 0.0))
        self.viewer.set_camera(pos=wp.vec3((1.5, 0.0, 0.75 + table_height)), pitch=-30, yaw=180)

    def _pre_finalize_scene(self) -> None:
        """Add ground plane before finalizing the combined model."""
        self.multi_scene.default_shape_cfg.ke = 5.0e4
        self.multi_scene.default_shape_cfg.kd = 5.0e2
        self.multi_scene.add_ground_plane()

    def _load_urdf_asset(self, asset, pos: np.ndarray, quat: np.ndarray, xform) -> None:
        urdf_path = resolve_asset_path(asset.path)

        robot_builder = newton.ModelBuilder()
        robot_builder.default_shape_cfg = self.urdf_shape_cfg
        robot_builder.rigid_contact_margin = 0.001  # 1mm — important for ARX contact handling
        robot_builder.add_urdf(
            str(urdf_path),
            xform=xform,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        robot_builder.default_shape_cfg = self.shape_cfg

        def find_body(name):
            return next(i for i, lbl in enumerate(robot_builder.body_label) if lbl.endswith(f"/{name}"))

        finger_body_indices = {
            find_body("link6"),
            find_body("link7"),
            find_body("link8"),
        }
        non_finger_shape_indices = []
        for shape_idx, body_idx in enumerate(robot_builder.shape_body):
            if body_idx in finger_body_indices and robot_builder.shape_type[shape_idx] == newton.GeoType.MESH:
                mesh = robot_builder.shape_source[shape_idx]
                if mesh is not None and mesh.sdf is None:
                    shape_scale = np.asarray(robot_builder.shape_scale[shape_idx], dtype=np.float32)
                    if not np.allclose(shape_scale, 1.0):
                        mesh = mesh.copy(vertices=mesh.vertices * shape_scale, recompute_inertia=True)
                        robot_builder.shape_source[shape_idx] = mesh
                        robot_builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
                    mesh.build_sdf(
                        max_resolution=self.hydro_mesh_sdf_max_resolution,
                        narrow_band_range=self.shape_cfg.sdf_narrow_band_range,
                        margin=self.shape_cfg.contact_margin if self.shape_cfg.contact_margin is not None else 0.05,
                    )
                robot_builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
            elif body_idx not in finger_body_indices:
                robot_builder.shape_flags[shape_idx] &= ~newton.ShapeFlags.HYDROELASTIC
                non_finger_shape_indices.append(shape_idx)

        robot_builder.approximate_meshes(
            method="convex_hull", shape_indices=non_finger_shape_indices, keep_visual_shapes=True
        )

        init_q = getattr(asset, "init_q", None)
        assert init_q is not None, "URDF assets must specify init_q."
        robot_builder.joint_q[:8] = [*init_q, 0.01, 0.01]
        robot_builder.joint_target_pos[:8] = [*init_q, 1.0, 1.0]

        robot_builder.joint_target_ke[:8] = [500.0] * 8
        robot_builder.joint_target_kd[:8] = [50.0] * 8
        robot_builder.joint_effort_limit[:6] = [80.0] * 6
        robot_builder.joint_effort_limit[6:8] = [20.0] * 2
        robot_builder.joint_armature[:6] = [0.1] * 6
        robot_builder.joint_armature[6:8] = [0.5] * 2

        if getattr(self.cfg.env, "gravity_compensation", False):
            # Enable MuJoCo gravity compensation for all arm joints and bodies so
            # the impedance controller (which zeros joint_target_ke) is not fighting gravity.
            # robot_builder is a fresh ModelBuilder (not built from MJCF), so we must explicitly
            # register the MuJoCo custom attributes before accessing them.
            newton.solvers.SolverMuJoCo.register_custom_attributes(robot_builder)
            gravcomp_jnt = robot_builder.custom_attributes["mujoco:jnt_actgravcomp"]
            if gravcomp_jnt.values is None:
                gravcomp_jnt.values = {}
            for dof_idx in range(self.num_arm_joints):
                gravcomp_jnt.values[dof_idx] = True

            gravcomp_body = robot_builder.custom_attributes["mujoco:gravcomp"]
            if gravcomp_body.values is None:
                gravcomp_body.values = {}
            for body_idx in range(robot_builder.body_count):
                gravcomp_body.values[body_idx] = 1.0

        base_pose = np.concatenate([pos, quat], axis=0)
        self.b2w_list.append(xyzw_to_wxyz(base_pose))

        # Finalize a standalone single-arm model for IK before merging into the combined scene
        self._single_arm_models.append(robot_builder.finalize())

        self.scene.add_builder(robot_builder)
        asset_name = getattr(asset, "name", None)
        self.urdf_assets.append({
            "name": asset_name,
            "path": str(urdf_path),
            "pose": np.concatenate([pos, quat], axis=0),
            "joint_q": robot_builder.joint_q,
        })
