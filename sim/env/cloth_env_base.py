from pathlib import Path
from abc import abstractmethod
from typing import Any
import numpy as np
import warp as wp
import copy
from omegaconf import OmegaConf
from pxr import Usd, UsdGeom
import transforms3d as t3d
import sys
sys.path.append(str(Path(__file__).parents[2] / "newton"))
import newton
from newton._src.geometry.utils import load_mesh
from newton.solvers import style3d

from .base_env import BaseEnv
from ..utils.env_utils import resolve_asset_path, pose_to_pos_quat


class ClothEnvBase(BaseEnv):

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        self.num_robot = int(self.cfg.env.num_robot)
        self.endeffector_id = int(self.cfg.env.endeffector_id)
        self.num_arm_joints = int(self.cfg.env.num_arm_joints)
        self.num_gripper_joints = int(self.cfg.env.num_gripper_joints)
        self.endeffector_offset = wp.transform(
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_identity(),
        )

        self.sim_substeps = 15
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.collide_substeps = 1

        self.cloth_solver_type = self.cfg.env.cloth_solver_type
        assert self.cloth_solver_type in ['vbd', 'style3d'], "Unsupported cloth solver type"

        self.cloth_body_contact_margin = 0.01
        self.cloth_self_contact_radius = 0.002
        self.cloth_self_contact_margin = 0.003

        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 0.1

        self.robot_friction = 1.0
        self.table_friction = 0.25
        self.cloth_friction = 0.25

        self._build_scene()

        self.viewer.set_model(self.model)
        self._setup_viewer_camera()

        # Base control arrays
        self.control = self.model.control()

        # Evaluate FK and copy initial joint positions to control targets
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Initialize control targets
        for j in range(self.model.joint_count):
            joint_q_start = int(self.model.joint_q_start.numpy()[j])
            joint_q_end = int(self.model.joint_q_start.numpy()[j + 1])
            joint_qd_start = int(self.model.joint_qd_start.numpy()[j])
            joint_qd_end = int(self.model.joint_qd_start.numpy()[j + 1])
            if self.model.joint_type.numpy()[j] == newton.JointType.FREE:
                assert joint_q_end - joint_q_start == 7 and joint_qd_end - joint_qd_start == 6
                wp.copy(
                    self.control.joint_target_pos,
                    self.model.joint_q,
                    dest_offset=joint_qd_start,
                    src_offset=joint_q_start,
                    count=6,  # special treatment for free joints because coord_dim=7 but dof_dim=6
                )
            elif joint_q_end - joint_q_start > 0:
                assert joint_q_end - joint_q_start == joint_qd_end - joint_qd_start
                wp.copy(
                    self.control.joint_target_pos,
                    self.model.joint_q,
                    dest_offset=joint_qd_start,
                    src_offset=joint_q_start,
                    count=joint_q_end - joint_q_start,
                )

        # Collision pipeline
        sdf_hydroelastic_config = newton.geometry.HydroelasticSDF.Config(
            output_contact_surface=hasattr(self.viewer, "renderer"),
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase="explicit",
            sdf_hydroelastic_config=sdf_hydroelastic_config,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        # CUDA graph capture
        self.capture()

    @abstractmethod
    def _load_urdf_asset(self, asset, pos: np.ndarray, quat: np.ndarray, xform) -> None:
        """Load a URDF robot asset with robot-specific gains, DOF, and finger names."""
        ...

    @abstractmethod
    def _setup_viewer_camera(self) -> None:
        """Set viewer camera position and angle for this robot's workspace."""
        ...

    def _pre_finalize_scene(self) -> None:
        """Hook called after loading assets but before model.finalize(). Override as needed."""
        pass

    def _post_build_scene(self) -> None:
        """Hook called after model is fully set up. Override to add robot-specific arrays."""
        pass

    def _build_scene(self):
        self.scene = newton.ModelBuilder()

        newton.solvers.SolverMuJoCo.register_custom_attributes(self.scene)
        if self.cloth_solver_type == 'style3d':
            newton.solvers.SolverStyle3D.register_custom_attributes(self.scene)

        self.scene.default_shape_cfg.ke = 5.0e4
        self.scene.default_shape_cfg.kd = 5.0e2
        self.scene.default_shape_cfg.kf = 1.0e3
        self.scene.default_shape_cfg.mu = self.table_friction

        self.gs_assets = []
        self.mesh_assets = {}
        self.urdf_assets = []

        self.shape_cfg = newton.ModelBuilder.ShapeConfig(
            kh=1e11,
            sdf_max_resolution=64,
            is_hydroelastic=True,
            sdf_narrow_band_range=(-0.01, 0.01),
            contact_margin=0.01,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )
        self.mesh_shape_cfg = copy.deepcopy(self.shape_cfg)
        self.mesh_shape_cfg.sdf_max_resolution = None
        self.mesh_shape_cfg.sdf_target_voxel_size = None
        self.mesh_shape_cfg.sdf_narrow_band_range = (-0.1, 0.1)
        self.hydro_mesh_sdf_max_resolution = 64

        self.urdf_shape_cfg = copy.deepcopy(self.shape_cfg)
        self.urdf_shape_cfg.is_hydroelastic = False
        self.urdf_shape_cfg.sdf_max_resolution = None
        self.urdf_shape_cfg.sdf_target_voxel_size = None
        self.urdf_shape_cfg.sdf_narrow_band_range = (-0.1, 0.1)

        self.b2w_list = []           # one base-to-world transform per arm (wxyz)
        self._single_arm_models = [] # finalized per-arm models for IK

        for asset in self.cfg.env.assets:
            self.load_asset(asset)

        self._pre_finalize_scene()

        self.model = self.scene.finalize(requires_grad=False)
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.cloth_friction

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.sim_time = 0.0

        self.robot_solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=500,
            nconmax=500,
            iterations=15,
            ls_iterations=100,
            impratio=1000.0,
        )

        if self.cloth_solver_type == 'style3d':
            self.cloth_solver = newton.solvers.SolverStyle3D(
                model=self.model,
                iterations=self.iterations,
                two_way_coupling=True,
            )
            self.cloth_solver.precompute(self.scene)
            self.cloth_solver.collision.radius = 3.5e-3
        else:
            self.model.edge_rest_angle.zero_()
            self.cloth_solver = newton.solvers.SolverVBD(
                self.model,
                iterations=self.iterations,
                integrate_with_external_rigid_solver=True,
                two_way_coupling=True,
                particle_self_contact_radius=self.cloth_self_contact_radius,
                particle_self_contact_margin=self.cloth_self_contact_margin,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.5,
                particle_enable_self_contact=True,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                particle_collision_detection_interval=-1,
                rigid_contact_k_start=1e4,
            )

        self._post_build_scene()

    def load_asset(self, asset):
        """Load a scene asset (urdf/mesh/gs). URDF loading delegates to _load_urdf_asset()."""
        asset_type = str(asset.type).lower()
        asset_name = getattr(asset, "name", None)
        pose = getattr(asset, "pose", None)
        pos, quat = pose_to_pos_quat(pose)
        xform = wp.transform(wp.vec3(*pos), wp.quat(*quat))

        if asset_type == "urdf":
            self._load_urdf_asset(asset, pos, quat, xform)
            return

        if asset_type == "mesh":
            mesh_type = str(getattr(asset, "mesh_type", "rigid")).lower()

            if getattr(asset, "primitive_type", None) is not None:
                primitive_type = str(getattr(asset, "primitive_type")).lower()
                if primitive_type == "box":
                    is_static = asset.is_static
                    body = self.scene.add_body(xform=xform, mass=1.0) if not is_static else -1
                    size = np.array(getattr(asset, "size", [0.1, 0.1, 0.1]), dtype=np.float32)

                    mesh_shape_cfg = copy.deepcopy(self.mesh_shape_cfg)
                    if getattr(asset, "density", None) is not None:
                        mesh_shape_cfg.density = float(asset.density)

                    box_mesh = newton.Mesh.create_box(
                        size[0] * 0.5,
                        size[1] * 0.5,
                        size[2] * 0.5,
                        duplicate_vertices=True,
                        compute_normals=False,
                        compute_uvs=False,
                        compute_inertia=True,
                    )
                    box_mesh._color = (0.8, 0.8, 0.8)
                    box_mesh.build_sdf(
                        max_resolution=self.hydro_mesh_sdf_max_resolution,
                        narrow_band_range=mesh_shape_cfg.sdf_narrow_band_range,
                        margin=mesh_shape_cfg.contact_margin if mesh_shape_cfg.contact_margin is not None else 0.05,
                    )
                    self.scene.add_shape_mesh(
                        body=body,
                        xform=xform if is_static else None,
                        mesh=box_mesh,
                        label=asset_name,
                        cfg=mesh_shape_cfg,
                    )
                else:
                    raise ValueError(f"Unsupported primitive type: {primitive_type}")
                self.mesh_assets[asset_name] = {
                    "path": None,
                    "type": mesh_type,
                    "is_static": is_static,
                    "pose": np.concatenate([pos, quat], axis=0),
                }
                return

            mesh_path = getattr(asset, "path", None)
            if mesh_path is None:
                raise ValueError("Mesh asset missing path.")
            mesh_path = resolve_asset_path(mesh_path)

            panel_verts = None
            panel_indices = None
            if mesh_path.suffix.lower() == ".usd":
                usd_stage = Usd.Stage.Open(asset.path)
                usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath(asset.prim_path))
                mesh_points = np.array(usd_geom.GetPointsAttr().Get())
                mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
                if mesh_type == "cloth":
                    prim = UsdGeom.PrimvarsAPI(usd_geom.GetPrim()).GetPrimvar("st")
                    panel_indices = np.array(prim.GetIndices())
                    panel_verts = np.array(prim.Get())
                    assert panel_verts is not None and panel_indices is not None
                    uv_scale = float(getattr(asset, "cloth_uv_scale", 1e-3))
                    panel_verts = panel_verts * uv_scale
                    if getattr(asset, "center", False):
                        mesh_center = 0.5 * (mesh_points.max(axis=0) + mesh_points.min(axis=0))
                        mesh_points = mesh_points - mesh_center
            else:
                mesh_points, mesh_indices = load_mesh(str(mesh_path))

            if mesh_type == "cloth":
                is_static = False
                vertices = [wp.vec3(*v.tolist()) for v in mesh_points]
                indices = mesh_indices.tolist()
                if self.cloth_solver_type == 'style3d':
                    style3d.add_cloth_mesh(
                        self.scene,
                        pos=wp.vec3(*pos.tolist()),
                        rot=wp.quat(*quat),
                        scale=float(getattr(asset, "scale", 1.0)),
                        vel=wp.vec3(*getattr(asset, "initial_velocity", [0.0, 0.0, 0.0])),
                        vertices=vertices,
                        indices=indices,
                        density=float(getattr(asset, "cloth_density", 0.5)),
                        tri_aniso_ke=wp.vec3(*getattr(asset, "tri_aniso_ke", [5.0e2, 5.0e2, 5.0e2])),
                        edge_aniso_ke=wp.vec3(*getattr(asset, "edge_aniso_ke", [2.0e-5, 2.0e-5, 2.0e-5])),
                        panel_verts=panel_verts.tolist(),
                        panel_indices=panel_indices.tolist(),
                        particle_radius=getattr(asset, "particle_radius", 0.003),
                    )
                else:
                    self.scene.add_cloth_mesh(
                        pos=wp.vec3(*pos.tolist()),
                        rot=wp.quat(*quat),
                        scale=float(getattr(asset, "scale", 1.0)),
                        vel=wp.vec3(*getattr(asset, "initial_velocity", [0.0, 0.0, 0.0])),
                        vertices=vertices,
                        indices=indices,
                        density=float(getattr(asset, "cloth_density", 0.5)),
                        tri_ke=1.0e2,
                        tri_ka=1.0e2,
                        tri_kd=1.5e-6,
                        edge_ke=0.05,
                        edge_kd=1e-4,
                        particle_radius=getattr(asset, "particle_radius", 0.003),
                    )
                    self.scene.color()
            elif mesh_type == "rigid":
                mesh = newton.Mesh(mesh_points, mesh_indices)
                is_static = asset.is_static
                body = self.scene.add_body(xform=xform, mass=1.0) if not is_static else -1

                mesh_shape_cfg = copy.deepcopy(self.mesh_shape_cfg)
                if getattr(asset, "density", None) is not None:
                    mesh_shape_cfg.density = float(asset.density)

                mesh.build_sdf(
                    max_resolution=self.hydro_mesh_sdf_max_resolution,
                    narrow_band_range=mesh_shape_cfg.sdf_narrow_band_range,
                    margin=mesh_shape_cfg.contact_margin if mesh_shape_cfg.contact_margin is not None else 0.05,
                )
                self.scene.add_shape_mesh(
                    body=body,
                    xform=xform if is_static else None,
                    mesh=mesh,
                    scale=wp.vec3(1.0, 1.0, 1.0),
                    label=asset_name,
                    cfg=mesh_shape_cfg,
                )
            else:
                raise ValueError(f"Unsupported mesh type: {mesh_type}")

            self.mesh_assets[asset_name] = {
                "name": asset_name,
                "path": str(mesh_path),
                "type": mesh_type,
                "is_static": is_static if mesh_type == "rigid" else False,
                "pose": np.concatenate([pos, quat], axis=0),
            }
            return

        raise ValueError(f"Unsupported asset type: {asset_type}")

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)

        for _step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.viewer.apply_forces(self.state_0)

            if _step % self.collide_substeps == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)

            particle_count = self.model.particle_count

            self.model.particle_count = 0
            self.model.shape_contact_pair_count = 0
            self.robot_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.model.particle_count = particle_count

            if particle_count > 0:
                self.state_0.particle_f.zero_()

            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
