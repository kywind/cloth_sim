from pathlib import Path
from typing import Optional, Any
import numpy as np
np.atan2 = np.arctan2
np.pow = np.power
import warp as wp
import OpenGL.GL as gl
import ctypes
import time
from pxr import Usd, UsdGeom
from omegaconf import OmegaConf
import newton
import newton.utils
import sys
sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[2] / "newton"))
import sim
from ..utils.env_utils import combine_transforms
from ..controller import BaseController
from ..renderer import BaseRenderer


class BaseEnv:

    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.viewer = newton.viewer.ViewerGL(headless=cfg.env.headless)

        self.num_envs = int(self.cfg.env.num_envs)
        self.model = None
        self.state_0 = None
        self.state_1 = None

        self.sim_time = 0.0
        self._shape_mesh_cache = {}

        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)

    def initialize_resources(self, no_renderer: bool = False):
        if not no_renderer:
            self.setup_renderer()
        self.setup_controller()

    def reset(self, seed=None, options=None):
        if self.task is not None:
            self.task.reset(self)
            return self.task.observation(self), {}
        return {}, {}

    def capture(self):
        if self.cfg.env.use_graph and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None
    
    def step(self, action: Optional[dict] = None):
        assert action is not None and 'target' in action, "Action must contain 'target' key."

        self.controller.compute(self, action['target'])

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        return {}, 0.0, False, False, {}

    def setup_controller(self):
        self.controller: BaseController = eval(self.cfg.controller.name)(cfg=self.cfg)
        self.controller.initialize_resources(self)

    def setup_renderer(self):
        self.renderer: BaseRenderer = eval(self.cfg.renderer.name)(cfg=self.cfg)
        self.renderer.initialize_resources(self)

    def render(self, *args, **kwargs):
        if self.renderer is None:
            return {}
        return self.renderer.render(self, *args, **kwargs)

    def _get_shape_local_mesh(self, shape_id: int, geo_type: newton.GeoType, scale: np.ndarray):
        cache = self._shape_mesh_cache.get(shape_id)
        if cache is not None and cache["geo_type"] == geo_type and np.allclose(cache["scale"], scale):
            return cache["vertices"], cache["faces"]

        if geo_type == newton.GeoType.MESH:
            src = self.model.shape_source[shape_id]
            vertices = src.vertices * scale
            faces = src.indices
        elif geo_type == newton.GeoType.PLANE:
            width = scale[0] if len(scale) > 0 and scale[0] > 0.0 else 1000.0
            length = scale[1] if len(scale) > 1 and scale[1] > 0.0 else 1000.0
            m = newton.Mesh.create_plane(width, length, compute_inertia=False)
            vertices, faces = m.vertices, m.indices
        elif geo_type == newton.GeoType.SPHERE:
            m = newton.Mesh.create_sphere(scale[0], compute_inertia=False)
            vertices, faces = m.vertices, m.indices
        elif geo_type == newton.GeoType.CAPSULE:
            m = newton.Mesh.create_capsule(scale[0], scale[1], up_axis=newton.Axis.Z, compute_inertia=False)
            vertices, faces = m.vertices, m.indices
        elif geo_type == newton.GeoType.CYLINDER:
            m = newton.Mesh.create_cylinder(scale[0], scale[1], up_axis=newton.Axis.Z, compute_inertia=False)
            vertices, faces = m.vertices, m.indices
        elif geo_type == newton.GeoType.CONE:
            m = newton.Mesh.create_cone(scale[0], scale[1], up_axis=newton.Axis.Z, compute_inertia=False)
            vertices, faces = m.vertices, m.indices
        elif geo_type == newton.GeoType.BOX:
            ext = tuple(scale[:3]) if len(scale) >= 3 else (scale[0],) * 3
            m = newton.Mesh.create_box(*ext, compute_inertia=False)
            vertices, faces = m.vertices, m.indices
        elif geo_type == newton.GeoType.ELLIPSOID:
            ext = tuple(scale[:3]) if len(scale) >= 3 else (scale[0],) * 3
            m = newton.Mesh.create_ellipsoid(*ext, compute_inertia=False)
            vertices, faces = m.vertices, m.indices
        else:
            return None, None

        faces = np.asarray(faces, dtype=np.int32)
        if faces.ndim == 1:
            faces = faces.reshape(-1, 3)

        vertices = np.asarray(vertices, dtype=np.float32)
        if vertices.ndim == 2 and vertices.shape[1] > 3:
            vertices = vertices[:, :3]  # other dims: (nz, ny, nz, u, v)
        vertices = vertices.reshape(-1, 3)

        self._shape_mesh_cache[shape_id] = {
            "geo_type": geo_type,
            "scale": np.array(scale, dtype=np.float32),
            "vertices": vertices,
            "faces": faces,
        }
        return vertices, faces

    def get_live_meshes(self):
        if self.state_0 is None or self.model is None:
            return []

        meshes = []
        if self.state_0.particle_count > 0:
            vertices = self.state_0.particle_q.numpy()
            faces = np.asarray(self.scene.tri_indices, dtype=np.int32)
            if faces.ndim == 1:
                faces = faces.reshape(-1, 3)
            meshes.append(
                {
                    "name": "cloth_particles",
                    "vertices": vertices,
                    "faces": faces,
                    "deformable": True,
                }
            )

        shape_type = self.model.shape_type.numpy()
        shape_scale = self.model.shape_scale.numpy()
        shape_body = self.model.shape_body.numpy()
        shape_transform = self.model.shape_transform.numpy()
        body_q = self.state_0.body_q.numpy()
        shape_flags = None
        if hasattr(self.model, "shape_flags"):
            shape_flags = self.model.shape_flags.numpy()

        for sid in range(self.model.shape_count):
            if shape_flags is not None:
                if not (shape_flags[sid] & int(newton.ShapeFlags.VISIBLE)):
                    continue
            geo_type = newton.GeoType(int(shape_type[sid]))
            if geo_type == newton.GeoType.NONE:
                continue
            vertices, faces = self._get_shape_local_mesh(sid, geo_type, shape_scale[sid])
            if vertices is None:
                continue

            body = int(shape_body[sid])
            if body >= 0:
                world_tf = combine_transforms(body_q[body], shape_transform[sid])
            else:
                world_tf = np.asarray(shape_transform[sid], dtype=np.float32)

            meshes.append(
                {
                    "name": f"shape_{sid}",
                    "vertices": vertices,
                    "faces": faces,
                    "transform": {
                        "pos": world_tf[:3],
                        "quat": world_tf[3:7],
                    },
                    "is_robot": body >= 0,
                    "shape_body": body,
                    "shape_label": self.model.shape_label[sid] if self.model.shape_label else None,
                }
            )

        return meshes
