import numpy as np
import warp as wp

from .base_controller import BaseController
from ..utils.env_utils import write_targets_kernel


class JointPDController(BaseController):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def initialize_resources(self, env) -> None:
        self._dofs_per_world = env.model.joint_dof_count // env.num_envs
        self._num_joints = env.num_robot * env.num_joints

    def compute(self, env, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float32)
        if target.ndim == 1:
            target = target[np.newaxis, :]  # (1, num_robot * 8)
        assert target.shape == (env.num_envs, self._num_joints)
        joint_targets_wp = wp.array(target, dtype=float)
        wp.launch(
            write_targets_kernel,
            dim=(env.num_envs, self._num_joints),
            inputs=[joint_targets_wp, env.control.joint_target_pos, self._dofs_per_world],
        )
