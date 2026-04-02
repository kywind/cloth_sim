import numpy as np
import warp as wp
import newton
import newton.ik as ik

from .base_controller import BaseController
from ..utils.env_utils import quat_to_vec4, scatter_ik_solutions_kernel


class IKController(BaseController):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def initialize_resources(self, env) -> None:
        self.setup_ik(env)

        self.graph_ik_list = []
        if self.cfg.env.use_graph:
            for ri in range(env.num_robot):
                with wp.ScopedCapture() as capture:
                    self.ik_solvers[ri].step(
                        self.joint_q_ik_list[ri], self.joint_q_ik_list[ri],
                        iterations=self.ik_iters,
                    )
                self.graph_ik_list.append(capture.graph)

    def setup_ik(self, env) -> None:
        assert len(env._single_arm_models) == env.num_robot, (
            f"Expected {env.num_robot} single-arm models, got {len(env._single_arm_models)}. "
            f"Ensure controller='ik' was set before _build_scene() ran."
        )

        self.ik_iters = 24
        self.n_problems = getattr(env, 'num_envs', 1)
        num_total_joints = env.num_arm_joints + env.num_gripper_joints
        self.dofs_per_world = getattr(
            env, '_dofs_per_world', env.num_robot * num_total_joints
        )

        self.pos_objs = []
        self.rot_objs = []
        self.obj_joint_limits_list = []
        self.joint_q_ik_list = []
        self.ik_solvers = []
        self.gripper_values_list = []
        self.init_q_list = []

        for model_single in env._single_arm_models:
            state_single = model_single.state()
            body_q_np = state_single.body_q.numpy()
            ee_tf = wp.transform(*body_q_np[env.endeffector_id])

            init_pos = wp.transform_get_translation(ee_tf)
            init_rot = quat_to_vec4(wp.transform_get_rotation(ee_tf))

            pos_obj = ik.IKObjectivePosition(
                link_index=env.endeffector_id,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=wp.array([init_pos] * self.n_problems, dtype=wp.vec3),
            )
            rot_obj = ik.IKObjectiveRotation(
                link_index=env.endeffector_id,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=wp.array([init_rot] * self.n_problems, dtype=wp.vec4),
            )
            obj_joint_limits = ik.IKObjectiveJointLimit(
                joint_limit_lower=model_single.joint_limit_lower,
                joint_limit_upper=model_single.joint_limit_upper,
            )

            init_q_np = model_single.joint_q.numpy().astype(np.float32)
            joint_q_ik = wp.array(
                np.tile(init_q_np, (self.n_problems, 1)),
                dtype=wp.float32,
            )

            ik_solver = ik.IKSolver(
                model=model_single,
                n_problems=self.n_problems,
                objectives=[pos_obj, rot_obj, obj_joint_limits],
                lambda_initial=0.1,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
            )

            gripper_values = wp.zeros(self.n_problems, dtype=wp.float32)

            self.pos_objs.append(pos_obj)
            self.rot_objs.append(rot_obj)
            self.obj_joint_limits_list.append(obj_joint_limits)
            self.joint_q_ik_list.append(joint_q_ik)
            self.ik_solvers.append(ik_solver)
            self.gripper_values_list.append(gripper_values)
            self.init_q_list.append(init_q_np)

    def compute(self, env, target: np.ndarray) -> None:
        # Accept (num_robot * 8,) for single-env or (num_envs, num_robot * 8) for multi-env.
        target = np.asarray(target, dtype=np.float32)
        if target.ndim == 1:
            target = target[np.newaxis, :]  # (1, num_robot * 8)

        for ri in range(env.num_robot):
            target_positions = target[:, ri * 8: ri * 8 + 3]      # (num_envs, 3)
            target_rotations = target[:, ri * 8 + 3: ri * 8 + 7]  # (num_envs, 4) xyzw
            gripper_targets = target[:, ri * 8 + 7]               # (num_envs,)

            self.pos_objs[ri].set_target_positions(
                wp.array([wp.vec3(*p) for p in target_positions], dtype=wp.vec3)
            )
            self.rot_objs[ri].set_target_rotations(
                wp.array([quat_to_vec4(q) for q in target_rotations], dtype=wp.vec4)
            )
            self.gripper_values_list[ri].assign(gripper_targets)

            if self.graph_ik_list:
                wp.capture_launch(self.graph_ik_list[ri])
            else:
                self.ik_solvers[ri].step(
                    self.joint_q_ik_list[ri], self.joint_q_ik_list[ri],
                    iterations=self.ik_iters,
                )

            wp.launch(
                scatter_ik_solutions_kernel,
                dim=self.n_problems,
                inputs=[
                    self.joint_q_ik_list[ri],
                    self.gripper_values_list[ri],
                    env.num_arm_joints,
                    env.num_gripper_joints,
                    ri,
                    self.dofs_per_world,
                    env.control.joint_target_pos,
                ],
            )
