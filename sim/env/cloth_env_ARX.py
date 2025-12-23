from pathlib import Path
from typing import Optional
import numpy as np
import warp as wp
import time
from pxr import Usd, UsdGeom
import sys
sys.path.append(str(Path(__file__).parents[2] / "newton"))
import newton
import newton.utils
import newton.viewer
from newton import Model, ModelBuilder, Style3DModel, Style3DModelBuilder, State, eval_fk, ParticleFlags
from newton.solvers import SolverFeatherstone, SolverVBD, SolverStyle3D, SolverXPBD
from newton.utils import transform_twist
from ..utils.planner import CuroboPlanner


@wp.kernel
def compute_ee_delta(
    body_q: wp.array(dtype=wp.transform),
    offset: wp.transform,
    body_id: int,
    bodies_per_robot: int,
    target: wp.array(dtype=wp.transform),
    # outputs
    ee_delta: wp.array(dtype=wp.spatial_vector),
):
    robot_id = wp.tid()
    tf = body_q[bodies_per_robot * robot_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target[robot_id])
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target[robot_id])
    ang_diff = rot_des * wp.quat_inverse(rot)
    # compute pose difference between end effector and target
    ee_delta[robot_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


@wp.kernel
def compute_body_out(
    body_qd: wp.array(dtype=wp.spatial_vector), 
    body_id: int,
    bodies_per_robot: int,
    body_offset: wp.transform,
    # outputs
    body_out: wp.array(dtype=float)
):
    # TODO verify transform twist
    robot_id = wp.tid()
    mv = transform_twist(body_offset, body_qd[bodies_per_robot * robot_id + body_id])
    for i in range(6):
        body_out[6 * robot_id + i] = mv[i]  # 6: twist dimension


class ClothEnvARXV1:
    def __init__(self, cfg, viewer):
        self.cfg = cfg
        self.viewer = viewer

        self.num_robot = 2

        # simulation parameters
        self.add_cloth = True
        self.add_robot = True
        self.sim_substeps = 15
        self.iterations = 10
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # body-cloth contact
        self.cloth_particle_radius = 0.003  # style3d
        self.cloth_body_contact_margin = 0.01
        self.soft_contact_ke = 100
        self.soft_contact_kd = 2e-3

        # cloth-cloth contact
        self.self_contact_radius = 0.002
        self.self_contact_margin = 0.003

        # friction
        self.robot_friction = 10.0
        self.table_friction = 0.25
        self.cloth_friction = 0.25

        self.scene = Style3DModelBuilder()

        self.scene.default_shape_cfg.ke = 5.0e4
        self.scene.default_shape_cfg.kd = 5.0e2
        self.scene.default_shape_cfg.kf = 1.0e3
        self.scene.default_shape_cfg.mu = self.table_friction

        if self.add_robot:
            arx = Style3DModelBuilder()
            arx.default_shape_cfg.mu = self.robot_friction

            arx.add_urdf(
                'experiments/assets/robots/ARX-X5/X5A.urdf',
                xform=wp.transform(
                    (0.5, -0.35, 0.0),
                    wp.quat_identity(),
                ),
                floating=False,
                scale=1,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                force_show_colliders=False,
            )
            arx.joint_q[:8] = [0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0, 0.01, 0.01]
            self.left_b2w = np.array([0.5, -0.35, 0.0, 1.0, 0.0, 0.0, 0.0])  # wxyz

            self.num_arm_joints = 6
            self.num_gripper_joints = 2
            self.bodies_per_robot = arx.body_count
            self.dof_per_robot = arx.joint_dof_count
            
            self.endeffector_id = arx.body_count - 3
            self.endeffector_offset = wp.transform(
                [
                    0.0,
                    0.0,
                    0.0,
                ],
                wp.quat_identity(),
            )

            arx.add_urdf(
                'experiments/assets/robots/ARX-X5/X5A.urdf',
                xform=wp.transform(
                    (0.5, 0.35, 0.0),
                    wp.quat_identity(),
                ),
                floating=False,
                scale=1,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                force_show_colliders=False,
            )
            arx.joint_q[8:16] = [0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0, 0.01, 0.01]
            self.right_b2w = np.array([0.5, 0.35, 0.0, 1.0, 0.0, 0.0, 0.0])  # wxyz

            arx.joint_target_kd[:self.num_arm_joints] = [200.0] * self.num_arm_joints
            arx.joint_target_ke[:self.num_arm_joints] = [100.0] * self.num_arm_joints
            arx.joint_target_kd[self.num_arm_joints:self.dof_per_robot] = [400.0] * self.num_gripper_joints
            arx.joint_target_ke[self.num_arm_joints:self.dof_per_robot] = [200.0] * self.num_gripper_joints

            arx.joint_target_kd[self.dof_per_robot:self.dof_per_robot + self.num_arm_joints] = [200.0] * self.num_arm_joints
            arx.joint_target_ke[self.dof_per_robot:self.dof_per_robot + self.num_arm_joints] = [100.0] * self.num_arm_joints
            arx.joint_target_kd[self.dof_per_robot + self.num_arm_joints:] = [400.0] * self.num_gripper_joints
            arx.joint_target_ke[self.dof_per_robot + self.num_arm_joints:] = [200.0] * self.num_gripper_joints

            self.scene.add_builder(arx)

        # add a table
        self.scene.add_shape_box(
            -1,
            wp.transform(
                wp.vec3(0.70, 0.0, 0.10),
                wp.quat_identity(),
            ),
            hx=0.15,
            hy=0.10,
            hz=0.01,
        )

        # add the garment
        garment_usd_name = "Female_T_Shirt"
        asset_path = newton.utils.download_asset("style3d")
        usd_stage = Usd.Stage.Open(f"{asset_path}/garments/{garment_usd_name}.usd")
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath(f"/Root/{garment_usd_name}/Root_Garment"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
        prim = UsdGeom.PrimvarsAPI(usd_geom.GetPrim()).GetPrimvar("st")
        mesh_uv_indices = np.array(prim.GetIndices())
        mesh_uv = np.array(prim.Get()) * 1e-3

        mesh_points -= mesh_points.mean(axis=0)

        vertices = [wp.vec3(v) for v in mesh_points]

        if self.add_cloth:
            self.scene.add_aniso_cloth_mesh(
                vertices=vertices,
                indices=mesh_indices,
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2),
                pos=wp.vec3(0.4, 0.0, 0.2),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=0.5,
                scale=1.0,
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e2) * 5.0,
                edge_aniso_ke=wp.vec3(1.0e-6, 1.0e-6, 1.0e-6) * 4.0 * 5.0,
                panel_verts=mesh_uv.tolist(),
                panel_indices=mesh_uv_indices.tolist(),
                particle_radius=self.cloth_particle_radius,
            )
            self.scene.color()

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.cloth_friction

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.target_joint_q = wp.empty_like(self.state_0.joint_q)
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        self.sim_time = 0.0

        # initialize robot solver
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.cloth_solver: SolverVBD | None = None
        if self.add_cloth:
            # initialize cloth solver
            self.cloth_solver = SolverStyle3D(
                model=self.model,
                iterations=self.iterations,
            )
            self.cloth_solver.precompute(self.scene)
            self.cloth_solver.collision.radius = 3.5e-3

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3((1.5, 0.0, 0.75)), pitch=-30, yaw=180)  # x: left-right, y: forward-backward, z: up-down

        # create Warp arrays for gravity so we can swap Model.gravity during
        # a simulation running under CUDA graph capture
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)  # used for the robot solver
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)  # used for the cloth solver

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.left_planner = None
        self.right_planner = None
        if self.cfg.env.planner == 'curobo':
            self.left_planner = CuroboPlanner(cfg, self.frame_dt, self.left_b2w)
            self.right_planner = CuroboPlanner(cfg, self.frame_dt, self.right_b2w)

        self.prev_target = None
        self.left_path = {}
        self.right_path = {}
        self.plan_step = 0

        # graph capture
        self.graph = None
        if self.add_cloth:
            self.capture()

    def set_up_control(self):
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # we are controlling the velocity
        self.J_out_dim = 6 * self.num_robot
        self.J_in_dim = self.model.joint_dof_count

        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x

        self.Jacobian_one_hots = [onehot(i, self.J_out_dim) for i in range(self.J_out_dim)]

        # for robot control
        self.delta_q = wp.empty(self.model.joint_count, dtype=float)
        self.joint_q_des = wp.array(self.model.joint_q.numpy(), dtype=float)

        # for jacobian computation
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)
        self.body_out = wp.empty(self.J_out_dim, dtype=float, requires_grad=True)

        self.J_flat = wp.empty(self.J_out_dim * self.J_in_dim, dtype=float)
        self.ee_delta = wp.empty(self.num_robot, dtype=wp.spatial_vector)

    def capture(self):
        if self.cfg.env.use_graph and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def compute_body_jacobian(
        self,
        model: Model | Style3DModel,
        joint_q: wp.array,
        joint_qd: wp.array,
    ):
        """
        Compute the Jacobian of the end effector's velocity related to joint_q
        """

        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                compute_body_out, 
                dim=self.num_robot, 
                inputs=[
                    self.temp_state_for_jacobian.body_qd,
                    self.endeffector_id,
                    self.bodies_per_robot,
                    self.endeffector_offset,
                ],
                outputs=[self.body_out]
            )

        for i in range(self.J_out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * self.J_in_dim : (i + 1) * self.J_in_dim], joint_qd.grad)
            tape.zero()

    def generate_joint_control(
        self,
        state_in: State,
        target: np.ndarray,
    ):
        target_wp = wp.array(
            [wp.transform(*target[i * 8 : i * 8 + 7]) for i in range(self.num_robot)], dtype=wp.transform
        )
        wp.launch(
            compute_ee_delta,
            dim=self.num_robot,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_robot,
                target_wp,
            ],
            outputs=[self.ee_delta],
        )
        ee_delta = self.ee_delta.numpy()  # (num_robot, 6)

        xyz = ee_delta[:, :3]
        omega = ee_delta[:, 3:]

        dist = np.linalg.norm(xyz, axis=1)
        w_sq = 1.0 - (omega[:, 0] ** 2 + omega[:, 1] ** 2 + omega[:, 2] ** 2)
        w_sq = np.clip(w_sq, 0.0, 1.0)
        w = np.sqrt(w_sq)
        theta = 2 * np.arccos(w)

        xyz_all = xyz.copy()
        omega_all = omega.copy()
        dist_all = dist.copy()
        theta_all = theta.copy()

        self.compute_body_jacobian(
            self.model,
            state_in.joint_q,
            state_in.joint_qd,
        )
        J_full = self.J_flat.numpy().reshape(-1, self.J_in_dim)
        q_full = state_in.joint_q.numpy()

        qd_list = []
        q_target_list = []

        for ri in range(self.num_robot):

            xyz = xyz_all[ri]
            omega = omega_all[ri]
            theta = theta_all[ri]
            dist = dist_all[ri]
            
            if theta < 1e-6:
                u = np.array([0.0, 0.0, 0.0])
            else:
                u = omega / np.sin(theta / 2)

            # u is the rotation axis, theta is the rotation angle

            alpha = 0.1
            xd = np.zeros(6, dtype=np.float32)
            xd[0] = xyz[0] * alpha / self.frame_dt
            xd[1] = xyz[1] * alpha / self.frame_dt
            xd[2] = xyz[2] * alpha / self.frame_dt
            xd[3] = u[0] * theta * alpha / self.frame_dt
            xd[4] = u[1] * theta * alpha / self.frame_dt
            xd[5] = u[2] * theta * alpha / self.frame_dt

            J = J_full[ri * 6 : (ri + 1) * 6, ri * self.dof_per_robot : (ri + 1) * self.dof_per_robot]
            q = q_full[ri * self.dof_per_robot : (ri + 1) * self.dof_per_robot]
            gripper_target = target[8 * ri + 7]
            
            # compute inverse and add damping
            J_inv = np.linalg.pinv(J)
            lambda_damping = 0.01
            J_inv = J.T @ np.linalg.pinv(J @ J.T + (lambda_damping ** 2) * np.eye(6, dtype=np.float32))

            # no null space control
            qd = J_inv @ xd

            # Apply gripper finger control
            qd[-2] = min(abs(gripper_target - q[-2]), 0.01) / self.frame_dt
            qd[-1] = min(abs(gripper_target - q[-1]), 0.01) / self.frame_dt

            if gripper_target < q[-2]:
                qd[-2] *= -1.0
            if gripper_target < q[-1]:
                qd[-1] *= -1.0

            if dist < 0.01 and theta < 0.05:
                qd[:-2] *= 0.0

            q_target = q + qd * self.frame_dt

            qd_list.append(qd)
            q_target_list.append(q_target)

        qd_full = np.concatenate(qd_list, axis=0)
        q_target_full = np.concatenate(q_target_list, axis=0)

        self.target_joint_qd.assign(qd_full)
        self.target_joint_q.assign(q_target_full)

    def get_state(self):
        return self.state_0.numpy()

    def generate_joint_control_curobo(
        self,
        state_in: State,
        target: np.ndarray,
    ):
        if self.prev_target is None or not np.allclose(self.prev_target, target):
            self.left_path, self.right_path = self.plan_path(state_in, target)
            self.plan_step = 0
            self.prev_target = target.copy()
        
        current_step_left = min(self.plan_step, len(self.left_path['position']) - 1)
        current_step_right = min(self.plan_step, len(self.right_path['position']) - 1)
        left_target_pos = self.left_path['position'][current_step_left]
        right_target_pos = self.right_path['position'][current_step_right]

        left_target_vel = self.left_path['velocity'][current_step_left]
        right_target_vel = self.right_path['velocity'][current_step_right]

        target_pos = np.concatenate([left_target_pos, right_target_pos], axis=0)
        target_vel = np.concatenate([left_target_vel, right_target_vel], axis=0)

        self.target_joint_q.assign(wp.array(target_pos, dtype=float))
        self.target_joint_qd.assign(wp.array(target_vel, dtype=float))
        
        self.plan_step += 1

    def plan_path(
        self,
        state_in: State,
        target: np.ndarray,
    ):
        # use curobo planner
        joint_q = state_in.joint_q.numpy()
        target_pose = target.reshape(self.num_robot, 8)
        target_pose = self.xyzw_to_wxyz(target_pose)

        left_path_result = self.left_planner.plan_path(
            curr_joint_pos=joint_q[:8],
            target_pose=target_pose[0],
        )
        right_path_result = self.right_planner.plan_path(
            curr_joint_pos=joint_q[8:],
            target_pose=target_pose[1],
        )
        if left_path_result['status'] != 'Success' or right_path_result['status'] != 'Success':
            print('Planning failed!')
            import ipdb; ipdb.set_trace()
        
        return left_path_result, right_path_result
    
    def xyzw_to_wxyz(self, pose: np.ndarray):
        if len(pose.shape) == 1 and pose.shape[0] >= 7:
            new_pose = pose.copy()
            new_pose[3] = pose[6]
            new_pose[4] = pose[3]
            new_pose[5] = pose[4]
            new_pose[6] = pose[5]
        elif len(pose.shape) == 1 and pose.shape[0] == 4:
            new_pose = pose.copy()
            new_pose[0] = pose[3]
            new_pose[1] = pose[0]
            new_pose[2] = pose[1]
            new_pose[3] = pose[2]
        elif pose.shape[1] >= 7:
            new_pose = pose.copy()
            new_pose[:, 3] = pose[:, 6]
            new_pose[:, 4] = pose[:, 3]
            new_pose[:, 5] = pose[:, 4]
            new_pose[:, 6] = pose[:, 5]
        elif pose.shape[1] == 4:
            new_pose = pose.copy()
            new_pose[:, 0] = pose[:, 3]
            new_pose[:, 1] = pose[:, 0]
            new_pose[:, 2] = pose[:, 1]
            new_pose[:, 3] = pose[:, 2]
        return new_pose
    
    def wxyz_to_xyzw(self, pose: np.ndarray):
        if len(pose.shape) == 1 and pose.shape[0] >= 7:
            new_pose = pose.copy()
            new_pose[6] = pose[3]
            new_pose[3] = pose[4]
            new_pose[4] = pose[5]
            new_pose[5] = pose[6]
        elif len(pose.shape) == 1 and pose.shape[0] == 4:
            new_pose = pose.copy()
            new_pose[3] = pose[0]
            new_pose[0] = pose[1]
            new_pose[1] = pose[2]
            new_pose[2] = pose[3]
        elif pose.shape[1] >= 7:
            new_pose = pose.copy()
            new_pose[:, 6] = pose[:, 3]
            new_pose[:, 3] = pose[:, 4]
            new_pose[:, 4] = pose[:, 5]
            new_pose[:, 5] = pose[:, 6]
        elif pose.shape[1] == 4:
            new_pose = pose.copy()
            new_pose[:, 3] = pose[:, 0]
            new_pose[:, 0] = pose[:, 1]
            new_pose[:, 1] = pose[:, 2]
            new_pose[:, 2] = pose[:, 3]
        return new_pose
    
    def determine_grasp(self, do_grasp: bool):
        if not self.is_grasped and do_grasp:  # first time grasp
            self.is_grasped = True
            self.lock_particle = True
            self.release_particle = False
        elif self.is_grasped and not do_grasp:  # first time release
            self.is_grasped = False
            self.lock_particle = False
            self.release_particle = True
        elif self.is_grasped and do_grasp:  # continue to grasp
            self.lock_particle = False
            self.release_particle = False
        elif not self.is_grasped and not do_grasp:  # continue to release
            self.lock_particle = False
            self.release_particle = False
        else:
            raise ValueError("Invalid grasp state.")

    def step(self, action: Optional[dict] = None):
        if action is not None and 'target' in action:
            if self.cfg.env.planner == 'diffik':
                self.generate_joint_control(self.state_0, action['target'])

            elif self.cfg.env.planner == 'curobo':
                self.generate_joint_control_curobo(self.state_0, action['target'])

        else:
            raise ValueError("Action must contain 'target' key.")

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def simulate(self):
        if self.add_cloth:
            self.cloth_solver.rebuild_bvh(self.state_0)

        for _step in range(self.sim_substeps):
            # robot sim
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                particle_count = self.model.particle_count
                # set particle_count = 0 to disable particle simulation in robot solver
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                # Update the robot pose - this will modify state_0 and copy to state_1
                self.model.shape_contact_pair_count = 0

                self.control.joint_target_pos.assign(self.target_joint_q)
                self.control.joint_target_vel.assign(self.target_joint_qd)

                n_robot_substeps = 20
                for _robot_substep in range(n_robot_substeps):
                    if _robot_substep == 0:
                        self.contacts = self.model.collide(self.state_0)
                        self.robot_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt / n_robot_substeps)
                    else:
                        self.contacts = self.model.collide(self.state_1)
                        self.robot_solver.step(self.state_1, self.state_1, self.control, self.contacts, self.sim_dt / n_robot_substeps)

                # restore original settings
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            if particle_count > 0:
                self.state_0.particle_f.zero_()  # prevent solver fighting

            # cloth sim
            if self.add_cloth:
                self.contacts = self.model.collide(self.state_0, soft_contact_margin=self.cloth_body_contact_margin)
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()
