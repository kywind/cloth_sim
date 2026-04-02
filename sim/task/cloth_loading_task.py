import numpy as np
import torch
import gymnasium

from .base_task import BaseTask
from ..utils.env_utils import pose_to_pos_quat, axis_angle_to_quat_xyzw, quat_mul_xyzw


class ClothLoadingTask(BaseTask):
    """
    Cloth loading task for ClothEnvARX — drape cloth over a board.

    Reward: negative sum of minimum distances from each of the four board edge
    midpoints (±175 mm along x, ±125 mm along y from board centre) to the
    nearest cloth vertex.  Lower (more negative) distance → higher reward.

    Supports num_envs > 1 (vectorised). All returns are (num_envs, ...) shaped.
    """

    _GRIPPER_LO = 0.005
    _GRIPPER_HI = 0.04

    OBS_DIM = None  # set in initialize_resources based on num_keypoints
    ACT_DIM = 14   # 7 per arm × 2 arms

    def __init__(self, cfg=None):
        super().__init__()
        task_cfg = getattr(cfg, "task", None) if cfg is not None else None

        def _get(name, default):
            if task_cfg is not None and hasattr(task_cfg, name):
                return getattr(task_cfg, name)
            return default

        self.action_repeat   = _get("action_repeat", 1)
        self.time_limit      = _get("time_limit", 20.0)
        self.max_xyz_delta   = _get("max_xyz_delta", 0.005)
        self.max_rot_delta   = _get("max_rot_delta", 0.01)
        self.num_keypoints   = _get("num_keypoints", 16)

        self._left_ee_indices = None
        self._right_ee_indices = None
        self._env_time = None

        self._left_ee_target_pos = None
        self._left_ee_target_quat = None
        self._right_ee_target_pos = None
        self._right_ee_target_quat = None

    def initialize_resources(self, env) -> None:
        assert env.num_robot == 2, "ClothLoadingTask requires num_robot=2 (bimanual)"
        assert env._particles_per_world > 0, "ClothLoadingTask requires a cloth asset"

        n = env.num_envs
        bodies_per_world = env.scene.body_count

        left_ee_local = env.endeffector_id
        left_arm_body_count = env._single_arm_models[0].body_count
        right_ee_local = left_arm_body_count + env.endeffector_id

        self._left_ee_indices = np.array(
            [w * bodies_per_world + left_ee_local for w in range(n)],
            dtype=np.int64,
        )
        self._right_ee_indices = np.array(
            [w * bodies_per_world + right_ee_local for w in range(n)],
            dtype=np.int64,
        )

        self._env_time = np.zeros(n, dtype=np.float64)
        self._ep_len = np.zeros(n, dtype=np.int32)

        # Seed EE targets from initial FK state
        body_q = env.state_0.body_q.numpy()
        self._left_ee_init_pos = body_q[self._left_ee_indices[0], :3].astype(np.float32)
        self._left_ee_init_quat = body_q[self._left_ee_indices[0], 3:7].astype(np.float32)
        self._right_ee_init_pos = body_q[self._right_ee_indices[0], :3].astype(np.float32)
        self._right_ee_init_quat = body_q[self._right_ee_indices[0], 3:7].astype(np.float32)

        self._left_ee_target_pos = np.tile(self._left_ee_init_pos, (n, 1))
        self._left_ee_target_quat = np.tile(self._left_ee_init_quat, (n, 1))
        self._right_ee_target_pos = np.tile(self._right_ee_init_pos, (n, 1))
        self._right_ee_target_quat = np.tile(self._right_ee_init_quat, (n, 1))

        # Board edge midpoints: board centre from the "board" asset pose,
        # edges at ±175 mm (x) and ±125 mm (y).
        board_pos = None
        for asset in env.cfg.env.assets:
            if getattr(asset, "name", None) == "board":
                pos, _ = pose_to_pos_quat(getattr(asset, "pose", None))
                board_pos = pos
                break
        assert board_pos is not None, "ClothLoadingTask requires a 'board' asset"

        self._board_edges = np.array([
            [board_pos[0] + 0.175, board_pos[1],         board_pos[2]],
            [board_pos[0] - 0.175, board_pos[1],         board_pos[2]],
            [board_pos[0],         board_pos[1] + 0.125, board_pos[2]],
            [board_pos[0],         board_pos[1] - 0.125, board_pos[2]],
        ], dtype=np.float32)  # (4, 3)

        # Select cloth keypoint indices via farthest-point sampling on the
        # initial mesh so they are spread evenly across the garment.
        ppw = env._particles_per_world
        init_particles = env.state_0.particle_q.numpy()[:ppw]  # world-0
        self._keypoint_indices = self._farthest_point_sample(init_particles, self.num_keypoints)

        # 22 base dims + num_keypoints * 3
        self.OBS_DIM = 22 + self.num_keypoints * 3

        self.observation_space = gymnasium.spaces.Box(
            low=np.full(self.OBS_DIM, -np.inf, dtype=np.float32),
            high=np.full(self.OBS_DIM, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=np.full(self.ACT_DIM, -1.0, dtype=np.float32),
            high=np.full(self.ACT_DIM, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, env, mask=None) -> None:
        if mask is None:
            self._env_time[:] = 0.0
            self._left_ee_target_pos[:] = self._left_ee_init_pos
            self._left_ee_target_quat[:] = self._left_ee_init_quat
            self._right_ee_target_pos[:] = self._right_ee_init_pos
            self._right_ee_target_quat[:] = self._right_ee_init_quat
        else:
            self._env_time[mask] = 0.0
            self._left_ee_target_pos[mask] = self._left_ee_init_pos
            self._left_ee_target_quat[mask] = self._left_ee_init_quat
            self._right_ee_target_pos[mask] = self._right_ee_init_pos
            self._right_ee_target_quat[mask] = self._right_ee_init_quat

    @staticmethod
    def _farthest_point_sample(points: np.ndarray, k: int) -> np.ndarray:
        """Return indices of k points selected by farthest-point sampling."""
        n = len(points)
        k = min(k, n)
        selected = [0]
        min_dists = np.full(n, np.inf)
        for _ in range(k - 1):
            d = np.linalg.norm(points - points[selected[-1]], axis=1)
            min_dists = np.minimum(min_dists, d)
            selected.append(int(np.argmax(min_dists)))
        return np.array(selected, dtype=np.int64)

    @staticmethod
    def _quat_rotate_batch(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Rotate vec by quaternion (xyzw) for a batch of quaternions (N, 4)."""
        qvec = quat[:, :3]  # (N, 3)
        w = quat[:, 3:4]    # (N, 1)
        uv = np.cross(qvec, vec)           # (N, 3)
        uuv = np.cross(qvec, uv)           # (N, 3)
        return vec + 2.0 * (w * uv + uuv)  # (N, 3)

    def _get_ee(self, env):
        body_q = env.state_0.body_q.numpy()
        return (
            body_q[self._left_ee_indices, :3],
            body_q[self._left_ee_indices, 3:7],
            body_q[self._right_ee_indices, :3],
            body_q[self._right_ee_indices, 3:7],
        )

    def _get_cloth_particles(self, env):
        particle_q = env.state_0.particle_q.numpy()
        ppw = env._particles_per_world
        return particle_q.reshape(env.num_envs, ppw, 3)

    def _get_gripper_widths(self, env):
        joint_q = env.state_0.joint_q.numpy()
        robot_q = joint_q.reshape(env.num_envs, env._coords_per_world)
        nj = env.num_joints
        left_fingers = robot_q[:, env.num_arm_joints:nj]
        right_fingers = robot_q[:, nj + env.num_arm_joints:2 * nj]
        return (
            left_fingers.sum(axis=1, keepdims=True),
            right_fingers.sum(axis=1, keepdims=True),
        )

    def observation(self, env) -> dict:
        l_pos, l_quat, r_pos, r_quat = self._get_ee(env)
        cloth_pos = self._get_cloth_particles(env)
        l_gw, r_gw = self._get_gripper_widths(env)

        # Compute EE tip positions by applying a local offset along the EE frame
        tip_offset_local = np.array([0.155, 0.0, 0.0], dtype=np.float32)
        l_tip = l_pos + self._quat_rotate_batch(l_quat, tip_offset_local)
        r_tip = r_pos + self._quat_rotate_batch(r_quat, tip_offset_local)

        l_dists = np.linalg.norm(cloth_pos - l_tip[:, None, :], axis=2)
        l_nearest = l_dists.argmin(axis=1)
        l_ee_to_cloth = cloth_pos[np.arange(env.num_envs), l_nearest] - l_tip

        r_dists = np.linalg.norm(cloth_pos - r_tip[:, None, :], axis=2)
        r_nearest = r_dists.argmin(axis=1)
        r_ee_to_cloth = cloth_pos[np.arange(env.num_envs), r_nearest] - r_tip

        # Cloth keypoint positions: (N, num_keypoints, 3) → (N, num_keypoints * 3)
        cloth_keypoints = cloth_pos[:, self._keypoint_indices, :].reshape(env.num_envs, -1)

        return {
            "left_ee_pos":         l_pos.astype(np.float32),
            "left_ee_quat":        l_quat.astype(np.float32),
            "right_ee_pos":        r_pos.astype(np.float32),
            "right_ee_quat":       r_quat.astype(np.float32),
            "left_ee_to_cloth":    l_ee_to_cloth.astype(np.float32),
            "right_ee_to_cloth":   r_ee_to_cloth.astype(np.float32),
            "left_gripper_width":  l_gw.astype(np.float32),
            "right_gripper_width": r_gw.astype(np.float32),
            "cloth_keypoints":     cloth_keypoints.astype(np.float32),
        }

    def flatten_obs(self, obs_dict: dict) -> np.ndarray:
        return np.concatenate([
            obs_dict["left_ee_pos"],
            obs_dict["left_ee_quat"],
            obs_dict["right_ee_pos"],
            obs_dict["right_ee_quat"],
            obs_dict["left_ee_to_cloth"],
            obs_dict["right_ee_to_cloth"],
            obs_dict["left_gripper_width"],
            obs_dict["right_gripper_width"],
            obs_dict["cloth_keypoints"],
        ], axis=1).astype(np.float32)

    def _apply_delta(self, actions_7, target_pos, target_quat):
        xyz_delta = actions_7[:, :3] * self.max_xyz_delta
        rot_delta = actions_7[:, 3:6] * self.max_rot_delta
        gripper_norm = (actions_7[:, 6:7] + 1.0) * 0.5
        gripper = self._GRIPPER_LO + gripper_norm * (self._GRIPPER_HI - self._GRIPPER_LO)

        new_pos = target_pos + xyz_delta
        dq = axis_angle_to_quat_xyzw(rot_delta)
        new_quat = quat_mul_xyzw(dq, target_quat)
        new_quat /= np.linalg.norm(new_quat, axis=1, keepdims=True)

        target_pos[:] = new_pos
        target_quat[:] = new_quat

        return new_pos, new_quat, gripper

    def scale_actions(self, actions: np.ndarray, env) -> np.ndarray:
        left_actions = actions[:, :7]
        right_actions = actions[:, 7:14]

        l_pos, l_quat, l_grip = self._apply_delta(
            left_actions, self._left_ee_target_pos, self._left_ee_target_quat
        )
        r_pos, r_quat, r_grip = self._apply_delta(
            right_actions, self._right_ee_target_pos, self._right_ee_target_quat
        )

        return np.concatenate([
            l_pos, l_quat, l_grip,
            r_pos, r_quat, r_grip,
        ], axis=1).astype(np.float32)

    def _board_edge_distances(self, env):
        """Min distance from each board edge midpoint to nearest cloth vertex. (N, 4)."""
        cloth_pos = self._get_cloth_particles(env)  # (N, P, 3)
        edges = self._board_edges  # (4, 3)
        diff = cloth_pos[:, None, :, :] - edges[None, :, None, :]  # (N, 4, P, 3)
        dists = np.linalg.norm(diff, axis=3)  # (N, 4, P)
        return dists.min(axis=2)  # (N, 4)

    def reward_components(self, env) -> dict:
        min_dists = self._board_edge_distances(env)  # (N, 4)
        return {
            "edge_dist": np.exp(-10.0 * min_dists).sum(axis=1).astype(np.float32),
        }

    def reward(self, env) -> np.ndarray:
        return self.reward_components(env)["edge_dist"]

    def terminated(self, env) -> np.ndarray:
        return np.zeros(env.num_envs, dtype=bool)

    def truncated(self, env) -> np.ndarray:
        self._env_time += env.frame_dt * self.action_repeat
        return self._env_time >= self.time_limit

    def build_episode_info(self, done_mask: np.ndarray, terminated: np.ndarray) -> dict:
        idx = np.where(done_mask)[0]
        return {
            "avg_length": torch.tensor(self._ep_len[idx].astype(np.float32)),
        }

    def update_episode_metrics(self, components: dict) -> None:
        self._ep_len += 1

    def reset_episode_metrics(self, mask: np.ndarray) -> None:
        self._ep_len[mask] = 0
