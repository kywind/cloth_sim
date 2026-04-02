import os
import cv2
import numpy as np
import warp as wp
from omegaconf import OmegaConf
import hydra
import transforms3d as t3d
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
OmegaConf.register_new_resolver("eval", eval, replace=True)
from experiments.utils.dir_utils import mkdir
import sim.env
import newton
import newton.utils
import newton.viewer


class DemoEnv:

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.save_state:
            self.cnt = 0
            self.episode_id = 0
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.out_path = os.path.join(cfg.exp_root, 'output_demo', f'{timestamp}')
            mkdir(Path(f'{self.out_path}'), resume=False, overwrite=False)
            OmegaConf.save(cfg, f'{self.out_path}/hydra.yaml', resolve=True)

    def _make_ik_targets(self, cfg):
        table_height = float(cfg.env.get("table_height", 0.0))
        targets = np.array(
            [
                [
                    0.50, -0.25, 0.40 + table_height, 0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4), 0.04,
                    0.50, 0.25, 0.40 + table_height, 0.0, 0.0, -np.sin(np.pi / 4), np.cos(np.pi / 4), 0.04,
                    0.5
                ],
                [
                    0.70, -0.40, 0.40 + table_height, 0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8), 0.04,
                    0.70, 0.40, 0.40 + table_height, 0.0, 0.0, -np.sin(np.pi / 8), np.cos(np.pi / 8), 0.04,
                    0.5
                ],
            ], dtype=np.float32
        )  # xyzw
        return targets

    def _make_joint_pd_targets(self, env, num_targets=5, duration=1.0):
        rng = np.random.default_rng(seed=42)
        init_q = np.array(self.cfg.env.assets[1].init_q, dtype=np.float32)  # first robot arm init_q
        num_arm = env.num_arm_joints
        num_gripper = env.num_gripper_joints
        num_joints = num_arm + num_gripper

        # Joint limits (from URDF): arm joints [-pi, pi], gripper [0, 0.04]
        arm_lower = np.full(num_arm, -np.pi, dtype=np.float32)
        arm_upper = np.full(num_arm, np.pi, dtype=np.float32)
        gripper_lower = np.full(num_gripper, 0.0, dtype=np.float32)
        gripper_upper = np.full(num_gripper, 0.04, dtype=np.float32)

        rows = []
        for _ in range(num_targets):
            per_robot = []
            for _ in range(env.num_robot):
                arm_q = init_q[:num_arm] + rng.uniform(-0.5, 0.5, size=num_arm).astype(np.float32)
                arm_q = np.clip(arm_q, arm_lower, arm_upper)
                grip_q = rng.uniform(gripper_lower, gripper_upper).astype(np.float32)
                per_robot.append(np.concatenate([arm_q, grip_q]))
            row = np.concatenate([*per_robot, [duration]])
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    def run(self):
        cfg = self.cfg

        env = eval(cfg.env.name)(cfg=cfg)
        env.initialize_resources()

        is_joint_pd = cfg.controller.name == "sim.controller.JointPDController"
        if is_joint_pd:
            targets = self._make_joint_pd_targets(env)
        else:
            targets = self._make_ik_targets(cfg)

        target_idx = 0
        target = targets[target_idx, :-1].copy()
        to_target_time = targets[target_idx, -1]

        while env.viewer.is_running():

            if not env.viewer.is_paused():
                env.step({'target': np.tile(target, (env.num_envs, 1))})

            if env.sim_time > to_target_time and target_idx < (targets.shape[0] - 1):
                target_idx += 1
                target = targets[target_idx, :-1].copy()
                to_target_time += targets[target_idx, -1]

            render_result = env.render(return_renderings=self.cfg.save_state)

            if self.cfg.save_state:
                assert 'rgba' in render_result
                rgba = render_result['rgba']
                cv2.imwrite(
                    f'{self.out_path}/{self.cnt:06d}.png',
                    cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                )
                self.cnt += 1


@hydra.main(version_base='1.2', config_path='../cfg', config_name="default")
def main(cfg):
    demo_env = DemoEnv(cfg)
    demo_env.run()


if __name__ == "__main__":
    main()
