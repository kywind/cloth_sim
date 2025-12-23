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
import sim.env
import newton
import newton.utils
import newton.viewer


class ReachEnv:

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        cfg = self.cfg

        if cfg.env.viewer == "gl":
            viewer = newton.viewer.ViewerGL(headless=cfg.env.headless)
        elif cfg.env.viewer == "usd":
            if cfg.env.output_path is None:
                raise ValueError("--output-path is required when using usd viewer")
            viewer = newton.viewer.ViewerUSD(output_path=cfg.env.output_path, num_frames=cfg.env.num_frames)
        elif cfg.env.viewer == "rerun":
            viewer = newton.viewer.ViewerRerun()
        elif cfg.env.viewer == "null":
            viewer = newton.viewer.ViewerNull(num_frames=cfg.env.num_frames)
        else:
            raise ValueError(f"Invalid viewer: {cfg.env.viewer}")

        env = eval(cfg.env.name)(cfg=cfg, viewer=viewer)

        targets = np.array(
            [
                [
                    0.60, -0.25, 0.50, 0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8), 0.01,
                    0.60, 0.25, 0.50, 0.0, 0.0, -np.sin(np.pi / 8), np.cos(np.pi / 8), 0.01,
                    0.5
                ],
                [
                    0.45, -0.17, 0.205, -0.10401744, 0.45360428, 0.26122794, 0.8456853, 0.04,
                    0.45, 0.17, 0.205, 0.10401744, 0.45360428, -0.26122794, 0.845685, 0.04,
                    0.5
                ],
                [
                    0.45, -0.17, 0.205, -0.10401744, 0.45360428, 0.26122794, 0.8456853, 0.01,
                    0.45, 0.17, 0.205, 0.10401744, 0.45360428, -0.26122794, 0.845685, 0.01,
                    0.5
                ],
                [
                    0.45, -0.17, 0.30, -0.10401744, 0.45360428, 0.26122794, 0.8456853, 0.01,
                    0.45, 0.17, 0.30, 0.10401744, 0.45360428, -0.26122794, 0.845685, 0.01,
                    0.5
                ],
                [
                    0.60, -0.17, 0.30, -0.10401744, 0.45360428, 0.26122794, 0.8456853, 0.01,
                    0.60, 0.17, 0.30, 0.10401744, 0.45360428, -0.26122794, 0.845685, 0.01,
                    0.5
                ],
                [
                    0.75, -0.17, 0.30, -0.10401744, 0.45360428, 0.26122794, 0.8456853, 0.01,
                    0.75, 0.17, 0.30, 0.10401744, 0.45360428, -0.26122794, 0.845685, 0.01,
                    0.5
                ],
                [
                    0.75, -0.17, 0.30, -0.10401744, 0.45360428, 0.26122794, 0.8456853, 0.04,
                    0.75, 0.17, 0.30, 0.10401744, 0.45360428, -0.26122794, 0.845685, 0.04,
                    0.5
                ],
                [
                    0.75, -0.17, 0.40, -0.10401744, 0.45360428, 0.26122794, 0.8456853, 0.04,
                    0.75, 0.17, 0.40, 0.10401744, 0.45360428, -0.26122794, 0.845685, 0.04,
                    0.5
                ],
            ], dtype=np.float32
        )  # xyzw

        target_idx = 0
        target = targets[target_idx, :-1].copy()
        to_target_time = targets[target_idx, -1]

        if hasattr(env, "gui") and hasattr(env.viewer, "register_ui_callback"):
            env.viewer.register_ui_callback(lambda ui: env.gui(ui), position="side")

        while env.viewer.is_running():

            if not env.viewer.is_paused():
                with wp.ScopedTimer("step", active=False):
                    env.step({'target': target})
            
            if env.sim_time > to_target_time and target_idx < (targets.shape[0] - 1):
                target_idx += 1
                target = targets[target_idx, :-1].copy()
                to_target_time += targets[target_idx, -1]

            with wp.ScopedTimer("render", active=False):
                env.render()


@hydra.main(version_base='1.2', config_path='../cfg', config_name="default")
def main(cfg):
    teleop = ReachEnv(cfg)
    teleop.run()


if __name__ == "__main__":
    main()
