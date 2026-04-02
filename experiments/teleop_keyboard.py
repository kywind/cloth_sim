import os
import cv2
import numpy as np
import warp as wp
from omegaconf import OmegaConf
import hydra
import transforms3d as t3d
from datetime import datetime
from pathlib import Path
from pynput import keyboard
import sys
sys.path.append(str(Path(__file__).parents[1]))
OmegaConf.register_new_resolver("eval", eval, replace=True)
from experiments.utils.dir_utils import mkdir
import sim.env
import newton
import newton.utils
import newton.viewer


class KeyboardTeleopEnv:

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.save_state:
            self.cnt = 0
            self.episode_id = 0
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.out_path = os.path.join(cfg.exp_root, 'output_teleop_keyboard', f'{timestamp}')
            mkdir(Path(f'{self.out_path}'), resume=False, overwrite=False)
            OmegaConf.save(cfg, f'{self.out_path}/hydra.yaml', resolve=True)

    def on_press(self, key):
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.remove(key.char)
        except (KeyError, AttributeError):
            try:
                self.pressed_keys.remove(str(key))
            except KeyError:
                pass

    def get_trans_change(self):
        for key in self.pressed_keys:
            if key in ["i", "j", "k", "l", "p", ";"]:
                print(f"Pressed key: {key}")
                return np.array(self.key_mappings[key])[None]
        return None

    def get_finger_change(self):
        for key in self.pressed_keys:
            if key in [",", "."]:
                print(f"Pressed key: {key}")
                return self.key_mappings[key]
        return 0.0

    def get_rot_change(self):
        for key in self.pressed_keys:
            if key in ["z", "x", "c", "v", "b", "n"]:
                print(f"Pressed key: {key}")
                return np.array(self.key_mappings[key])
        return None

    def get_arm_change(self):
        for key in self.pressed_keys:
            if key in ["1", "2"]:
                print(f"Pressed key: {key}")
                return int(key) - 1
        return None

    def run(self):
        cfg = self.cfg

        env = eval(cfg.env.name)(cfg=cfg)
        env.initialize_resources()

        v = 0.01  # m per step
        w = 2 / 180 * np.pi  # rad per step
        g = 0.002  # gripper per step
        self.key_mappings = {
            # Set translation controls
            "i": np.array([-v, 0, 0]),
            "k": np.array([v, 0, 0]),
            "j": np.array([0, -v, 0]),
            "l": np.array([0, v, 0]),
            ";": np.array([0, 0, -v]),
            "p": np.array([0, 0, v]),

            # Set the finger
            ",": g,
            ".": -g,

            # Set the rotation
            "z": [0, 0, 1, w],
            "x": [0, 0, 1, -w],
            "c": [1, 0, 0, w],
            "v": [1, 0, 0, -w],
            "b": [0, 1, 0, w],
            "n": [0, 1, 0, -w]
        }
        self.pressed_keys = set()

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        if cfg.env.num_robot == 2:  # ARX
            table_height = float(cfg.env.get("table_height", 0.0))
            targets = np.array(
                [
                    [
                        0.50, -0.25, 0.40 + table_height, 0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4), 0.04,
                        0.50, 0.25, 0.40 + table_height, 0.0, 0.0, -np.sin(np.pi / 4), np.cos(np.pi / 4), 0.04,
                    ],
                ], dtype=np.float32
            )  # xyzw
            target = targets[0].copy()
            arm_id = 0

        else:
            raise NotImplementedError

        while env.viewer.is_running():

            trans_change = self.get_trans_change()
            rot_change = self.get_rot_change()
            finger_change = self.get_finger_change()
            arm_id_change = self.get_arm_change()

            if arm_id_change is not None:
                arm_id = arm_id_change
                print(f"Switched to arm {arm_id}")

            if trans_change is not None:
                target[arm_id * 8:arm_id * 8 + 3] = target[arm_id * 8:arm_id * 8 + 3] + trans_change[0]
            if rot_change is not None:
                quat_change = t3d.quaternions.axangle2quat(rot_change[:3], rot_change[3])
                target[arm_id * 8 + 3:arm_id * 8 + 7] = np.roll(
                    t3d.quaternions.qmult(
                        quat_change, 
                        np.roll(target[arm_id * 8 + 3:arm_id * 8 + 7], 1)
                    ),
                    -1
                )

            target[arm_id * 8 + 7] = target[arm_id * 8 + 7] + finger_change
            target[arm_id * 8 + 7] = np.clip(target[arm_id * 8 + 7], cfg.env.gripper_min, cfg.env.gripper_max)

            print(f"Target: {target}")

            if not env.viewer.is_paused():
                env.step({'target': np.tile(target, (env.num_envs, 1))})

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
    teleop_env = KeyboardTeleopEnv(cfg)
    teleop_env.run()


if __name__ == "__main__":
    main()
