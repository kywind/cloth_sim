# Cloth Manipulation Simulation Environment

A physics-based simulation environment for **bimanual cloth manipulation** with deformable contact-rich interactions. Built on [NVIDIA Newton](https://github.com/newton-physics/newton) and designed for the [2026 WBCD competition](https://wbcdcompetition.github.io/) deformable manipulation track ([task details](https://wbcdcompetition.github.io/competition-tracks.html#dm)).

> **Note:** Newton is in active beta development. This repo is subject to updates and may need case-specific integration into existing robot learning frameworks.

https://github.com/user-attachments/assets/c8cdb991-998f-4cec-9853-c7a65b4a8f7d

---

## Installation

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:kywind/cloth_sim.git
cd cloth_sim

# Create and activate a Python 3.11 venv
uv venv --python=3.11
source .venv/bin/activate

# Install Newton
cd newton && uv pip install -e ".[examples]" && cd ..

# (Optional) Verify Newton installation
cd newton && python -m newton.examples robot_h1 && cd ..

# Install PyTorch (CUDA 12.8)
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install additional dependencies
uv pip install opencv-python omegaconf hydra-core pynput transforms3d

# Version fixes
uv pip install imgui_bundle==1.92.5
uv pip install --upgrade warp-lang

# Install rl_games (for RL training and eval)
cd rl_games && uv pip install -e . && cd ..
```

---

## Quick Start

Run a scripted demo with two ARX arms manipulating a cloth garment:

```bash
# Default (IK controller, GL renderer)
python experiments/demo.py

# Multiple parallel environments
python experiments/demo.py env.num_envs=4

# Joint-space control
python experiments/demo.py controller=joint_pd env.num_envs=4

# Headless with frame saving
python experiments/demo.py env.headless=True save_state=True

# Convert saved frames to video
python experiments/make_video.py log/experiments/output_demo/<timestamp> --fps 60 --output output.mp4
```

**Renderer options:**

```bash
# Tiled camera renderer (fast, parallelizable, supports depth)
python experiments/demo.py renderer=tiled_camera_renderer

# Tiled camera renderer without random shape colors
python experiments/demo.py renderer=tiled_camera_renderer renderer.colors_per_shape=False
```

https://github.com/user-attachments/assets/21b3ced4-b4a9-4085-85cb-d2ad1032deb4

---

## Keyboard Teleoperation

```bash
python experiments/teleop_keyboard.py                  # single env
python experiments/teleop_keyboard.py env.num_envs=4   # multi-env
```

**Key mappings:**

| Keys | Action |
|---|---|
| `1` / `2` | Switch to left / right arm |
| `i` / `k` | Forward (farther) / backward (closer) |
| `j` / `l` | Left / right |
| `p` / `;` | Up / down |
| `z` / `x` | Rotate around z-axis |
| `c` / `v` | Rotate around x-axis |
| `b` / `n` | Rotate around y-axis |
| `,` / `.` | Open / close gripper |

All Hydra config overrides (e.g. `env.num_envs`, `controller`, `renderer`) work the same as in the demo.

---

## RL Training and Evaluation

### Task

The `ClothLoadingTask` (`sim/task/cloth_loading_task.py`) trains two ARX arms to drape a cloth garment over a board. Task parameters are configured in `cfg/task/cloth_loading.yaml`.

**Observation** (flattened vector per env):

| Component | Dim | Description |
|---|---|---|
| `left_ee_pos` | 3 | Left end-effector position |
| `left_ee_quat` | 4 | Left end-effector orientation (xyzw) |
| `right_ee_pos` | 3 | Right end-effector position |
| `right_ee_quat` | 4 | Right end-effector orientation (xyzw) |
| `left_ee_to_cloth` | 3 | Vector from left EE to nearest cloth vertex |
| `right_ee_to_cloth` | 3 | Vector from right EE to nearest cloth vertex |
| `left_gripper_width` | 1 | Left gripper finger sum |
| `right_gripper_width` | 1 | Right gripper finger sum |
| `cloth_keypoints` | num_keypoints * 3 | Cloth keypoint positions (farthest-point sampled) |

**Action** (14-dim): `[left_7, right_7]`, each `[dx, dy, dz, drx, dry, drz, gripper]` in `[-1, 1]`, mapped to delta EE pose targets.

**Reward**: `sum(exp(-10 * d_i))` over 4 board edge midpoints, where `d_i` is the min distance to any cloth vertex. Range: [0, 4].

### Training

Uses [rl_games](https://github.com/Denys88/rl_games) PPO. Config is composed from `cfg/train_vec.yaml`, `cfg/rl/ppo.yaml`, `cfg/task/cloth_loading.yaml`, and `cfg/env/cloth_env_ARX.yaml`.

```bash
# Basic training (rl.params.wandb.entity is required)
python experiments/rl_train.py env=cloth_env_ARX task=cloth_loading controller=ik \
    train.exp_name=cloth_loading rl.params.wandb.entity=<your_wandb_entity>

# Scale up parallel environments
python experiments/rl_train.py env=cloth_env_ARX task=cloth_loading controller=ik \
    train.exp_name=cloth_loading env.num_envs=128 rl.params.wandb.entity=<your_wandb_entity>

# Resume from checkpoint
python experiments/rl_train.py env=cloth_env_ARX task=cloth_loading controller=ik \
    train.exp_name=cloth_loading train.checkpoint=log/runs/<run_name>/nn/<file>.pth \
    rl.params.wandb.entity=<your_wandb_entity>
```

Logs and checkpoints are saved to `log/runs/<run_name>/` and synced to Weights & Biases.

> **Note:** `batch_size = num_envs * horizon_length` must be divisible by `minibatch_size` (default 64, set in `cfg/rl/ppo.yaml`). When scaling up `num_envs`, you may want to increase `minibatch_size` accordingly (e.g. `rl.params.config.minibatch_size=1024` for `num_envs=128`).

### Evaluation

```bash
# Evaluate with viewer
python experiments/rl_eval.py env=cloth_env_ARX task=cloth_loading controller=ik \
    train.checkpoint=log/runs/<run_name>/nn/<file>.pth

# Multiple episodes
python experiments/rl_eval.py env=cloth_env_ARX task=cloth_loading controller=ik \
    train.checkpoint=log/runs/<run_name>/nn/<file>.pth eval.n_episodes=10

# Log to wandb
python experiments/rl_eval.py env=cloth_env_ARX task=cloth_loading controller=ik \
    train.checkpoint=log/runs/<run_name>/nn/<file>.pth eval.wandb=true
```

Eval opens a single-env viewer, prints per-step reward components, and saves videos to `log/runs/<run_name>/eval_videos/`.

---

## Scene Layout

The scene (robots, cloth, table, board) is configured in `cfg/env/cloth_env_ARX.yaml`. Each asset entry specifies its type, mesh path, and 4x4 pose matrix. Key parameters:

- `table_height` -- height of the table surface (all assets are placed relative to this)
- `assets` -- list of scene objects (URDFs, rigid meshes, cloth meshes)
- `num_robot`, `num_arm_joints`, `num_gripper_joints` -- robot configuration

---

## Contact

For questions, please [open an issue](https://github.com/kywind/cloth_sim/issues) or contact [Kaifeng Zhang](https://kywind.github.io/).
