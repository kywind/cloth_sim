from pathlib import Path
import sys
import importlib
import time

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "newton"))
sys.path.insert(0, str(_ROOT / "rl_games"))

import shutil
import tempfile

import numpy as np
import torch
import hydra
import wandb
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from experiments.utils.ffmpeg import make_video

from sim.renderer.default_renderer import DefaultRenderer
from rl_games.torch_runner import Runner


@hydra.main(version_base="1.2", config_path=str(_ROOT / "cfg"), config_name="train_vec")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    # --- eval-specific overrides ---
    use_wandb     = bool (OmegaConf.select(cfg, "eval.wandb",         default=False))
    n_episodes    = int  (OmegaConf.select(cfg, "eval.n_episodes",    default=10))
    video_fps     =       OmegaConf.select(cfg, "eval.fps",           default=None)
    deterministic = bool (OmegaConf.select(cfg, "eval.deterministic", default=True))
    slowmo        = float(OmegaConf.select(cfg, "eval.slowmo",        default=0.0))

    checkpoint = OmegaConf.select(cfg, "train.checkpoint", default=None)
    if not checkpoint:
        raise ValueError("Provide train.checkpoint=<path.pth>")

    # Infer the training run name from the checkpoint path:
    #   log/runs/<run_name>/nn/<file>.pth  →  run_name
    checkpoint_path = Path(checkpoint)
    train_run_name = checkpoint_path.parents[1].name  # e.g. "franka_pickup_20240311_142500"

    # Load the saved training config if present (for reference / wandb logging)
    train_config_path = checkpoint_path.parents[1] / "train_config.yaml"
    if train_config_path.exists():
        train_cfg_saved = OmegaConf.load(train_config_path)
    else:
        train_cfg_saved = None

    # Single env with the viewer open
    cfg.env.num_envs = 1
    cfg.env.headless = False

    # Dynamically import and register the IVecEnv class.
    module_path, class_name = cfg.rl.params.config.env_class.rsplit(".", 1)
    VecEnvClass = getattr(importlib.import_module(module_path), class_name)
    vecenv_type = cfg.rl.params.config.vecenv_type
    env_name = cfg.rl.params.config.env_name
    VecEnvClass.register(vecenv_type, env_name, cfg)

    # Disable training-specific noise for deterministic eval
    if deterministic:
        cfg.rl.params.config.player = {
            "deterministic": True,
            "games_num": n_episodes,
        }

    runner = Runner()
    runner.load(
        OmegaConf.to_container(cfg.rl, resolve=True)
    )

    player = runner.create_player()

    # Restore weights
    player.restore(checkpoint)
    player.reset()

    # --- build renderer ---
    # Wait until _make_env has been called (it's called lazily inside player.run())
    # We intercept by wrapping player.env_reset so we can grab frames.
    # Instead: manually drive the rollout loop here so we control frame capture.

    vec_env = player.env  # IVecEnv wrapper
    sim_env = vec_env._env  # underlying vectorized env

    video_fps = int(video_fps) if video_fps is not None else int(sim_env.fps)
    print(f"[rl_eval] video_fps={video_fps}  sim_env.fps={sim_env.fps}")

    renderer = DefaultRenderer(cfg=None)
    renderer.initialize_resources(sim_env)

    frames = []
    episode_rewards = []
    episode_successes = []

    obs = vec_env.reset()
    obs_torch = torch.tensor(obs, dtype=torch.float32, device=player.device)

    ep_reward = 0.0
    ep_count  = 0
    frame     = 0

    print(f"\n[rl_eval] Running {n_episodes} episodes from {checkpoint}\n")

    while ep_count < n_episodes:
        # Policy inference
        with torch.no_grad():
            action = player.get_action(obs_torch, is_deterministic=deterministic)

        action_np = action.cpu().numpy().reshape(1, -1)
        obs, reward, done, info = vec_env.step(action_np)
        obs_torch = torch.tensor(obs, dtype=torch.float32, device=player.device)
        ep_reward += float(reward[0])
        frame += 1

        components = vec_env._task.reward_components(sim_env)
        comp_str = "  ".join(f"{k}={v[0]:.3f}" for k, v in components.items())
        print(
            f"\n  ep={ep_count+1}  frame={frame:5d}  {comp_str}"
            f"  reward={float(reward[0]):.3f}  done={bool(done[0])}",
            flush=True,
        )

        # Render and capture frame
        result = renderer.render(sim_env, return_renderings=True)
        if result and "rgba" in result:
            frames.append(result["rgba"][..., :3])  # drop alpha → (H, W, 3)

        if slowmo > 0.0:
            time.sleep(slowmo)

        if done[0]:
            success = bool(info.get("episode", {}).get("percent_success", torch.tensor([0]))[0])
            episode_rewards.append(ep_reward)
            episode_successes.append(success)
            print(f"  Episode {ep_count + 1:3d}  reward={ep_reward:7.2f}  success={success}")
            ep_reward = 0.0
            ep_count += 1
            obs = vec_env.reset()
            obs_torch = torch.tensor(obs, dtype=torch.float32, device=player.device)

    mean_reward  = float(np.mean(episode_rewards))
    success_rate = float(np.mean(episode_successes))
    print(f"\n[rl_eval] mean_reward={mean_reward:.2f}  success_rate={success_rate:.2%}\n")

    # --- save video locally (always) ---
    if frames:
        video_dir = checkpoint_path.parents[1] / "eval_videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = video_dir / f"eval_{timestamp}.mp4"

        tmp_dir = Path(tempfile.mkdtemp())
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(tmp_dir / f"{i:06d}.png")
        make_video(tmp_dir, video_path, image_pattern="%06d.png", frame_rate=video_fps)
        shutil.rmtree(tmp_dir)
        print(f"[rl_eval] Video saved → {video_path} ({len(frames)} frames)")

    if use_wandb and frames:
        run = wandb.init(
            project=cfg.rl.params.wandb.project,
            entity=cfg.rl.params.wandb.entity,
            name=f"eval_{train_run_name}",
            config={
                "checkpoint": checkpoint,
                "train_run": train_run_name,
                "n_episodes": n_episodes,
                "deterministic": deterministic,
            },
        )

        video_array = np.stack(frames, axis=0)  # (T, H, W, 3)
        wandb.log({
            "eval/rollout_video": wandb.Video(video_array, fps=video_fps, format="mp4"),
            "eval/mean_reward":   mean_reward,
            "eval/success_rate":  success_rate,
        })
        print(f"[rl_eval] wandb run → {run.url}")
        wandb.finish()


if __name__ == "__main__":
    main()
