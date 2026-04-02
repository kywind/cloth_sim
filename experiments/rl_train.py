from pathlib import Path
import sys
import importlib

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "newton"))
sys.path.insert(0, str(_ROOT / "rl_games"))

import hydra
import wandb
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner


@hydra.main(version_base="1.2", config_path=str(_ROOT / "cfg"), config_name="train_vec")
def main(cfg: DictConfig) -> None:
    # Dynamically import and register the IVecEnv class
    module_path, class_name = cfg.rl.params.config.env_class.rsplit(".", 1)
    VecEnvClass = getattr(importlib.import_module(module_path), class_name)
    vecenv_type = cfg.rl.params.config.vecenv_type
    env_name = cfg.rl.params.config.env_name
    VecEnvClass.register(vecenv_type, env_name, cfg)

    # Overwrite rl configs
    cfg.env.headless = True
    cfg.rl.params.config.num_actors = int(cfg.env.num_envs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_name = cfg.rl.params.wandb.name or f"{env_name}_{timestamp}"

    # Force rl_games to use the same name as the wandb run for its experiment dir,
    # so checkpoints and logs live at log/runs/<wandb_name>/
    cfg.rl.params.config.full_experiment_name = wandb_name

    run = wandb.init(
        project=cfg.rl.params.wandb.project,
        entity=cfg.rl.params.wandb.entity,
        name=wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,
        save_code=True,
    )

    # Save the full Hydra config alongside checkpoints so eval can reconstruct it
    run_dir = _ROOT / "log" / "runs" / wandb_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_config.yaml").write_text(OmegaConf.to_yaml(cfg))

    runner = Runner(algo_observer=IsaacAlgoObserver())
    runner.load(
        OmegaConf.to_container(cfg.rl, resolve=True)
    )

    print("")
    print(f"[rl_train] rl_task   → {cfg.rl.params.config.env_class}")
    print(f"[rl_train] rl algo   → {cfg.rl.params.algo.name}")
    print(f"[rl_train] run dir   → {run_dir}")
    print(f"[rl_train] wandb run → {run.url}")
    print("")

    try:
        runner.run({
            "train":     True,
            "play":      False,
            "checkpoint": cfg.train.checkpoint,
            "sigma":     None,
            "profile":   False,
        })
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
