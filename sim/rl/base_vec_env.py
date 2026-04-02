
from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from rl_games.common.ivecenv import IVecEnv
from rl_games.common import vecenv as rl_vecenv, env_configurations
import sim
from ..env import BaseEnv
from ..task import BaseTask


class BaseVecEnv(IVecEnv):
    """rl_games IVecEnv wrapper."""

    @classmethod
    def register(cls, vecenv_type: str, env_name: str, cfg: OmegaConf) -> None:

        def _make(config_name=None, num_actors=None, **kwargs):
            num_envs_override = kwargs.pop("num_envs", None)
            if num_envs_override is not None:
                OmegaConf.set_struct(cfg, False)
                cfg.env.num_envs = int(num_envs_override)
            return cls(cfg=cfg)

        rl_vecenv.register(
            vecenv_type,
            lambda config_name, num_actors, **kw: _make(config_name, num_actors, **kw)
        )
        env_configurations.register(env_name, {
            "vecenv_type": vecenv_type,
            "env_creator": lambda **kw: _make(**kw),  # required by player.create_env()
        })

    def __init__(self, cfg: OmegaConf):
        self._env: BaseEnv = eval(cfg.env.name)(cfg=cfg)
        self._env.initialize_resources(no_renderer=True)

        self._task: BaseTask = eval(cfg.task.name)(cfg=cfg)
        self._task.initialize_resources(self._env)

    def step(self, actions):
        joint_targets = self._task.scale_actions(np.asarray(actions, dtype=np.float64), self._env)

        # Action repeat: hold the same target for action_repeat physics steps,
        # accumulating rewards. Early-exit per-env on termination is not done here
        # since all envs share a single sim step; we just sum rewards across repeats.
        rew = np.zeros(self._env.num_envs, dtype=np.float32)
        for _ in range(self._task.action_repeat):
            self._env.step({'target': joint_targets})
            rew += self._task.reward(self._env)

        # Compute obs first so we can detect physics explosions before anything
        # else reads from the (potentially NaN) simulation state.
        obs_dict = self._task.observation(self._env)
        obs = self._task.flatten_obs(obs_dict)
        nan_mask = ~np.all(np.isfinite(obs), axis=1)

        # terminated/truncated: suppress NaN envs so their broken state doesn't
        # produce wrong done signals, and so truncated()'s _env_time side-effect
        # doesn't tick for envs that are about to be reset anyway.
        terminated = self._task.terminated(self._env)
        truncated = self._task.truncated(self._env)
        terminated[nan_mask] = False
        truncated[nan_mask] = False

        # Per-component rewards for monitoring. Zero NaN envs before updating
        # accumulators so they never see non-finite values.
        components = self._task.reward_components(self._env)
        if nan_mask.any():
            for k in components:
                components[k][nan_mask] = 0.0
            rew[nan_mask] = 0.0
        self._task.update_episode_metrics(components)

        # NaN envs are treated as episode-ending crashes (not terminated, not
        # timed-out) so they get episode info logged and are immediately reset.
        is_done = terminated | truncated | nan_mask
        info = {
            "time_outs": truncated,
            "nan_resets": float(nan_mask.sum()),
            **{f'components/{k}': v.mean() for k, v in components.items()}
        }

        if is_done.any():
            # Build episode-level metrics for all done envs (normal + NaN crashes).
            info["episode"] = self._task.build_episode_info(is_done, terminated)

            # Reset all done envs and patch their observations.
            self._env.reset(mask=is_done)
            self._task.reset(self._env, mask=is_done)
            reset_obs = self._task.observation(self._env)
            for k in obs_dict:
                obs_dict[k][is_done] = reset_obs[k][is_done]

            self._task.reset_episode_metrics(mask=is_done)

        obs = self._task.flatten_obs(obs_dict)

        return obs, rew.astype(np.float32), is_done, info

    def reset(self):
        """Reset all envs and return flat obs (num_envs, 19)."""
        self._env.reset()
        self._task.reset(self._env, mask=None)
        obs_dict = self._task.observation(self._env)
        return self._task.flatten_obs(obs_dict)

    def get_env_info(self) -> dict:
        return {
            "action_space": self._task.action_space,
            "observation_space": self._task.observation_space,
            "state_space": None,
            "use_global_observations": False,
            "agents": 1,
            "value_size": 1,
        }

    @property
    def observation_space(self):
        return self._task.observation_space

    @property
    def action_space(self):
        return self._task.action_space
