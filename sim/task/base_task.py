import numpy as np


class BaseTask:
    """
    Defines the RL task interface: observations, rewards, and termination.

    A Task is injected into BaseEnv and called each step/reset.
    It receives the env instance so it can read simulation state
    (e.g. env.state_0.particle_q, env.state_0.joint_q, env.sim_time).
    """

    def __init__(self):
        pass
    
    def initialize_resources(self, env):
        """Called once at environment initialization. Override to set up task-specific resources."""
        pass

    def observation(self, env) -> dict[str, np.ndarray]:
        """Return the current observation as a dict of numpy arrays."""
        raise NotImplementedError("observation method must be implemented by subclasses.")

    def reward(self, env) -> float:
        """Return the scalar reward for the current state."""
        raise NotImplementedError("reward method must be implemented by subclasses.")

    def terminated(self, env) -> bool:
        """Return True when the episode reaches a terminal success/failure state."""
        raise NotImplementedError("terminated method must be implemented by subclasses.")

    def truncated(self, env) -> bool:
        """Return True when the episode is cut short (e.g. time limit). Override as needed."""
        return False

    def reset(self, env) -> None:
        """Called at the start of each episode. Override to reset task-specific state."""
        pass
