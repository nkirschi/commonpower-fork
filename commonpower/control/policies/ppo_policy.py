from gymnasium import Env
from stable_baselines3 import PPO

from commonpower.control.policies.sb3_policy import SB3Policy


class PPOPolicy(SB3Policy):
    """
    Facade for SB3's PPO policy.
    """

    def __init__(self, env: Env, seed: int, callback=None, **library_specific_kwargs):
        super().__init__(env, seed, callback)
        self.library_specific_policy = PPO(env=env, seed=seed, **library_specific_kwargs)
