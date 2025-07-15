from gymnasium import Env
from stable_baselines3 import SAC

from commonpower.control.policies.sb3_policy import SB3Policy


class SACPolicy(SB3Policy):
    """
    Facade for SB3's SAC policy.
    """

    def __init__(self, env: Env, seed: int, callback=None, **library_specific_kwargs):
        super().__init__(env, seed, callback)
        self.library_specific_policy = SAC(env=env, seed=seed, **library_specific_kwargs)
