import torch

from commonpower.control.logging_utils.loggers import WandBLoggerPCN


class PCNAdapter:
    """
    Adapter that makes PCN compatible with SB3 callbacks and wraps PCN methods
    to inject callback calls at appropriate points.
    """

    def __init__(self, pcn_agent=None, callbacks=None):
        self.pcn_agent = pcn_agent
        self.callbacks = callbacks or []

        # Add missing attributes needed by SB3 callbacks
        self.num_timesteps = 0
        self.logger = WandBLoggerPCN()

        # Store original methods
        if pcn_agent is None:
            self._original_run_episode = None
            self._original_update = None
        else:
            self._original_run_episode = pcn_agent._run_episode
            self._original_update = pcn_agent.update

            # Apply wrappers
            self._wrap_methods()

    def get_env(self):
        """Return the training environment."""
        return self.pcn_agent.env

    def _wrap_methods(self):
        """Wrap PCN methods to inject callback calls."""

        # Create wrapped run_episode method
        def wrapped_run_episode(env, desired_return, desired_horizon, max_return, eval_mode=True):
            transitions = self._original_run_episode(env, desired_return, desired_horizon, max_return, eval_mode)

            # Update num_timesteps
            self.num_timesteps = self.pcn_agent.global_step

            # Call callbacks
            for callback in self.callbacks:
                callback.on_rollout_end()

            return transitions

        # Create wrapped update method
        def wrapped_update():
            result = self._original_update()

            # Update num_timesteps
            self.num_timesteps = self.pcn_agent.global_step

            # Call callbacks
            for callback in self.callbacks:
                callback.on_step()

            self.logger.dump(self.num_timesteps)

            return result

        # Apply wrappers
        self.pcn_agent._run_episode = wrapped_run_episode
        self.pcn_agent.update = wrapped_update

    def restore_original_methods(self):
        """Restore original PCN methods."""
        self.pcn_agent._run_episode = self._original_run_episode
        self.pcn_agent.update = self._original_update

    def init_callbacks(self):
        """Initialize callbacks with this adapter."""
        for callback in self.callbacks:
            callback.init_callback(self)

    def on_training_start(self):
        """Call on_training_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_start({}, {})

    def train(self, total_timesteps, eval_env, ref_point):
        if not self.pcn_agent:
            raise ValueError("PCN agent is not initialized.")
        self.pcn_agent.train(total_timesteps=total_timesteps, eval_env=eval_env, ref_point=ref_point)

    def on_training_end(self):
        """Call on_training_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_end()

    def save(self, path):
        """Save the PCN agent state."""
        torch.save(self.pcn_agent.model, path)

    def load(self, path):
        """Load the PCN agent state."""
        self.pcn_agent.model = torch.load(path)

    def predict(self, obs, deterministic=True):
        """
        Predict action based on observation.
        :param obs: Observation input.
        :param deterministic: Whether to use deterministic policy.
        :return: Predicted action.
        """
        return self.pcn_agent.eval(obs, deterministic=deterministic)
