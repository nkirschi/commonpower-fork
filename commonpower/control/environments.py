"""
Base API (based on gymnasium API) between controlled system and RL training algorithms.
"""

from collections import OrderedDict, deque
from copy import copy, deepcopy
from datetime import datetime
from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec

from commonpower.modeling.base import ControllableModelEntity
from commonpower.modeling.history import ModelHistory
from commonpower.utils.cp_exceptions import ControllerError


def default_scalarisation_fn(reward_vector: np.ndarray) -> float:
    cost = reward_vector[0]
    penalty = reward_vector[1]
    return cost + penalty


class ControlEnv(gym.Env):
    def __init__(
        self,
        system: ControllableModelEntity,
        continuous_control: bool = False,
        episode_length: int = 24,
        fixed_start: datetime = None,
        normalize_action_space: bool = True,
        history: ModelHistory = None,
        scalarisation_fn: Optional[callable] = default_scalarisation_fn,
    ):
        """
        Class that provides the interface between our power system and any reinforcement learning algorithm. Based on
        the OpenAI Gym API (which is now maintained as 'gymnasium', see https://gymnasium.farama.org/). Manages all
        RL controllers within the power system.

        Args:
            system (ControllableModelEntity): power system including Pyomo model with all constraints
            continuous_control (bool): if true, the environment is never resetted
            episode_length (int): how many environment interaction steps to complete before resetting the environment
            fixed_start (datetime): if None, we will train from multiple random start times.
                Otherwise, we will always train from the same start time.
            normalize_action_space (bool): whether to normalize the action space to [-1,1]
            history (ModelHistory): logger


        Returns:
            ControlEnv

        """
        from commonpower.control.controllers import RLBaseController

        self.controllers = system.get_controllers(ctrl_types=[RLBaseController])
        self.sys = system
        self.current_action = None
        self.train_history = {}
        self.episode_history = {agent_id: deque(maxlen=100) for agent_id in self.controllers.keys()}
        self.normalize_actions = normalize_action_space
        self.system_history = history
        self.scalarisation_fn = scalarisation_fn

        # training or deployment?
        self.train = True

        # ToDo: shared observation space?
        self.observation_space = self._get_observation_space()
        if self.normalize_actions:
            self.action_space, self.original_action_space = self._get_normalized_action_space()
        else:
            self.action_space = self._get_action_space()

        # whether to just continuously step through the year or not
        self.continuous_control = continuous_control
        # step counter
        self.completed_steps = 0
        self.episode_length = episode_length
        # whether or not to train on a fixed day
        self.fixed_start = fixed_start

    def set_mode(self, mode: str):
        # set train flag of RL runners to False
        for ctrl in self.controllers.values():
            ctrl.set_mode(mode)
        if mode == "train":
            self.train = True
        else:
            self.train = False

    def step(self, action: Union[OrderedDict, None]) -> Tuple[dict, dict, bool, bool, dict]:
        """
        Advance the environment (in our case, the power system) by one step in time by applying control actions to
        discrete-time dynamics and updating data sources. Handled within the System class. The actions of the RL agent
        are selected within the RL training algorithm and are passed on to the power system using a callback. After the
        system update, a reward is computed which indicates how good the action selected by the algorithm was in the
        current state. This reward is passed to the training algorithm to gradually improve the policies of the RL
        agents.

        Args:
            action (OrderedDict): actions of RL agents (here as a dictionary of agent IDs and their respective actions)

        Returns:
            Tuple: tuple containing:
                - observations of all RL agents (dict), here as a dictionary of agent IDs and their respective \
                observations
                - rewards of all RL agents (dict)
                - whether the episode has terminated (bool). We assume that all agents terminate an episode at the \
                same time, as we have a centralized time management. Always false for continuous control
                - same as above but the gymnasium API makes a difference between terminated and truncated, which can \
                be useful for other environments but is not needed in our case
                - additional information (dict)

        """
        if action is not None:
            # expects a list of actions or a single action (numpy array) as an input
            if len(self.controllers) == 1 and isinstance(action, list):
                raise ControllerError(self.controllers[0], "One agent but multiple actions")
            if len(self.controllers) > 1 and isinstance(action, float):
                raise ControllerError(self.controllers[0], "Multiple agents but only one action")

            # store action
            if self.normalize_actions:
                action = self._denormalize_action(action)

        self.current_action = action

        obs, costs, info = self.sys.step(rl_action_callback=self.rl_action_callback, history=self.system_history)

        self.completed_steps += 1
        terminated, truncated = self._is_done()
        # extract only the info for the RL controllers
        # the obs_handler of each controller will
        # a) take care of removing unwanted forecasts and
        # b) stack past observations if specified
        obs = {
            agent: self.controllers[agent].obs_handler.get_adjusted_obs(agent_obs)
            for agent, agent_obs in obs.items()
            if agent in self.controllers.keys()
        }
        # rewards are vectors of negative cost and safety penalty
        rewards = {
            agent: np.array([-costs[agent], -info["safety_penalties"][agent]]) for agent in self.controllers.keys()
        }
        # update history with reward penalty
        for agent_id, agent in self.controllers.items():
            agent.update_history({"reward_without_penalty": rewards[agent_id][0]})
        # get train history at end of episode:
        if terminated or truncated:
            self.train_history = {agent_id: copy(agent.history) for agent_id, agent in self.controllers.items()}
            for agent_id in self.controllers.keys():
                self.episode_history[agent_id].append(
                    {
                        "mean_penalty": np.mean([t[1] for t in self.train_history[agent_id]["safety_penalty"]]),
                        "rew_without_penalty": np.sum(
                            [t[1] for t in self.train_history[agent_id]["reward_without_penalty"]]
                        ),
                        "n_corrections": np.sum([t[1] for t in self.train_history[agent_id]["action_corrected"]]),
                    }
                )

        # scalarise reward vectors if a scalarisation function is provided
        if self.scalarisation_fn is not None:
            rewards = {agent: self.scalarisation_fn(reward) for agent, reward in rewards.items()}

        return obs, rewards, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """
        Reset the power system to the beginning of an episode (which spans 24 hours).

        Args:
            seed:  The seed that is used to initialize the environment’s PRNG (np_random). If the environment does not
                already have a PRNG and seed=None (the default option) is passed, a seed will be chosen from some \
                source of entropy (e.g. timestamp or /dev/urandom). However, if the environment already has a PRNG and \
                seed=None is passed, the PRNG will not be reset. If you pass an integer, the PRNG will be reset even \
                if it already exists. Usually, you want to pass an integer right after the environment has been \
                initialized and then never again. This should be taken care of by calling super().reset(seed=seed) in \
                the first line of this function. (https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)
            options: not needed

        Returns:
            Tuple: tuple containing
                - observations of all RL agents after reset (dict)
                - additional information for observations (dict)

        """
        # call reset() of gymnasium Env class to ensure that we only reset ONCE right after initialization
        # and then never again
        super().reset(seed=seed)

        self.completed_steps = 0
        reset_time = self.sys.sample_start_date(self.fixed_start)
        # reset system history
        if self.system_history:
            self.system_history.reset()
        if self.train:
            self.sys.reset(reset_time)
        else:
            # environment is already reset once in deployment runner. We only have to reset it once more here
            # in case we use pre-trained RL policies (load them from a directory)
            if len(self.controllers) > 0:
                # check if we are working with pre-trained policies
                pretrained_controllers = [getattr(ctrl, 'load_path') is not None for ctrl in self.controllers.values()]
                only_pretrained_controllers = all(pretrained_controllers)
                no_pretrained_controllers = not any(pretrained_controllers)
                if not (only_pretrained_controllers or no_pretrained_controllers):
                    raise ValueError("The controllers all have to be either pre-trained or not. Mixing not possible.")
                # we have to reset the system again because when loading the policies with single-agent RL,
                # the env is seeded...
                if only_pretrained_controllers:
                    self.sys.reset(reset_time)

        obs, obs_info = self.sys.observe()
        # extract only the info for the RL controllers
        # the obs_handler of each controller will
        # a) take care of removing unwanted forecasts and
        # b) stack past observations if specified
        obs = {
            agent: self.controllers[agent].obs_handler.get_adjusted_obs(agent_obs)
            for agent, agent_obs in obs.items()
            if agent in self.controllers.keys()
        }
        return obs, obs_info

    def rl_action_callback(self, ctrl_id: str):
        """
        Passes current action selected by training algorithm to the compute_control_input() function of the
        BaseController class.

        Args:
            ctrl_id(str): ID of the controller for which to retrieve the action

        Returns:
            dict: actions for all controlled entities assigned to this controller

        """
        return self.current_action[ctrl_id]

    def _is_done(self) -> Tuple[bool, bool]:
        """
        Determines whether the environment has to be reset. "Done" normally means that a goal has been reached,
        which is never the case in power systems control. It can also mean that a safety violation occured
        (which also should not happen in our case, but could be implemented in case we want to let a system fail.)
        "Truncated" means that we have reach the end of a pre-defined time limit and therefore want to reset.
        We currently assume that all agents terminate an episode at the same time, as we have a centralized time
        management

        Returns:
            tuple(bool, bool): Done, truncated
        """
        done = False
        if self.continuous_control:
            truncated = False
        else:
            truncated = self.completed_steps == self.episode_length

        return done, truncated

    def _get_observation_space(self) -> gym.spaces.Dict:
        """
        Retrieve observation space from list of RL controllers and their observation masks.

        Returns:
            gym.spaces.Dict: dictionary of agent IDs and their observation spaces

        """
        # ToDo: What happens in case we don't have a box space?
        obs_spaces = OrderedDict()
        for ctrl_id, ctrl in self.controllers.items():
            nodes = ctrl.get_nodes()
            ctrl_obs_space = ctrl.obs_handler.get_observation_space(nodes)
            obs_spaces[ctrl_id] = ctrl_obs_space

        obs_spaces = gym.spaces.Dict(obs_spaces)
        return obs_spaces

    def _get_action_space(self) -> gym.spaces.Dict:
        """
        Retrieve action space from RL controllers

        Returns:
            gym.spaces.Dict: dictionary of agent IDs and their action spaces

        """
        # ToDo: What happens in case we don't have a box space?
        act_spaces = OrderedDict()
        for ctrl_id, ctrl in self.controllers.items():
            act_spaces[ctrl_id] = ctrl.get_input_space(normalize=False)
        act_spaces = gym.spaces.Dict(act_spaces)
        return act_spaces

    def _get_normalized_action_space(self) -> gym.spaces.Dict:
        """
        Normalize all actions to [-1,1]

        Returns:
            gym.spaces.Dict: dictionary of agent IDs and their action spaces

        """
        # ToDo: What happens in case we don't have a box space?
        act_spaces = OrderedDict()
        norm_act_spaces = OrderedDict()
        for ctrl_id, ctrl in self.controllers.items():
            act_spaces[ctrl_id] = ctrl.get_input_space(normalize=False)
            norm_act_spaces[ctrl_id] = ctrl.get_input_space(normalize=True)
        norm_act_spaces = gym.spaces.Dict(norm_act_spaces)
        act_spaces = gym.spaces.Dict(act_spaces)
        return norm_act_spaces, act_spaces

    def _denormalize_action(self, action: OrderedDict) -> OrderedDict:
        """
        Denormalize action to original input space.

        Args:
            action (OrderedDict): normalized action

        Returns:

        """
        scaled_action = deepcopy(action)
        for ctrl_id, ctrl_action in action.items():
            for node_id, node_action in ctrl_action.items():
                for el_id, el_action in node_action.items():
                    action_low = self.original_action_space[ctrl_id][node_id][el_id].low
                    action_high = self.original_action_space[ctrl_id][node_id][el_id].high

                    new_action = (action[ctrl_id][node_id][el_id] - (-1 * np.ones((len(action_high,))))) / 2 * np.ones(
                        (
                            len(
                                action_high,
                            )
                        )
                    ) * (action_high - action_low) + action_low
                    scaled_action[ctrl_id][node_id][el_id] = new_action

        return scaled_action


class MORLEnv(gym.Env):
    def __init__(
        self,
        system,
        continuous_control=False,
        episode_length=24,
        fixed_start=None,
        normalize_action_space=True,
        history=None,
        scalarisation_fn=None,  # Default to None to handle vectors
        reward_dim=2,  # Specify reward dimensionality
        wrapper=None,
    ):
        if wrapper:
            env = wrapper(
                ControlEnv(
                    system=system,
                    continuous_control=continuous_control,
                    episode_length=episode_length,
                    fixed_start=fixed_start,
                    normalize_action_space=normalize_action_space,
                    history=history,
                    scalarisation_fn=None,  # Handle scalarization yourself
                )
            )

        # Create the base environment
        self.envs = [env]

        self.envs[0].episode_history = deque(maxlen=100)
        # Make sure there's only one agent
        if len(self.envs[0].unwrapped.controllers) > 1:
            raise ValueError("MyMORLEnvironment cannot handle more than 1 agent")

        self.ctrl_id = list(self.envs[0].unwrapped.controllers.keys())[0]

        # Add MORL-specific attributes
        self.reward_dim = reward_dim
        self.custom_scalarisation_fn = scalarisation_fn

        # Define reward space for MORL algorithms
        self.reward_space = gym.spaces.Box(
            low=np.array([-float('inf')] * reward_dim), high=np.array([float('inf')] * reward_dim), dtype=np.float32
        )

        self.observation_space = self.envs[0].observation_space

        self.action_space = self.envs[0].action_space

        self.spec = EnvSpec(
            id="MyMORLEnvironment-v0",
            entry_point=None,
            max_episode_steps=episode_length,
        )

    def _flatten_observation(self, obs_dict):
        """Flatten a nested observation dictionary into a 1D array"""
        flat_obs = np.array([])
        ctrl_obs = obs_dict[self.ctrl_id]

        for el_id, el_obs in recursive_items(ctrl_obs):
            flat_obs = np.concatenate((flat_obs, el_obs))

        return flat_obs

    def _unflatten_action(self, action):
        """Convert a flat action array back to the nested dictionary format"""
        # Create the dictionary structure
        action_dict = {self.ctrl_id: {}}
        ctrl_act_space = self.envs[0].action_space[self.ctrl_id]

        # Fill in the action values
        idx = 0
        for n_id, n_act_space in ctrl_act_space.items():
            action_dict[self.ctrl_id][n_id] = {}
            for el_id, el_space in n_act_space.items():
                size = np.prod(el_space.shape)
                action_dict[self.ctrl_id][n_id][el_id] = action[idx : idx + size].reshape(el_space.shape)
                idx += size

        return action_dict

    def reset(self, *, seed=None, options=None):
        return self.envs[0].reset(seed=seed, options=options)

    def step(self, action):
        return self.envs[0].step(action)

    def set_mode(self, mode):
        return self.envs[0].set_mode(mode)


def recursive_items(dictionary):
    """
    Recursive extraction of all values in a nested dictionary or gym.spaces.Dict
    """
    for key, value in dictionary.items():
        if isinstance(value, (gym.spaces.Dict, dict)):
            yield from recursive_items(value)
        else:
            yield (key, value)
