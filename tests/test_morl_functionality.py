import unittest
import os
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from commonpower.core import System
from commonpower.models.components import Load, ESS
from commonpower.models.buses import Bus, TradingBus
from commonpower.models.powerflow import PowerBalanceModel
from commonpower.data_forecasting.data_sources import CSVDataSource, ConstantDataSource
from commonpower.data_forecasting.base import DataProvider
from commonpower.data_forecasting.forecasters import LookBackForecaster
from commonpower.modeling.param_initialization import RangeInitializer

from commonpower.control.environments import ControlEnv
from commonpower.control.runners import SingleAgentTrainer
from commonpower.control.controllers import RLController
from commonpower.control.policies.ppo_policy import PPOPolicy
from commonpower.control.wrappers import SingleAgentWrapper
from commonpower.control.safety_layer.safety_layers import ActionProjectionSafetyLayer
from commonpower.control.safety_layer.penalties import DistanceDependingPenalty
from commonpower.control.configs.algorithms import MetaConfig, PPO_Config

class TestMORLFunctionality(unittest.TestCase):
    """Unit tests for Multi-Objective Reinforcement Learning functionality"""

    def setUp(self):
        self.artifacts_path = Path("./tests/artifacts/").resolve()
        os.makedirs(self.artifacts_path, exist_ok=True)

        frequency = timedelta(hours=1)
        horizon = timedelta(hours=24)
        data_path = (Path(__file__).parent / "data" / "1-LV-rural2--1-sw").resolve()

        load_p_ds = CSVDataSource(
            data_path / "LoadProfile.csv",
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "H0-A_pload": "p"},
            auto_drop=True,
        )
        load_q_ds = ConstantDataSource({"q": 0.0}, date_range=load_p_ds.get_date_range())

        price_ds = CSVDataSource(
            data_path / "LoadProfile.csv",
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "G1-B_pload": "psib", "G1-C_pload": "psis"},
            auto_drop=True,
        )

        load_p_dp = DataProvider(load_p_ds, LookBackForecaster(frequency=frequency, horizon=horizon))
        load_q_dp = DataProvider(load_q_ds, LookBackForecaster(frequency=frequency, horizon=horizon))
        price_dp = DataProvider(price_ds, LookBackForecaster(frequency=frequency, horizon=horizon))

        # System components
        bus = Bus("TestBus")
        load = Load("TestLoad").add_data_provider(load_p_dp).add_data_provider(load_q_dp)
        ess = ESS(
            "TestESS",
            {
                "cap": 10.0,  # Capacity in kWh
                "p": [-5, 5],  # Active power bounds
                "q": [0, 0],  # Reactive power bounds
                "soc": [0.2 * 10, 0.8 * 10],  # State of charge bounds (20% to 80% of capacity)
                "soc_init": RangeInitializer(4.0, 6.0),  # Initial state of charge
                "rho": 0.01,  # Self-discharge rate
                "etac": 0.95,  # Charging efficiency
                "etad": 0.95,  # Discharging efficiency
                "etas": 0.99,  # Standstill efficiency
            },
        )
        trading_bus = TradingBus("Trading").add_data_provider(price_dp)

        # System assembly
        self.sys = System(power_flow_model=PowerBalanceModel()).add_node(bus).add_node(trading_bus)
        bus.add_node(load).add_node(ess)

        # Create RL controller with safety layer and penalty function
        penalty = DistanceDependingPenalty(penalty_factor=10.0)
        self.agent = RLController(name="morl_agent", safety_layer=ActionProjectionSafetyLayer(penalty=penalty))
        self.agent.add_entity(ess)
        self.agent.add_entity(trading_bus)
        self.agent.add_system(self.sys)

        self.sys.initialize()
        self.sys.reset(at_time=datetime(2016, 11, 27))

    def tearDown(self):
        """Clean up test artifacts after test"""
        shutil.rmtree(self.artifacts_path)

    def test_control_env_returns_vector_reward_when_scalarizer_is_none(self):
        """Tests that the ControlEnv returns correctly shaped vector reward when scalarisation_fn is None"""

        # Create environment
        env = ControlEnv(system=self.sys, scalarisation_fn=None)
        env.reset()
        dummy_action = env.action_space.sample()

        # Take step in environment
        _, rewards, _, _, _ = env.step(dummy_action)
        agent_reward = rewards[self.agent.name]

        # Check reward format
        self.assertIsInstance(agent_reward, np.ndarray, "Reward should be a NumPy array for MORL.")
        self.assertEqual(agent_reward.shape, (2,), "Vector reward should have shape (2,) for [cost, penalty].")

    def test_control_env_uses_default_scalarization(self):
        """Tests that ControlEnv defaults to returning first element of reward vector (i.e negative cost) if no scalarizer is provided"""

        env = ControlEnv(system=self.sys)
        env.reset()
        dummy_action = env.action_space.sample()

        original_step = env.sys.step

        def mock_sys_step(*args, **kwargs):
            obs, _, _ = original_step(*args, **kwargs)
            mock_costs = {self.agent.name: 50.0}
            mock_info = {"safety_penalties": {self.agent.name: 5.0}}    
            return obs, mock_costs, mock_info

        env.sys.step = mock_sys_step

        _, rewards, _, _, _ = env.step(dummy_action)
        scalar_reward = rewards[self.agent.name]

        print(scalar_reward)

        # Check if correct calculation i.e the reward is a scalar equal to negative cost
        self.assertIsInstance(scalar_reward, (float, np.floating), msg="Default reward should be a scalar.")
        self.assertAlmostEqual(scalar_reward, -55.0, msg="Default reward should be the negative of the cost.")

    def test_control_env_applies_custom_linear_scalarization(self):
        """Tests that ControlEnv correctly applies custom linear scalarization function to the vectorized reward."""

        weights = np.array([0.7, 0.3])
        scalarizer = lambda r: np.dot(r, weights)
        env = ControlEnv(system=self.sys, scalarisation_fn=scalarizer)
        env.reset()
        dummy_action = env.action_space.sample()

        original_step = env.sys.step

        def mock_sys_step(*args, **kwargs):
            obs, _, info = original_step(*args, **kwargs)
            mock_costs = {self.agent.name: 100.0}
            mock_info = {"safety_penalties": {self.agent.name: 10.0}}   
            return obs, mock_costs, mock_info

        env.sys.step = mock_sys_step

        _, rewards, _, _, _ = env.step(dummy_action)
        scalar_reward = rewards[self.agent.name]

        expected_reward = np.dot(np.array([-100.0, -10.0]), weights)

        self.assertIsInstance(scalar_reward, (float, np.floating), "Scalarized reward should be a scalar.")
        self.assertAlmostEqual(scalar_reward, expected_reward, places=5)

    def test_trainer_integration_with_scalarization(self):
        """Integration test to ensure SingleAgentTrainer can successfully execute training run with a given scalarisation_fn"""

        model_path = self.artifacts_path / "morl_integration_model"
        weights = np.array([0.8, 0.2])
        scalarisation_fn = lambda r: np.dot(r, weights)

        alg_config = MetaConfig(total_steps=2, seed=42, policy_class=PPOPolicy, algorithm_config=PPO_Config(n_steps=2))

        trainer = SingleAgentTrainer(
            sys=self.sys,
            alg_config=alg_config,
            wrapper=SingleAgentWrapper,
            save_path=str(model_path),
            seed=42,
            scalarisation_fn=scalarisation_fn,
        )

        try:
            trainer.run()
        except Exception as e:
            self.fail(f"SingleAgentTrainer failed to run with scalarization: {e}")

        self.assertTrue(os.path.exists(os.path.join(model_path, 'model.zip')), "Trainer should save a model file.")


if __name__ == "__main__":
    unittest.main()
