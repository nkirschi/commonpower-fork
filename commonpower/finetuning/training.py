import os

import numpy as np
from scenarios import *

from commonpower.control.configs.algorithms import AlgorithmBaseConfig, MetaConfig, PCN_Config, PPO_Config
from commonpower.control.logging_utils.loggers import *
from commonpower.control.runners import SingleAgentTrainerMORL, SingleAgentTrainerSB3
from commonpower.control.wrappers import *
from commonpower.finetuning.utils import RLAlgorithm


def run_experiment(
    run_id: str,
    algo_config: AlgorithmBaseConfig,
    forecast_horizon: timedelta,
    episode_length: int,
    train_sys: System,
    scalarisation_fn: callable,
    scenario_constructor: Scenario,
    seed: int,
    n_episodes: int,
    fixed_start: str,
    limited_date_range: List[datetime],
):
    train_config = MetaConfig(
        total_steps=n_episodes * algo_config.n_steps,
        seed=seed,
        algorithm=rl_algorithm.to_algorithm_class(),
        algorithm_config=algo_config,
    )

    # set up logger
    tb_log_dir = os.getcwd() + f'/tensorboard/{run_id}/{seed}'
    logger = WandBLogger(
        log_dir=tb_log_dir,
        entity_name="sgmorl",
        project_name="sgmorl",
        run_name=f"{run_id}/{seed}",
    )

    # specify the path where the model should be saved
    model_dir = os.getcwd() + f'/models/{run_id}/{seed}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    wrappers = WrapperStack().add(SingleAgentWrapper)

    ref_point = calculate_ref_point(scenario_constructor)

    # start training
    if rl_algorithm.is_morl():
        runner = SingleAgentTrainerMORL(
            sys=train_sys,
            wrapper=wrappers.get_stack(),
            alg_config=train_config,
            ref_point=ref_point,
            horizon=forecast_horizon,
            episode_length=episode_length,
            logger=logger,
            save_path=model_dir,
            seed=seed,
            limited_date_range=limited_date_range,
            scalarisation_fn=scalarisation_fn,
        )
    elif rl_algorithm == RLAlgorithm.PPO:
        runner = SingleAgentTrainerSB3(
            sys=train_sys,
            wrapper=wrappers.get_stack(),
            alg_config=train_config,
            horizon=forecast_horizon,
            episode_length=episode_length,
            logger=logger,
            save_path=model_dir,
            seed=seed,
            limited_date_range=limited_date_range,
        )

    runner.run(fixed_start=fixed_start)


# TODO
def calculate_ref_point(scenario_constructor):
    if scenario_constructor == Scenario.AddedEVScenario:
        calculated_worst_case_cost = 1000.0
        action_space_diameter = 1000.0
        return np.array([-calculated_worst_case_cost, -action_space_diameter])
    else:
        raise NotImplementedError("Reference point calculation for this scenario is not implemented.")


def linear_scalariser(preference_vector: np.ndarray):
    return lambda reward_vector: np.dot(reward_vector, preference_vector)


if __name__ == "__main__":
    # BEGIN CONFIGURATION #

    # scaling hyperparameters
    seeds = [1]  # [1, 2, 3, 4, 5]
    n_episodes = 100  # 1000
    forecast_length = 6
    start_time = "2016-07-01 00:00:00"
    end_time = "2016-07-31 23:00:00"

    # modelling hyperparameters
    scenario_constructor = Scenario.AddedEVScenario
    approach = Approach.WithProjectionSafeguard
    penalty = Penalty.DDPenalty
    rl_algorithm = RLAlgorithm.PCN  # RLAlgorithm.PPO
    preference_vector = np.array([0.8, 0.2])  # only applied for single-objective RL algorithms

    # END CONFIGURATION #

    scalarisation_fn = None if rl_algorithm.is_morl() else linear_scalariser(preference_vector)
    stage = Stage.Train
    forecaster = PersistenceForecaster(
        frequency=timedelta(hours=1), horizon=timedelta(hours=forecast_length), look_back=timedelta(hours=24)
    )

    for seed in seeds:
        train_sys, deployment_runner = create_scenario(
            stage=stage,
            scenario_constructor=scenario_constructor.value,
            approach=approach,
            penalty=penalty,
            rl_algorithm=rl_algorithm,
            forecast_length=forecast_length,
            forecaster=forecaster,
        )

        # ID for this training run. Naming convention from original code
        run_id = f'{scenario_constructor.name}/{approach.name}/{penalty.name}/{rl_algorithm.name}'

        date_format = "%Y-%m-%d %H:%M:00"
        start = datetime.strptime(start_time, date_format)
        end = datetime.strptime(end_time, date_format)

        horizon = getattr(deployment_runner, "horizon")
        episode_length = 1 + (end - start).total_seconds() // 3600

        if rl_algorithm == RLAlgorithm.PCN:
            algo_config = PCN_Config(
                device='auto',
                n_steps=episode_length,
                batch_size=episode_length,
            )
        elif rl_algorithm == RLAlgorithm.PPO:
            algo_config = PPO_Config(
                device='auto',
                n_steps=episode_length,
                batch_size=episode_length,
                learning_rate=0.0008,
                n_epochs=5,
                policy_kwargs=dict(log_std_init=0),
            )

        run_experiment(
            run_id=run_id,
            algo_config=algo_config,
            seed=seed,
            n_episodes=n_episodes,
            forecast_horizon=horizon,
            episode_length=episode_length,
            train_sys=train_sys,
            scenario_constructor=scenario_constructor,
            scalarisation_fn=scalarisation_fn,
            fixed_start=start,
            limited_date_range=[start, end],
        )
