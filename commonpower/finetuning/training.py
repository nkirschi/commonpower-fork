import os

import numpy as np
from scenarios import *

from commonpower.control.configs.algorithms import (
    AlgorithmBaseConfig,
    CAPQL_Config,
    MetaConfig,
    PCN_Config,
    PPO_Config,
    SAC_Config,
)
from commonpower.control.logging_utils.loggers import *
from commonpower.control.runners import SingleAgentTrainer
from commonpower.control.wrappers import *
from commonpower.finetuning.utils import RLAlgorithm


def run_experiment(
    run_id: str,
    algo_config: AlgorithmBaseConfig,
    forecast_horizon: timedelta,
    episode_length: int,
    train_sys: System,
    rl_algorithm: RLAlgorithm,
    seed: int,
    n_episodes: int,
    fixed_start: str,
    limited_date_range: List[datetime],
    scalarisation_fn: Optional[callable],
    ref_point: np.ndarray | None,
):
    total_steps = n_episodes * episode_length

    train_config = MetaConfig(
        total_steps=total_steps,
        seed=seed,
        policy_class=rl_algorithm.to_policy_class(),
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

    # start training
    runner = SingleAgentTrainer(
        sys=train_sys,
        wrapper=wrappers.get_stack(),
        alg_config=train_config,
        horizon=forecast_horizon,
        episode_length=episode_length,
        logger=logger,
        save_path=model_dir,
        seed=seed,
        limited_date_range=limited_date_range,
        scalarisation_fn=scalarisation_fn,
        ref_point=ref_point,
    )
    runner.run(fixed_start=fixed_start)


def calculate_ref_point(scenario_constructor):
    if scenario_constructor == Scenario.AddedEVScenario:
        calculated_worst_case_cost = (
            846.79  # from PCN-Bounds-AddedEVScenario.ipynb
            # with HP calculated_worst_case_cost = 985.99
        )
        action_space_diameter = (
            2000.00  # from PCN-Diameter-AddedEVScenario.ipynb
            # with HP action_space_diameter = 2000.01
        )
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
    device = 'cpu'  # 'auto'

    # modelling hyperparameters
    scenario_constructor = Scenario.AddedEVScenario
    approach = Approach.WithProjectionSafeguard
    penalty = Penalty.DDPenalty
    rl_algorithm = RLAlgorithm.PPO  # one of [PPO, SAC, PCN, CAPQL]
    preference_vector = np.array([0.5, 0.5])  # only applied for single-objective RL algorithms

    # END CONFIGURATION #

    scalarisation_fn = linear_scalariser(preference_vector) if not rl_algorithm.to_policy_class().is_morl else None
    ref_point = calculate_ref_point(scenario_constructor) if rl_algorithm.to_policy_class().is_morl else None
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
            forecast_length=forecast_length,
            forecaster=forecaster,
        )

        # ID for this training run. Naming convention from original code
        run_id = f'{scenario_constructor.name}/{approach.name}/{penalty.name}/{rl_algorithm.name}'
        if not rl_algorithm.to_policy_class().is_morl:
            scalarisation_code = f'{int(100 * preference_vector[0])}-{int(100 * preference_vector[1])}'
            run_id += f'_{scalarisation_code}'

        date_format = "%Y-%m-%d %H:%M:00"
        start = datetime.strptime(start_time, date_format)
        end = datetime.strptime(end_time, date_format)

        horizon = getattr(deployment_runner, "horizon")
        episode_length = 1 + (end - start).total_seconds() // 3600  # entire period in hours

        match rl_algorithm:
            case RLAlgorithm.PPO:
                algo_config = PPO_Config(
                    device=device,
                    n_steps=episode_length,
                    batch_size=episode_length,
                    learning_rate=0.008,
                    n_epochs=5,
                )
            case RLAlgorithm.SAC:
                algo_config = SAC_Config(
                    device=device,
                    n_steps=episode_length,
                    batch_size=episode_length,
                    learning_rate=0.008,
                    train_freq=episode_length,
                )
            case RLAlgorithm.PCN:
                algo_config = PCN_Config(device=device, batch_size=episode_length, num_er_episodes=n_episodes // 10)
            case RLAlgorithm.CAPQL:
                algo_config = CAPQL_Config(device=device, batch_size=episode_length)

        run_experiment(
            run_id=run_id,
            algo_config=algo_config,
            seed=seed,
            n_episodes=n_episodes,
            forecast_horizon=horizon,
            episode_length=episode_length,
            train_sys=train_sys,
            rl_algorithm=rl_algorithm,
            fixed_start=start,
            limited_date_range=[start, end],
            scalarisation_fn=scalarisation_fn,
            ref_point=ref_point,
        )
