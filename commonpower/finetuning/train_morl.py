import os

from morl_baselines.multi_policy.pcn.pcn import PCN
from scenarios import *

from commonpower.control.configs.algorithms import MORL_AlgorithmBaseConfig, MORL_MetaConfig, MORL_PCNConfig
from commonpower.control.logging_utils.loggers import *
from commonpower.control.runners import SingleAgentTrainerMORL
from commonpower.control.wrappers import *


def run_experiment(
    save_path: str,
    algo_config: MORL_AlgorithmBaseConfig,
    forecast_horizon: timedelta,
    episode_length: int,
    train_sys: System,
    scalarisation_fn: callable,
    scenario_constructor: Scenario,
    seed: int = 1,
    n_eps: int = 900,
    fixed_start: str = None,
    limited_date_range: List[datetime] = None,
):
    train_config = MORL_MetaConfig(
        total_steps=n_eps * algo_config.n_steps,
        seed=seed,
        algorithm=PCN,
        algorithm_config=algo_config,
    )

    # set up logger
    tb_log_dir = os.getcwd() + f'/tensorboard/{save_path}/{seed}'
    logger = WandBLogger(
        log_dir=tb_log_dir,
        entity_name="sgmorl",
        project_name="sgmorl",
        run_name=f"{save_path}_{seed}",
    )

    # specify the path where the model should be saved
    model_dir = os.getcwd() + f'/models/{save_path}/{seed}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    wrappers = WrapperStack().add(SingleAgentWrapper)

    ref_point = calculate_ref_point(scenario_constructor)

    # start training
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
    runner.run(fixed_start=fixed_start)


# TODO
def calculate_ref_point(scenario_constructor):
    if scenario_constructor == Scenario.AddedEVScenario:
        calculated_worst_case_cost = 1000.0
        action_space_diameter = 1000.0
        return np.array([-calculated_worst_case_cost, -action_space_diameter])
    else:
        raise NotImplementedError("Reference point calculation for this scenario is not implemented.")


if __name__ == "__main__":
    seeds = [42]
    n_eps = 100
    approach = Approach.WithProjectionSafeguard
    penalty = Penalty.DDPenalty
    scenario_constructor = Scenario.AddedEVScenario

    stage = Stage.Train
    forecast_length = 6
    forecaster = PersistenceForecaster(
        frequency=timedelta(hours=1), horizon=timedelta(hours=forecast_length), look_back=timedelta(hours=24)
    )

    for seed in seeds:
        train_sys, deployment_runner = create_scenario(
            stage=stage,
            approach=approach,
            penalty=penalty,
            scenario_constructor=scenario_constructor.value,
            forecast_length=forecast_length,
            forecaster=forecaster,
        )

        # Updated save path to with MORL experiment. Naming convention from original code
        save_path = f'{scenario_constructor.name}/{approach.name}/{penalty.name}/MORL_PCN'

        date_format = "%Y-%m-%d %H:%M:00"
        start = datetime.strptime("2016-07-01 00:00:00", date_format)
        end = datetime.strptime("2016-07-31 23:00:00", date_format)

        horizon = getattr(deployment_runner, "horizon")
        episode_length = 24 * 31

        pcn_config = MORL_PCNConfig(
            device='auto',
            n_steps=episode_length,
            batch_size=episode_length,
        )

        run_experiment(
            save_path=save_path,
            algo_config=pcn_config,
            seed=seed,
            n_eps=n_eps,
            forecast_horizon=horizon,
            episode_length=episode_length,
            train_sys=train_sys,
            scenario_constructor=scenario_constructor,
            scalarisation_fn=None,
            fixed_start=start,
            limited_date_range=[start, end],
        )
