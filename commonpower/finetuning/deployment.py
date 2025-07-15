import os

from scenarios import *
from utils import *

from commonpower.control.configs.algorithms import *
from commonpower.control.wrappers import *


def compute_average_results_over_seeds(results_dir, seeds):
    all_results = pd.concat(
        [pd.read_csv(results_dir + f'/{seed}/seed_results.csv', index_col=0).tail(1) for seed in seeds],
        ignore_index=True,
        axis=0,
    )

    mean_std_df = pd.DataFrame()

    for key in all_results.keys():
        mean_std_df.loc[0, key + "_mean"] = all_results[key].mean()
        mean_std_df.loc[0, key + "_std"] = all_results[key].std()

    mean_std_df.to_csv(results_dir + "/approach_results.csv")


def run_deployment(
    scenario: System,
    train_seed: int,
    horizon: timedelta,
    approach: Approach,
    rl_algorithm: RLAlgorithm,
    algo_config: AlgorithmBaseConfig,
    save_path: str,
    eval_periods: List[str],
    n_eval_steps: int,
    eval_seed: int,
    desired_return: Optional[np.ndarray] = None,
    desired_horizon: Optional[int] = None,
):
    alg_config = None
    if approach is not Approach.OptimalController:
        alg_config = MetaConfig(
            total_steps=1,
            seed=train_seed,
            policy_class=rl_algorithm.to_policy_class(),
            algorithm_config=algo_config,
        )
    model_dir = os.path.join(os.getcwd(), 'final_models', save_path, str(train_seed))
    # specify path for results
    results_dir = os.path.join(os.getcwd(), 'results', save_path, str(train_seed))
    os.makedirs(results_dir, exist_ok=True)

    wrappers = WrapperStack()
    if approach is not Approach.OptimalController:
        wrappers.add(SingleAgentWrapper)
        # set pre-trained policy in RL controller
        system_nodes = getattr(scenario, "nodes")
        rl_controller = getattr(system_nodes[0], "controller")
        setattr(rl_controller, "load_path", model_dir)

    for i, eval_period in enumerate(eval_periods):
        history = ModelHistory([scenario])
        deployer = DeploymentRunner(
            sys=scenario,
            horizon=horizon,
            history=history,
            seed=eval_seed,
            alg_config=alg_config,
            wrapper=wrappers.get_stack(),
            continuous_control=True,
        )
        if rl_algorithm == RLAlgorithm.PCN:
            if desired_return is not None:
                deployer.desired_return = desired_return
            if desired_horizon is not None:
                deployer.desired_horizon = desired_horizon
        datetime_format = "%d.%m.%Y"
        deployer.set_start_time(datetime.strptime(eval_period, datetime_format))
        deployer.run(n_steps=n_eval_steps)

        results_df = pd.DataFrame(deployer.deployment_log)
        results_df = results_df[['cum_cost', 'cum_penalty', 'cum_interventions']]
        results_df.to_csv(results_dir + "/seed_results.csv")
        print(results_df)

        final_cost = results_df['cum_cost'].iloc[-1]
        print(f"Cumulative cost: {final_cost} €")


if __name__ == "__main__":
    approach = Approach.WithProjectionSafeguard  # Approach.WithProjectionSafeguard
    penalty = Penalty.DDPenalty  # Penalty.NoPenalty
    scenario_constructor = Scenario.AddedEVScenario
    rl_algorithm = RLAlgorithm.PPO  # RLAlgorithm.PCN or RLAlgorithm.PPO or None if Approach.OptimalController

    scalarisation_code = "50-50"  # 50-50

    if approach is Approach.OptimalController:
        save_path = f'{scenario_constructor.name}/{approach.name}'
        # Set to None for Approach.OptimalController if not already for
        rl_algorithm = None
        algo_config = None
    else:
        save_path = f'{scenario_constructor.name}/{approach.name}/{penalty.name}/{rl_algorithm.name}'
        if not rl_algorithm.to_policy_class().is_morl:
            save_path += f'_{scalarisation_code}'

    # Set the evaluation time frame - one year starting on January 1st
    # (quite time intensive, could also change to evaluating over multiple weeks during the year but less accurate)
    eval_periods = [
        "02.01.2016"
    ]  # since we only have data from 2016 and our forecaster uses a lookback horizon of 24 hours
    n_eval_steps = 364 * 24  # one year
    eval_seed = 5
    device = 'cpu'  # 'auto'

    # for now we just use the mean return from the last step in the training...
    # step 74400 	 return [-33.08533    -0.6472926], ([0. 0.]) 	 loss 7.450E-02 	 horizons 744.0
    # for the future we might use values from eval/front table...
    # or choose a point that is slightly better than the achieved front
    pcn_desired_return = np.array([-0, -0])
    pcn_desired_horizon = 364 * 24  # one year ( same as n_eval_steps )

    stage = Stage.Deploy
    forecast_length = 6
    forecaster = PersistenceForecaster(
        frequency=timedelta(hours=1), horizon=timedelta(hours=forecast_length), look_back=timedelta(hours=24)
    )

    if approach is Approach.OptimalController:
        seeds = [1]
    else:
        seeds = [1, 2, 3, 4, 5]

    for seed in seeds:
        scenario, deployment_runner = create_scenario(
            stage=stage,
            approach=approach,
            penalty=penalty,
            scenario_constructor=scenario_constructor.value,
            forecast_length=forecast_length,
            forecaster=forecaster,
        )

        # extract relevant parameters
        horizon = getattr(deployment_runner, "horizon")

        # set up configuration for the PCN/PPO algorithm
        if approach == Approach.OptimalController:
            algo_config = None
        else:
            match rl_algorithm:
                case RLAlgorithm.PCN:
                    algo_config = PCN_Config(
                        device=device,
                        n_steps=n_eval_steps,
                        batch_size=24,
                    )
                case RLAlgorithm.PPO:
                    algo_config = PPO_Config(
                        device=device,
                        n_steps=96,
                        batch_size=24,
                        learning_rate=0.0008,
                        n_epochs=5,
                        policy_kwargs=dict(log_std_init=-2),
                    )
                case RLAlgorithm.CAPQL:
                    algo_config = CAPQL_Config(
                        device=device,
                        batch_size=24,
                    )
                case _:
                    raise NotImplementedError(f"Configuration for {rl_algorithm} is not defined.")

        run_deployment(
            scenario=scenario,
            train_seed=seed,
            horizon=horizon,
            approach=approach,
            rl_algorithm=rl_algorithm,
            algo_config=algo_config,
            save_path=save_path,
            eval_periods=eval_periods,
            n_eval_steps=n_eval_steps,
            eval_seed=eval_seed,
            desired_return=pcn_desired_return,
            desired_horizon=pcn_desired_horizon,
        )

    # average results over seeds:
    results_dir = os.getcwd() + f'/results/{save_path}'
    compute_average_results_over_seeds(results_dir, seeds)
