import os

from scenarios import *
from utils import *

from commonpower.control.configs.algorithms import *
from commonpower.control.wrappers import *
from commonpower.utils.helpers import get_adjusted_cost


def compute_average_results_over_seeds(results_dir, seeds):
    for i, seed in enumerate(seeds):
        if i == 0:
            all_results = pd.read_csv(results_dir + f'/{seed}/seed_results.csv', index_col=0)
        else:
            seed_df = pd.read_csv(results_dir + f'/{seed}/seed_results.csv', index_col=0)
            all_results = pd.concat([all_results, seed_df], ignore_index=True, axis=0)

    mean_std_df = pd.DataFrame(
        columns=[
            "total_cum_reward",
            "cum_reward_mean",
            "cum_reward_std",
            "n_interventions_mean",
            "n_interventions_std",
        ],
        index=[0],
    )

    for key in all_results.keys():
        mean_std_df[key + "_mean"] = all_results[key].mean()
        mean_std_df[key + "_std"] = all_results[key].std()

    # we are also interested in the mean (over seeds) of the sum of returns over the five eval episodes
    mean_std_df["total_cum_reward"] = all_results["cum_reward"].sum() / len(seeds)

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
):
    alg_config = MetaConfig(
        total_steps=1,
        seed=train_seed,
        policy_class=rl_algorithm.to_policy_class(),
        algorithm_config=algo_config,
    )
    # model_dir = os.getcwd() + f'/models/{save_path}/{train_seed}'
    model_dir = os.getcwd() + f'/models/{save_path}/{train_seed}/PCN_model.pt'
    # specify path for results
    results_dir = os.getcwd() + f'/results/{save_path}/{train_seed}'
    os.makedirs(results_dir, exist_ok=True)
    # result data frames
    cum_cost_df = pd.DataFrame(
        columns=["cum_reward", "n_interventions"],
        index=[i for i in range(len(eval_periods))],
    )

    wrappers = WrapperStack()
    if not (approach is Approach.OptimalController):
        wrappers.add(SingleAgentWrapper)
        # set pre-trained policy in RL controller
        system_nodes = getattr(scenario, "nodes")
        rl_controller = getattr(system_nodes[0], "controller")
        setattr(rl_controller, "load_path", model_dir)

        if rl_algorithm == RLAlgorithm.PCN:
            # for now we just use the mean return from the last step in the training...
            # step 74400 	 return [-33.08533    -0.6472926], ([0. 0.]) 	 loss 7.450E-02 	 horizons 744.0
            # for the future we might use values from eval/front table...
            # or choose a point that is slightly better than the achieved front
            rl_controller.desired_return = np.array([-33.08533, -0.6472926])
            rl_controller.desired_horizon = 364 * 24  # one year ( same as n_eval_steps )

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
        datetime_format = "%d.%m.%Y"
        deployer.set_start_time(datetime.strptime(eval_period, datetime_format))
        deployer.run(n_steps=n_eval_steps)

        history.get_history_for_element(scenario.nodes[0], 'cost')
        cumulative_cost = get_adjusted_cost(history, scenario)
        print(f"Cumulative cost: {round(sum(cumulative_cost), 2)} €")
        cum_cost_df.iloc[i, 0] = round(sum(cumulative_cost), 2)
        if approach is Approach.OptimalController:
            num_interventions = 0
        else:
            controller_history = scenario.controllers["agent1"].deployment_history[0]["action_corrected"]
            num_interventions = sum([controller_history[i][1] for i in range(len(controller_history))])

        cum_cost_df.iloc[i, 1] = num_interventions

        cum_cost_df.to_csv(results_dir + "/seed_results.csv")


if __name__ == "__main__":
    approach = Approach.WithProjectionSafeguard  # Approach.OptimalController
    penalty = Penalty.DDPenalty  # Penalty.NoPenalty
    scenario_constructor = Scenario.AddedEVScenario
    rl_algorithm = RLAlgorithm.PCN  # RLAlgorithm.PPO
    save_path = f'{scenario_constructor}/{approach}/{penalty}'
    # Set the evaluation time frame - one year starting on January 1st
    # (quite time intensive, could also change to evaluating over multiple weeks during the year but less accurate)
    eval_periods = [
        "02.01.2016"
    ]  # since we only have data from 2016 and our forecaster uses a lookback horizon of 24 hours
    n_eval_steps = 364 * 24  # one year
    eval_seed = 5

    stage = Stage.Deploy
    forecast_length = 6
    forecaster = PersistenceForecaster(
        frequency=timedelta(hours=1), horizon=timedelta(hours=forecast_length), look_back=timedelta(hours=24)
    )

    if approach is Approach.OptimalController:
        seeds = [1]
    else:
        # seeds = [1, 2, 3, 4, 5]
        seeds = [1]

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
        if rl_algorithm == RLAlgorithm.PCN:
            algo_config = PCN_Config(
                device='auto',
                n_steps=n_eval_steps,
                batch_size=24,
            )
        elif rl_algorithm == RLAlgorithm.PPO:
            algo_config = PPO_Config(
                device="cpu",
                n_steps=96,
                batch_size=24,
                learning_rate=0.0008,
                n_epochs=5,
                policy_kwargs=dict(log_std_init=-2),
            )
        else:
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
        )

    # average results over seeds:
    results_dir = os.getcwd() + f'/results/{save_path}'
    compute_average_results_over_seeds(results_dir, seeds)
