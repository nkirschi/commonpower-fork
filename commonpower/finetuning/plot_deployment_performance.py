import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def load_and_process(
    results_dir: str, seeds: list[int], max_steps: int | None = None
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Loads and processes experimental results from multiple seeds, computes statistics, and returns summary arrays.
    Args:
        results_dir (str): Path to the directory containing results for each seed.
        seeds (list[int]): List of integer seed values corresponding to subdirectories to load.
        max_steps (int | None, optional): Maximum number of steps to pad/truncate results to.
    Returns:
        tuple[
            np.ndarray | None,  # mean_cost: Mean cumulative cost across seeds at each timestep.
            np.ndarray | None,  # std_cost: Standard deviation of cumulative cost across seeds at each timestep.
            np.ndarray | None,  # mean_interventions: Mean number of interventions across seeds at each timestep.
            np.ndarray | None,  # std_interventions: Standard deviation of interventions across seeds at each timestep.
            np.ndarray | None   # timesteps: Array of timestep indices (length = max_steps).
        ]
        If no data is found, all returned values are None.
    Notes:
        - Each seed is expected to have a 'seed_results.csv' file with columns 'cum_cost' and 'n_interventions'.
        - Shorter runs are padded with their last value to match the length of the longest run (or max_steps).
        - Prints status messages about loading and missing files.
    """

    all_costs: list[np.ndarray] = []
    all_interventions: list[np.ndarray] = []
    current_max_steps = 0

    print(f"\nLoading data from: {results_dir}")
    for seed in seeds:
        file_path = os.path.join(results_dir, str(seed), 'seed_results.csv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                all_costs.append(df['cum_cost'].values)
                all_interventions.append(df['n_interventions'].values)
                if len(df) > current_max_steps:
                    current_max_steps = len(df)
                print(f"  - Loaded seed {seed} ({len(df)} steps)")
            except Exception as e:
                print(f"Could not read or process file for seed {seed}: {e}")
        else:
            print(f"Warning: Could not find results file for seed {seed} at {file_path}")

    if not all_costs:
        print(f"No data found in {results_dir}. Skipping.")
        return None, None, None, None, None

    # number of steps for padding
    if max_steps is None:
        max_steps = current_max_steps

    # pad shorter runs with their last value
    costs_padded = [np.pad(c, (0, max_steps - len(c)), 'edge') for c in all_costs]
    interventions_padded = [np.pad(i, (0, max_steps - len(i)), 'edge') for i in all_interventions]

    # compute statistics
    costs_matrix = np.vstack(costs_padded)
    interventions_matrix = np.vstack(interventions_padded)

    mean_cost = np.mean(costs_matrix, axis=0)
    std_cost = np.std(costs_matrix, axis=0)
    mean_interventions = np.mean(interventions_matrix, axis=0)
    std_interventions = np.std(interventions_matrix, axis=0)

    timesteps = np.arange(max_steps)

    return mean_cost, std_cost, mean_interventions, std_interventions, timesteps


def plot(deployment_run: list[dict]) -> None:
    """
    Generates and saves a comparative plot of deployment performance for multiple deployment runs.
    For each experiment, this function loads results from multiple random seeds,
    computes the mean and standard deviation of cumulative cost and cumulative interventions over time,
    and plots these metrics with shaded error bands.
    The resulting figure contains two subplots:
        1. Economic Performance: Cumulative Cost (€) over time.
        2. Safety Performance: Cumulative Interventions over time.
    The x-axis is aligned across all deployment runs based on the maximum number of time steps found in the data.
    The plot is saved as 'results/deployment_plot.png' and also displayed.
    Args:
        deployment_run (list of dict): A list where each dict represents an experiment and must contain:
            - 'name' (str): Name of the experiment (used in the legend).
            - 'path' (str): Path to the experiment's results directory.
            - 'seeds' (list of int): List of random seeds to aggregate results from.
            - 'color' (str): Color for plotting this experiment.
            - 'linestyle' (str, optional): Line style for plotting (default is solid line).
    Returns:
        None
    """

    print("Generating comparison plot...")

    plt.style.use('default')

    plt.rcParams.update(
        {
            'font.size': 18,  # Base font size
            'axes.titlesize': 20,  # Subplot titles
            'axes.labelsize': 18,  # Axis labels
            'xtick.labelsize': 16,  # X tick labels
            'ytick.labelsize': 16,  # Y tick labels
            'legend.fontsize': 16,  # Legend
            'figure.titlesize': 24,  # Figure title
        }
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), sharex=True)
    fig.suptitle('PCN/PPO Deployment Performance', fontsize=22, y=0.96)

    # find max of steps of all experiments to align x axis
    max_steps = 0
    for exp in deployment_run:
        for seed in exp['seeds']:
            file_path = os.path.join(exp['path'], str(seed), 'seed_results.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if len(df) > max_steps:
                    max_steps = len(df)

    for exp in deployment_run:
        mean_cost, std_cost, mean_interventions, std_interventions, timesteps = load_and_process(
            exp['path'], exp['seeds'], max_steps
        )

        if timesteps is not None:
            linestyle = exp.get('linestyle', '-')

            # Plot Cumulative Cost
            ax1.plot(timesteps, mean_cost, label=f"{exp['name']}", color=exp['color'], linestyle=linestyle)
            ax1.fill_between(timesteps, mean_cost - std_cost, mean_cost + std_cost, color=exp['color'], alpha=0.15)

            # Plot Cumulative Interventions
            ax2.plot(timesteps, mean_interventions, label=f"{exp['name']}", color=exp['color'], linestyle=linestyle)
            ax2.fill_between(
                timesteps,
                mean_interventions - std_interventions,
                mean_interventions + std_interventions,
                color=exp['color'],
                alpha=0.15,
            )

    ax1.set_ylabel('Cumulative Cost (€)')
    ax1.set_title('Economic Performance')
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2.set_xlabel('Time Steps (Hours)')
    ax2.set_ylabel('Cumulative Interventions')
    ax2.set_title('Safety Performance')
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Make x-axis ticks more readable (1000 steps = 1000 hours)
    tick_spacing = 1000
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax1.tick_params(width=2, length=6)
    ax2.tick_params(width=2, length=6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_save_path = os.path.join('results', 'deployment_plot.png')
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {plot_save_path}")

    plt.show()


if __name__ == '__main__':

    # Define common path components for path construction
    scenario_constructor = "AddedEVScenario"
    approach_rl = "WithProjectionSafeguard"
    penalty = "DDPenalty"

    experiments_to_plot = [
        {
            "name": "Optimal Controller (Baseline)",
            "path": os.path.join('results', scenario_constructor, "OptimalController"),
            "seeds": [1],
            "color": "black",
            "linestyle": "--",
        },
        {
            "name": "PCN",
            "path": os.path.join('results', scenario_constructor, approach_rl, penalty, "PCN"),
            "seeds": [1, 2, 3, 4, 5],
            "color": "green",
            "linestyle": "-",
        },
        {
            "name": "PPO (50/50)",
            "path": os.path.join('results', scenario_constructor, approach_rl, penalty, "PPO_50-50"),
            "seeds": [1, 2, 3, 4, 5],
            "color": "blue",
            "linestyle": "-",
        },
        {
            "name": "PPO (80/20)",
            "path": os.path.join('results', scenario_constructor, approach_rl, penalty, "PPO_80-20"),
            "seeds": [1, 2, 3, 4, 5],
            "color": "purple",
            "linestyle": "-",
        },
    ]

    plot(experiments_to_plot)
