import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def load_and_process(
    results_dir: str, seeds: list[int], max_steps: int | None = None, normalize_interventions: bool = False
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:

    all_costs: list[np.ndarray] = []
    all_interventions: list[np.ndarray] = []
    current_max_steps = 0

    print(f"\nLoading data from: {results_dir}")
    for seed in seeds:
        file_path = os.path.join(results_dir, str(seed), "seed_results.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                costs = df["cum_cost"].values
                interventions = df["cum_interventions"].values

                if normalize_interventions:
                    final_interventions = interventions[-1]
                    if final_interventions > 0:
                        interventions = interventions / final_interventions

                all_costs.append(costs)
                all_interventions.append(interventions)

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

    if max_steps is None:
        max_steps = current_max_steps

    costs_padded = [np.pad(c, (0, max_steps - len(c)), "edge") for c in all_costs]
    interventions_padded = [np.pad(i, (0, max_steps - len(i)), "edge") for i in all_interventions]

    costs_matrix = np.vstack(costs_padded)
    interventions_matrix = np.vstack(interventions_padded)

    mean_cost = np.mean(costs_matrix, axis=0)
    std_cost = np.std(costs_matrix, axis=0)
    mean_interventions = np.mean(interventions_matrix, axis=0)
    std_interventions = np.std(interventions_matrix, axis=0)
    timesteps = np.arange(max_steps)

    return mean_cost, std_cost, mean_interventions, std_interventions, timesteps


def plot(deployment_runs: list[dict]) -> None:

    print("Generating final comparison plot...")

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.titlesize": 24,
        }
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 14), sharex=True)
    fig.suptitle("Deployment Performance: RL Controllers vs. Optimal Baseline", fontsize=22, y=0.96)

    max_steps = 0
    for exp in deployment_runs:
        for seed in exp["seeds"]:
            file_path = os.path.join(exp["path"], str(seed), "seed_results.csv")
            if os.path.exists(file_path):
                df_len = len(pd.read_csv(file_path))
                if df_len > max_steps:
                    max_steps = df_len

    results_data_cost = {}
    results_data_interventions = {}
    baseline_data_cost = None

    for exp in deployment_runs:
        mean_cost, std_cost, _, _, _ = load_and_process(
            exp["path"], exp["seeds"], max_steps, normalize_interventions=False
        )
        results_data_cost[exp["name"]] = (mean_cost, std_cost)
        if exp.get("is_baseline", False):
            baseline_data_cost = (mean_cost, std_cost)

        _, _, mean_interventions_norm, std_interventions_norm, timesteps = load_and_process(
            exp["path"], exp["seeds"], max_steps, normalize_interventions=True
        )
        results_data_interventions[exp["name"]] = (mean_interventions_norm, std_interventions_norm, timesteps)

    if baseline_data_cost is None:
        print("Error: No baseline experiment found. Mark one with 'is_baseline': True.")
        return

    baseline_mean_cost, baseline_std_cost = baseline_data_cost

    for exp in deployment_runs:
        mean_cost, std_cost = results_data_cost[exp["name"]]
        mean_interventions, std_interventions, timesteps = results_data_interventions[exp["name"]]
        linestyle = exp.get("linestyle", "-")

        if timesteps is not None:
            if exp.get("is_baseline", False):
                ax1.axhline(0, color=exp["color"], linestyle=linestyle, label=f"{exp['name']}")
            else:
                diff_mean_cost = mean_cost - baseline_mean_cost
                diff_std_cost = np.sqrt(std_cost**2 + baseline_std_cost**2)
                ax1.plot(timesteps, diff_mean_cost, label=f"{exp['name']}", color=exp["color"], linestyle=linestyle)
                ax1.fill_between(
                    timesteps,
                    diff_mean_cost - diff_std_cost,
                    diff_mean_cost + diff_std_cost,
                    color=exp["color"],
                    alpha=0.15,
                )

            ax2.plot(timesteps, mean_interventions, label=f"{exp['name']}", color=exp["color"], linestyle=linestyle)
            ax2.fill_between(
                timesteps,
                mean_interventions - std_interventions,
                mean_interventions + std_interventions,
                color=exp["color"],
                alpha=0.15,
            )

    ax1.set_ylabel("Cost Difference to Baseline (€)", labelpad=15)
    ax1.set_title("Relative Economic Performance")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax2.set_xlabel("Time Steps (Hours)")
    ax2.set_ylabel("Normalized Cumulative Interventions", labelpad=15)
    ax2.set_title("Safety Performance")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.set_ylim(bottom=0)

    tick_spacing = 1000
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax1.tick_params(width=2, length=6)
    ax2.tick_params(width=2, length=6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_save_path = os.path.join("results", "deployment_cdf_plot.png")
    os.makedirs("results", exist_ok=True)
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {plot_save_path}")

    plt.show()


if __name__ == "__main__":
    scenario_constructor = "AddedEVScenario"
    approach_rl = "WithProjectionSafeguard"
    penalty = "DDPenalty"

    experiments_to_plot = [
        {
            "name": "Optimal Controller (Baseline)",
            "path": os.path.join("final_results", scenario_constructor, "OptimalController"),
            "seeds": [1],
            "color": "black",
            "linestyle": "--",
            "is_baseline": True,
        },
        {
            "name": "PCN",
            "path": os.path.join("final_results", scenario_constructor, approach_rl, penalty, "PCN"),
            "seeds": [1, 2, 3, 4, 5],
            "color": "green",
            "linestyle": "-",
        },
        {
            "name": "PPO (50/50)",
            "path": os.path.join("final_results", scenario_constructor, approach_rl, penalty, "PPO_50-50"),
            "seeds": [1, 2, 3, 4, 5],
            "color": "blue",
            "linestyle": "-",
        },
        {
            "name": "PPO (80/20)",
            "path": os.path.join("final_results", scenario_constructor, approach_rl, penalty, "PPO_80-20"),
            "seeds": [1, 2, 3, 4, 5],
            "color": "purple",
            "linestyle": "-",
        },
    ]

    plot(deployment_runs=experiments_to_plot)
