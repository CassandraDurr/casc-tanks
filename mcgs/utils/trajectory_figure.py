"""Plotting function to plot and save the real vs expected trajectories."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_trajectories(
    real_sol: list[dict[str, np.ndarray]],
    file_dir: str,
    expr_idx: int,
    figsize=(10, 5),
):
    """Plot trajectory for real vs estimated trajectories using the top result.

    Args:
        real_sol (list[dict]): List of real SIR realisations (each dict has 'time', 's', 'i', 'r').
        file_dir (str): File directory of results.
        expr_idx (int): Experiment id for title.
        figsize (tuple, optional): Figure size. Defaults to (10, 5).
    """
    # Load estimated and clean data
    estimated_data_df = pd.read_csv(f"{file_dir}/estimated_data.csv")
    clean_data_df = pd.read_csv(f"{file_dir}/clean_data.csv")
    all_data_X = np.load(f"{file_dir}/all_data_X.npy", allow_pickle=True)

    # Calculate clean incidence/prevalence
    clean_incidence = -np.diff(clean_data_df["true_S"])  # Shape (T-1,)
    clean_prevalence = clean_data_df["true_I"]  # Shape (T,)

    # Calculate estimated incidence/prevalence
    estimated_incidence = -np.diff(estimated_data_df["estimated_S"])  # Shape (T-1,)
    estimated_prevalence = estimated_data_df["estimated_I"]  # Shape (T,)

    # Times
    time = estimated_data_df["time"]

    # --- State trajectories plot ---

    fig, ax = plt.subplots(figsize=figsize)

    # --- Plot all noisy realisations (lighter) ---
    for real in real_sol:
        ax.plot(real["time"], real["s"], color="blue", alpha=0.1)
        ax.plot(real["time"], real["i"], color="red", alpha=0.1)
        ax.plot(real["time"], real["r"], color="green", alpha=0.1)

    # --- Plot Estimates (dashed) ---
    ax.plot(
        time,
        estimated_data_df["estimated_S"],
        label="S (Estimated)",
        color="blue",
        linestyle="dashed",
    )
    ax.plot(
        time,
        estimated_data_df["estimated_I"],
        label="I (Estimated)",
        color="red",
        linestyle="dashed",
    )
    ax.plot(
        time,
        estimated_data_df["estimated_R"],
        label="R (Estimated)",
        color="green",
        linestyle="dashed",
    )

    # --- Plot Clean Trajectory (dotted) ---
    ax.plot(
        time,
        clean_data_df["true_S"],
        label="S (Clean)",
        color="blue",
        linestyle="dotted",
    )
    ax.plot(
        time,
        clean_data_df["true_I"],
        label="I (Clean)",
        color="red",
        linestyle="dotted",
    )
    ax.plot(
        time,
        clean_data_df["true_R"],
        label="R (Clean)",
        color="green",
        linestyle="dotted",
    )

    # Plot settings
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Population Proportion")
    ax.set_title(f"Actual vs Estimated Trajectories for Experiment {expr_idx}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{file_dir}/trajectory_plot.png")
    plt.close(fig)

    # --- Incidence plot ---
    fig, ax = plt.subplots(figsize=figsize)

    for data_X in all_data_X:
        ax.plot(
            data_X["time"][1:],
            data_X["incidence"],
            color="darkorange",
            alpha=0.1,
        )
    ax.plot(
        time[1:],
        clean_incidence,
        label="Clean",
        color="darkorange",
        linestyle="dotted",
    )
    ax.plot(
        time[1:],
        estimated_incidence,
        label="Estimated",
        color="darkorange",
        linestyle="dashed",
    )

    # Plot settings
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Incidence")
    ax.set_title("Incidence: Data vs Clean vs Estimated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{file_dir}/incidence_plot.png")
    plt.close(fig)

    # --- Prevalence plot ---
    fig, ax = plt.subplots(figsize=figsize)

    for data_X in all_data_X:
        ax.plot(
            data_X["time"],
            data_X["prevalence"],
            color="purple",
            alpha=0.1,
        )
    ax.plot(
        time,
        clean_prevalence,
        label="Clean",
        color="purple",
        linestyle="dotted",
    )
    ax.plot(
        time,
        estimated_prevalence,
        label="Estimated",
        color="purple",
        linestyle="dashed",
    )

    # Plot settings
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Prevalence")
    ax.set_title("Prevalence: Data vs Clean vs Estimated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{file_dir}/prevalence_plot.png")
    plt.close(fig)
