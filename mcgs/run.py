"""Monte Carlo Tree Search for Dynamical Symbolic Regression."""

from __future__ import annotations

import os
import random
import time

import matplotlib
import numpy as np
import pandas as pd
from data_generation import generate_noisy_sir_data, get_incidence, ode_solver
from utils.data_saving import (
    clean_data_saving,
    estimation_saving,
    text_saving,
    time_saving,
)
from utils.mcgs import MCGS
from utils.trajectory_figure import plot_trajectories
from utils.visualise_graph import export_mcgs_to_graphviz

matplotlib.use("Agg")

# Experiment name
experiment_name = "latent_states_noise_incidence"

# MCTS parameters
grammar = {
    "M": [
        "M -> s",  # Terminal rules
        "M -> i",
        "M -> r",
        "M -> C",
        "M -> M + M",  # Binary rules
        "M -> M - M",
        "M -> M * M",
    ],
}

t_max = 6  # Max number of production rules/ nodes
eta = 0.9999  # Parsimony factor
gamma_discount = 0.9  # Discount rule
epsilon = 0.01  # Epsilon for graph search update
top_N = 5  # Number of best equations to track
episodes = 100
steps = 2 * (t_max + t_max + 1)  # Max num steps in a path with force completion
num_realisations = 10
rollouts_per_leaf = 2

plot_dot = False  # dot file of graph
data_seed = 1234
search_seed = 1234

# SIR parameters
num_fluxes = 2  # SIR fluxes: (S->I) & (I->R)
# Define the stoichiometry of the system.
# This maps fluxes [J1, J2] to states [s, i, r]
# s_dot = -1*J1 + 0*J2
# i_dot = +1*J1 - 1*J2
# r_dot = +0*J1 + 1*J2
system_structure = {"s": [-1, 0], "i": [1, -1], "r": [0, 1]}
# Initial state: (action lists, stacks, rule counts)
initial_state = (
    [[] for _ in range(num_fluxes)],
    [["M"] for _ in range(num_fluxes)],  # Start with "M"
    [0 for _ in range(num_fluxes)],
)

# Priors on actions in fluxes
flux_priors = {
    0: {
        "M -> s": 1,
        "M -> i": 1,
        "M -> r": 0,  # Recovered compartment not part of infection (first flux)
        "M -> C": 1,
        "M -> M + M": 1,
        "M -> M - M": 1,
        "M -> M * M": 1,
    },
    1: {
        "M -> s": 0,  # Susceptible compartment not part of recovery (second flux)
        "M -> i": 1,
        "M -> r": 1,
        "M -> C": 1,
        "M -> M + M": 1,
        "M -> M - M": 1,
        "M -> M * M": 1,
    },
}

# Data parameters
days = 60
times = np.arange(0, days, 1)
beta, gamma = 0.3, 0.1

# Base initial conditions
init_S_base, init_I_base, init_R_base = 100, 2, 0
N_base = init_S_base + init_I_base + init_R_base
initial_conditions_base = [
    init_S_base / N_base,
    init_I_base / N_base,
    init_R_base / N_base,
]

# Noise conditions
conditions_df = pd.read_csv("../conditions.csv")
conditions_df.columns = conditions_df.columns.str.strip()

for index, row in conditions_df.iterrows():
    experiment_start_time = time.time()

    # Create directory to save results
    save_dir = f"results/{experiment_name}/experiment_{index}"

    results_file = os.path.join(save_dir, "full_results.csv")
    if os.path.exists(results_file):
        print(
            f"Results already exist at {results_file}, skipping experiment {index}..."
        )
        continue

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Extract conditions
    noise_level = row["noise_level"]

    # Set seed
    random.seed(data_seed)
    np.random.seed(data_seed)

    # Generate multiple data realisations
    data_X = []
    data_plot = []
    all_initial_conditions = []

    for _ in range(num_realisations):
        # Set initial conditions
        init_S, init_I, init_R = (
            100 + random.randint(-10, 10),
            2 + random.randint(-1, 5),
            0,
        )
        N = init_S + init_I + init_R
        initial_conditions = [init_S / N, init_I / N, init_R / N]
        all_initial_conditions.append(initial_conditions)

        # Get SIR trajectories for conditions
        if noise_level > 0:
            # Non-noisy solution for incidence calculation
            clean_sol = ode_solver(
                par=initial_conditions,
                time=times,
                beta=beta,
                gamma=gamma,
                proportion=True,
            )
            # Noisy solution
            sol = generate_noisy_sir_data(
                par=initial_conditions,
                time=times,
                beta=beta,
                gamma=gamma,
                proportion=True,
                noise_std=noise_level / N,
            )
            # Extract prevalence and incidence
            clean_susceptibles = clean_sol[:, 0]
            infection_prevalence = sol[:, 1]
            infection_incidence = get_incidence(clean_susceptibles, N=N)
        else:
            sol = ode_solver(
                par=initial_conditions,
                time=times,
                beta=beta,
                gamma=gamma,
                proportion=True,
            )
            # Extract prevalence and incidence
            susceptibles, infection_prevalence = sol[:, 0], sol[:, 1]
            infection_incidence = -np.diff(susceptibles)  # Clean incidence (no noise)

        # Model data
        data_X.append(
            {
                "time": times,
                "prevalence": infection_prevalence,
                "incidence": infection_incidence,
                "initial_conditions": initial_conditions,
            }
        )

        # Plotting data
        data_plot.append(
            {
                "time": times,
                "s": sol[:, 0],
                "i": sol[:, 1],
                "r": sol[:, 2],
            }
        )

    # Set seed
    random.seed(search_seed)
    np.random.seed(search_seed)

    # MCTS instance
    search_graph = MCGS(
        data_X=data_X,
        system_structure=system_structure,
        num_fluxes=num_fluxes,
        grammar=grammar,
        flux_priors=flux_priors,
        initial_state=initial_state,
        t_max=t_max,
        eta=eta,
        gamma=gamma_discount,
        epsilon=epsilon,
        top_N=top_N,
        rollouts_per_leaf=rollouts_per_leaf,
    )

    # Run the search
    top_results = search_graph.run_search(
        episodes=episodes,
        steps=steps,
        print_epi=1,
    )

    # Save all results
    search_graph.export_results_csv(f"{save_dir}/full_results.csv")
    search_graph.export_graph_bounds(f"{save_dir}/nodes_and_bounds.csv")

    # Save text file with final top equations
    text_saving(top_N, system_structure, save_dir, top_results)

    # Save all realisations used for training
    np.save(f"{save_dir}/all_data_X.npy", data_X)
    np.save(f"{save_dir}/all_initial_conditions.npy", all_initial_conditions)
    print(f"Saved all {num_realisations} realisations to {save_dir}/")

    # Save clean data
    clean_data_saving(
        beta,
        gamma,
        initial_conditions_base,
        times,
        save_dir,
    )

    # Save estimated data (from top-ranked system)
    if not top_results:
        print("Search failed. Skipping estimation.")
    else:
        # Get the top-ranked numerical flux strings
        # We use [0][2] to get the 'final_flux_strs' from the best result
        top_str_fluxes = top_results[0][2]

        try:
            estimation_saving(
                top_str_fluxes,
                initial_conditions_base,
                times,
                system_structure,
                save_dir,
            )

        except Exception as e:
            print(f"Error during estimation/saving: {e}")

    # Save experiment timing
    experiment_end_time = time.time()
    time_saving(
        experiment_end_time=experiment_end_time,
        experiment_start_time=experiment_start_time,
        save_dir=save_dir,
    )

    # Save plots
    plot_trajectories(
        real_sol=data_plot, file_dir=save_dir, expr_idx=index, figsize=(10, 5)
    )

    if plot_dot:
        export_mcgs_to_graphviz(search_graph, filename=f"{save_dir}/search_graph.dot")
