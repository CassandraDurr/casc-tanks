"""Helper functions to save the results of the MCTS run."""

import os

import numpy as np
import pandas as pd
import sympy
from data_generation import ode_solver
from scipy.integrate import odeint

# State variables using sympy
s, i, r = sympy.symbols("s i r")


def text_saving(
    top_N: int, system_structure: dict[str, list[int]], save_dir: str, top_results: list
):
    """Save top equations as a text file."""
    # Save text file with final top equations
    txt_path = os.path.join(save_dir, f"top_{top_N}_results.txt")
    with open(txt_path, "w") as f:
        f.write("--- Top System Results ---\n")

        if not top_results:
            f.write("Search failed to find a valid system.\n")

        for rank, (reward, sympy_flux_exprs, str_fluxes) in enumerate(top_results):
            f.write(f"\n--- Rank {rank + 1} System ---\n")
            f.write(f"  Reward = {reward:.8f}\n")

            # Write the found fluxes
            f.write("  Fluxes (with symbolic constants):\n")
            for j, flux_expr in enumerate(sympy_flux_exprs):
                f.write(f"    J{j+1} = {flux_expr}\n")

            f.write("  Fluxes (with estimated constants):\n")
            for j, flux_str in enumerate(str_fluxes):
                f.write(f"    J{j+1} = {flux_str}\n")

            # Reconstruct and write the full system equations
            f.write("  Full System Equations (estimated):\n")

            # Convert string fluxes to sympy expressions to build the system
            try:
                numerical_flux_exprs = [sympy.sympify(fs) for fs in str_fluxes]

                for state_key in system_structure.keys():  # 's', 'i', 'r'
                    state_eq = sympy.Float(0.0)
                    for i_flux, flux_coeff in enumerate(system_structure[state_key]):
                        if flux_coeff != 0:
                            state_eq += flux_coeff * numerical_flux_exprs[i_flux]
                    f.write(f"    {state_key}_dot = {state_eq}\n")
            except sympy.SympifyError:
                f.write("    Error: Could not parse flux strings to build system.\n")


def clean_data_saving(
    beta: float,
    gamma: float,
    initial_conditions_base: list[float],
    times: np.ndarray,
    save_dir: str,
):
    """Save clean data."""
    clean_sol = ode_solver(
        par=initial_conditions_base,
        time=times,
        beta=beta,
        gamma=gamma,
        proportion=True,
    )
    clean_S, clean_I, clean_R = clean_sol[:, 0], clean_sol[:, 1], clean_sol[:, 2]

    # Calculate theoretical (clean) derivatives
    clean_S_dot = -beta * clean_S * clean_I
    clean_I_dot = beta * clean_S * clean_I - gamma * clean_I
    clean_R_dot = gamma * clean_I

    # Save to DataFrame
    clean_data = {
        "time": times,
        "true_S": clean_S,
        "true_I": clean_I,
        "true_R": clean_R,
        "true_derivative_S": clean_S_dot,
        "true_derivative_I": clean_I_dot,
        "true_derivative_R": clean_R_dot,
    }
    clean_df = pd.DataFrame(clean_data)
    clean_df.to_csv(f"{save_dir}/clean_data.csv", index=False)
    print(f"Saved representative clean data to {save_dir}/clean_data.csv")


def estimation_saving(
    top_str_fluxes,
    initial_conditions_base: list[float],
    times: np.ndarray,
    system_structure: dict[str, list[int]],
    save_dir: str,
):
    """Save estimated data."""
    # Convert the numerical strings back into SymPy expressions
    top_flux_exprs = [sympy.sympify(fs) for fs in top_str_fluxes]

    # Build the full system equations from the top fluxes
    system_eqs = {}
    for state_key in system_structure.keys():  # 's', 'i', 'r'
        state_eq = sympy.Float(0.0)
        for i_flux, flux_coeff in enumerate(system_structure[state_key]):
            if flux_coeff != 0:
                state_eq += flux_coeff * top_flux_exprs[i_flux]
        system_eqs[state_key] = state_eq

    # Convert the numerical SymPy equations to functions
    s_dot_func = sympy.lambdify([s, i, r], system_eqs["s"], "numpy")
    i_dot_func = sympy.lambdify([s, i, r], system_eqs["i"], "numpy")
    r_dot_func = sympy.lambdify([s, i, r], system_eqs["r"], "numpy")

    # Integrate to get the estimated states
    # Define the system of ODEs for the solver
    def estimated_system(y, t):
        """Estimated derivatives from top MCTS results."""
        s_val, i_val, r_val = y
        ds_dt = s_dot_func(s_val, i_val, r_val)  # noqa: B023
        di_dt = i_dot_func(s_val, i_val, r_val)  # noqa: B023
        dr_dt = r_dot_func(s_val, i_val, r_val)  # noqa: B023
        return [ds_dt, di_dt, dr_dt]

    # Simulate the system using the found equations
    estimated_sol = odeint(
        estimated_system, initial_conditions_base, times
    )  # Use clean initial conditions
    estimated_S, estimated_I, estimated_R = (
        estimated_sol[:, 0],
        estimated_sol[:, 1],
        estimated_sol[:, 2],
    )

    # Get the estimated derivative using the estimated state trajectory
    estimated_S_derivative = s_dot_func(estimated_S, estimated_I, estimated_R)
    estimated_I_derivative = i_dot_func(estimated_S, estimated_I, estimated_R)
    estimated_R_derivative = r_dot_func(estimated_S, estimated_I, estimated_R)

    estimated_data = {
        "time": times,
        "estimated_S": estimated_S,
        "estimated_I": estimated_I,
        "estimated_R": estimated_R,
        "estimated_S_derivative": estimated_S_derivative,
        "estimated_I_derivative": estimated_I_derivative,
        "estimated_R_derivative": estimated_R_derivative,
    }
    estimated_df = pd.DataFrame(estimated_data)
    estimated_df.to_csv(f"{save_dir}/estimated_data.csv", index=False)
    print(f"Saved estimated data to {save_dir}/estimated_data.csv")


def time_saving(experiment_start_time, experiment_end_time, save_dir: str):
    """Save the time it took to run an experiment."""
    elapsed = experiment_end_time - experiment_start_time
    elapsed_minutes = elapsed / 60.0
    elapsed_hours = elapsed / 3600.0

    print(
        f"Time taken: {elapsed:.2f} seconds ({elapsed_minutes:.2f} min, {elapsed_hours:.2f} hr)\n"
    )

    # Save timing to text file
    timing_path = os.path.join(save_dir, "timing.txt")
    with open(timing_path, "w") as f:
        f.write(f"Time taken (seconds): {elapsed:.2f}\n")
        f.write(f"Time taken (minutes): {elapsed_minutes:.2f}\n")
        f.write(f"Time taken (hours): {elapsed_hours:.2f}\n")
