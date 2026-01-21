"""Euler method for ODE solving with prediction boundaries."""

import typing

import numpy as np


def euler_method(
    func: typing.Callable,
    initial_conditions_matrix: np.ndarray,
    times: np.ndarray,
    args=(),
) -> np.ndarray:
    """
    Solve a system of ODEs for multiple realisations simultaneously.

    Added prediction boundary clipping for stability of constant optimisation routine.
    During constant optimisation a poor constant may be suggested which results in exploding predicted trajectories (infs).
    This results in the failure penalty being awarded, even in the actual equation is good.

    Args:
        func (typing.Callable): Function defining ODEs. Must accept y of shape (num_states, num_realisations) & return dy/dt of the same shape.
        initial_conditions_matrix (np.ndarray): Initial conditions matrix. Shape: (num_states, num_realisations).
        times (np.ndarray): Time points vector. Assumes all realisations share the same time grid. Shape (time,).
        args (tuple, optional): Extra arguments for func (constants). Defaults to ().

    Returns:
        np.ndarray: Predicted trajectories. Shape: (num_times, num_states, num_realisations).
    """  # noqa: E501
    # Boundary to clip predicted trajectory
    prediction_boundary = 1e4

    # Get dimensions
    num_times = len(times)
    num_states, num_realisations = initial_conditions_matrix.shape

    # Initialise the predicted trajectory matrix
    # Shape: (time, num_states, num_realisations)
    pred_traj = np.zeros((num_times, num_states, num_realisations))

    # Set initial conditions
    pred_traj[0] = initial_conditions_matrix

    # Calculate time steps
    dts = times[1:] - times[:-1]

    # Iterate through time steps
    for idx in range(num_times - 1):
        dt = dts[idx]
        current_state = pred_traj[idx]  # Shape: (num_states, num_realisations)

        # Calculate derivative for all realisations at once
        dy_dt = func(times[idx], current_state, *args)

        # Euler step
        next_state = current_state + dt * np.asarray(dy_dt)

        # Clip stability boundary
        pred_traj[idx + 1] = np.clip(
            next_state, -prediction_boundary, prediction_boundary
        )

    return pred_traj
