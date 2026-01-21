"""Module to compute confidence intervals for rewards using binary KL divergence."""

import numpy as np
from scipy.optimize import root_scalar


def binary_kl_divergence(u, v, numerical_error=1e-12):
    """
    Binary KL divergence between Bernoulli(u) and Bernoulli(v).

    In the context of the paper, u represents the empirical mean reward (r_hat),
    and v is the variable we are optimising over to find confidence bounds.
    """
    # Convert to float (SymPy types may be passed in)
    u, v = float(u), float(v)
    # Numerical safety: v must be in (0, 1) to avoid infinite divergence
    u = np.clip(u, numerical_error, 1 - numerical_error)
    v = np.clip(v, numerical_error, 1 - numerical_error)
    return u * np.log(u / v) + (1 - u) * np.log((1 - u) / (1 - v))


def exploration_function(n_total: int, n_sa: int) -> float:
    """Exploration function for confidence intervals.

    Args:
        n_total (int): Total number of samples.
        n_sa (int): Number of times action a was taken in state s.

    Returns:
        float: Exploration bonus.
    """
    # Paper: np.log(n_total) / n_sa
    # - takes very long for confidence intervals to change from [0,1]

    # Alternative (1): sqrt(log(n_total)) / n_sa
    # Alternative (2): log(log(n_total)) / n_sa
    # Alternative (3): empirical_var*(log(n_total)) / n_sa

    # Or I could scale the exploration function by how far down the tree we are?
    # E.g., divide by complexity + 1
    return np.sqrt(np.log(n_total)) / n_sa


def confidence_intervals_rewards(
    r_hat: float, n_sa: int, n_total: int
) -> tuple[float, float]:
    """Compute [l_t, u_t] confidence interval for rewards using Brent's method.

    Formula: kl(r_hat, v) <= log(n_total) / n_sa

    Args:
        r_hat (float): Empirical mean reward.
        n_sa (int): Number of times action a was taken in state s.
        n_total (int): Total number of samples.

    Returns:
        tuple: Lower and upper confidence bounds (l_t, u_t).
    """
    if n_sa == 0:
        # No rollouts performed yet; return maximum uncertainty
        return 0.0, 1.0

    # Convert r_hat to float to avoid SymPy issues
    r_hat = float(r_hat)

    # Exploration function
    # -> beta_r is log(n) based on implementation details in the supplementary material
    divergence_budget = exploration_function(n_total, n_sa)

    # Define the root function for optimisation
    def root_function(v):
        return binary_kl_divergence(r_hat, v) - divergence_budget

    # Tolerance for root finding (balance between speed and accuracy)
    xtol = 1e-5  # Relative tolerance for x
    rtol = 1e-5  # Relative tolerance for function value

    # --- Calculate upper confidence bound u_t ---
    # If the max possible divergence (v=1) is within budget, u_t=1.
    if binary_kl_divergence(r_hat, 1.0) <= divergence_budget:
        u_t = 1.0
    else:
        try:
            # Max plausible reward >= empirical reward
            u_t = root_scalar(
                root_function,
                bracket=[r_hat, 1.0],
                method="brentq",
                xtol=xtol,
                rtol=rtol,
            ).root
        except Exception as e:
            raise Exception(f"Error computing upper confidence bound u_t: {e}")

    # --- Calculate lower confidence bound l_t ---
    # If the max possible divergence (v=0) is within budget, l_t=0.
    if binary_kl_divergence(r_hat, 0.0) <= divergence_budget:
        l_t = 0.0
    else:
        try:
            # Min plausible reward <= empirical reward
            l_t = root_scalar(
                root_function,
                bracket=[0.0, r_hat],
                method="brentq",
                xtol=xtol,
                rtol=rtol,
            ).root
        except Exception as e:
            raise Exception(f"Error computing lower confidence bound l_t: {e}")

    return l_t, u_t
