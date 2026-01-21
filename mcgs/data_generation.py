"""Module which generates S, I, R data using odeint."""

import numpy as np
from scipy import stats
from scipy.integrate import odeint


def sir_model(par: list, time: np.ndarray, beta: float, gamma: float) -> list:
    """SIR model differential equations.

    Args:
        par (list): List containing the current values of S, I, and R.
        time (np.ndarray): Array of time points at which to solve the ODE.
        beta (float): Infection rate of the disease.
        gamma (float): Recovery rate of the disease.

    Raises:
        ValueError: Require total population to be greater than 0.
        ValueError: Susceptible, infected, and recovered counts must be non-negative.
        ValueError: Beta and gamma must be non-negative.

    Returns:
        list: List containing the derivatives at each time point.
    """
    susceptible, infected, recovered = par
    population_size = susceptible + infected + recovered
    # Error handling for input parameters
    if population_size <= 0:
        raise ValueError("Total population N must be greater than 0")
    if susceptible < 0 or infected < 0 or recovered < 0:
        raise ValueError(
            "Susceptible, infected, and recovered counts must be non-negative"
        )
    if beta < 0 or gamma < 0:
        raise ValueError("Beta and gamma must be non-negative")
    # SIR model differential equations
    derivative_susceptible = -beta * susceptible * infected / population_size
    derivative_infected = (
        beta * susceptible * infected / population_size - gamma * infected
    )
    derivative_recovered = gamma * infected
    # Return the derivatives as a list
    return [derivative_susceptible, derivative_infected, derivative_recovered]


def sir_model_proportion(
    par: list, time: np.ndarray, beta: float, gamma: float
) -> list:
    """SIR model using proportions (S, I, R as fractions of total population).

    Args:
        par (list): List containing current values of S, I, and R (proportions).
        time (np.ndarray): Array of time points.
        beta (float): Infection rate.
        gamma (float): Recovery rate.

    Raises:
        ValueError: Proportions must be between 0 and 1 and sum to 1 (within tolerance).
        ValueError: Beta and gamma must be non-negative.

    Returns:
        list: Derivatives at each time point.
    """
    susceptible, infected, recovered = par
    total = susceptible + infected + recovered
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError("S, I, R proportions must sum to 1")
    if any(x < 0 or x > 1 for x in [susceptible, infected, recovered]):
        raise ValueError("S, I, R proportions must be between 0 and 1")
    if beta < 0 or gamma < 0:
        raise ValueError("Beta and gamma must be non-negative")
    derivative_susceptible = -beta * susceptible * infected
    derivative_infected = beta * susceptible * infected - gamma * infected
    derivative_recovered = gamma * infected
    return [derivative_susceptible, derivative_infected, derivative_recovered]


def ode_solver(
    par: list, time: np.ndarray, beta: float, gamma: float, proportion: bool = True
) -> np.ndarray:
    """Solve the SIR model using the odeint function from scipy.integrate.

    Args:
        par (list): List containing the initial values of S, I, and R.
        time (np.ndarray): Array of time points at which to solve the ODE.
        beta (float): Infection rate of the disease.
        gamma (float): Recovery rate of the disease.
        proportion (bool): If True, use proportions for S, I, R; otherwise use counts.

    Returns:
        np.ndarray: Array containing the solution of the SIR model at each time point.
    """
    # Solve the SIR model using odeint
    if proportion:
        result = odeint(sir_model_proportion, par, time, args=(beta, gamma))
    else:
        result = odeint(sir_model, par, time, args=(beta, gamma))
    return result


def generate_noisy_sir_data(
    par: list,
    time: np.ndarray,
    beta: float,
    gamma: float,
    proportion: bool = True,
    noise_std: float = 0.01,
) -> np.ndarray:
    """
    Generate noisy SIR data using the ode_solver.

    Args:
        par (list): Initial values of S, I, R (counts or proportions).
        time (np.ndarray): Time points.
        beta (float): Infection rate.
        gamma (float): Recovery rate.
        proportion (bool): Use proportions if True, else counts.
        noise_std (float): Standard deviation of Gaussian noise.

    Returns:
        np.ndarray: Noisy SIR data.
    """
    # Obtain clean SIR data
    clean_data = ode_solver(par, time, beta, gamma, proportion=proportion)
    susceptible, infected, recovered = clean_data.T

    # Get population size
    population_size = susceptible[0] + infected[0] + recovered[0]

    # Generate noisy data
    # Infections
    infected_noisy = stats.norm.rvs(loc=infected, scale=noise_std, size=infected.shape)
    infected_noisy = np.clip(infected_noisy, 0, population_size)

    # Recoveries
    recovered_noisy = stats.norm.rvs(
        loc=recovered,
        scale=noise_std,
        size=recovered.shape,
    )
    upper_bound = population_size - infected_noisy
    recovered_noisy = np.clip(recovered_noisy, 0, upper_bound)

    # Susceptibles
    susceptible_noisy = population_size - recovered_noisy - infected_noisy

    # Stack the noisy data
    noisy_data = np.vstack([susceptible_noisy, infected_noisy, recovered_noisy]).T

    return noisy_data


def get_incidence(clean_susceptible: np.ndarray, N: int = 1) -> np.ndarray:
    """Calculate incidence (new infections) from susceptible data.

    Incidence = Poisson distributed with mean equal to the decrease in susceptibles.

    Args:
        clean_susceptible (np.ndarray): Array of clean susceptible counts or proportions over time.
        N (int, optional): Total population size. Defaults to 1.

    Returns:
        np.ndarray: Array of incidence values.
    """
    new_infections = -np.diff(clean_susceptible) * N  # Shape (T-1,)
    incidence = np.random.poisson(lam=new_infections)
    return incidence / N  # Return as proportion if N > 1


def get_SIR_trajectories(
    sampling_step: int = 1,
    keep_fraction: float = 1.0,
    noise_level: float = 5.0,
    days: int = 150,
    initial_conditions: tuple[int, int, int] = (100, 2, 0),
    rate_parameters: tuple[float, float] = (0.3, 0.1),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate time series and SIR trajectories.

    Default time steps is every day from 0 to days.

    Args:
        sampling_step (int, optional): Sample every x time steps. Defaults to 1.
        keep_fraction (float, optional): Sample x% of the time steps. Defaults to 1.0.
        noise_level (int, optional): Amount of noise in the system. Defaults to 5.
        days (int, optional): Number of days for the trajectory. Defaults to 150.
        initial_conditions (tuple[int, int, int], optional): Number in S, I, R compartments at time 0. Defaults to (100, 2, 0).
        rate_parameters (tuple[float, float], optional): Beta and gamma parameters. Defaults to (0.3, 0.1).

    Raises:
        ValueError: Keep fraction is not between 0 and 1.
        ValueError: Noise is a negative value.
        ValueError: Sampling step is not a positive integer.

    Returns:
        tuple[np.ndarray, np.ndarray]: Time index and SIR trajectories.
    """  # noqa: E501
    if keep_fraction > 1 or keep_fraction < 0:
        raise ValueError(
            f"Keep fraction should be between 0 and 1. Received {keep_fraction}."
        )

    if noise_level < 0:
        raise ValueError(
            f"Noise level should be a positive number. Received {noise_level}."
        )

    if sampling_step < 0 or not isinstance(sampling_step, int):
        raise ValueError(
            f"Sampling step should be a positive integer. Received {noise_level}."
        )

    # Initialise variables
    init_S, init_I, init_R = initial_conditions
    beta, gamma = rate_parameters

    # Get initial conditions
    N = init_S + init_I + init_R
    init_S_prop, init_I_prop, init_R_prop = init_S / N, init_I / N, init_R / N
    initial_conditions_prop = [init_S_prop, init_I_prop, init_R_prop]

    # Base times
    times = np.arange(0, days, 1)

    # Sparsity (with regular spacing)
    if sampling_step > 1:
        times = times[::sampling_step]

    # Sparsity (with irregular spacing)
    if keep_fraction < 1:
        num_points = len(times)
        # Calculate how many data points to keep
        num_to_keep = int(num_points * keep_fraction)

        # Always keep the first data point (t=0)
        # Randomly choose the rest of the indices from the remaining points
        other_indices = np.random.choice(
            np.arange(1, num_points), size=num_to_keep - 1, replace=False
        )
        keep_indices = np.concatenate([np.array([0]), other_indices])
        keep_indices.sort()  # Keep the time points in order

        # Return the sparse data and corresponding time points
        times = times[keep_indices]

    # Noise
    if noise_level > 0:
        sol_prop = generate_noisy_sir_data(
            initial_conditions_prop,
            times,
            beta,
            gamma,
            proportion=True,
            noise_std=noise_level / N,
        )
    else:
        sol_prop = ode_solver(
            initial_conditions_prop, times, beta, gamma, proportion=True
        )

    return times, sol_prop
