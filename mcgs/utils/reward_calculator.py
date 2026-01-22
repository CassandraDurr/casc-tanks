"""Reward calculation utilities for MCGS."""

import math
import threading

import numpy as np
import sympy
from scipy.optimize import minimize

from .const_folding import constant_folding
from .count_rules import rules_count
from .euler_method import euler_method
from .expression_builder import sympy_expression_builder

# State variables using sympy
s, i, r = sympy.symbols("s i r")


def sympy_ode(
    t_point: float, y_val: np.ndarray, lambdified_eqs: dict, c_args_tuple: tuple
) -> list[np.ndarray]:
    """
    Calculate derivatives.

    y is shape (3, num_realisations).
    Returns dy/dt of shape (3, num_realisations).
    """
    # Separate out states
    s_val, i_val, r_val = y_val

    # Evaluate symbolic functions
    dsdt = lambdified_eqs["s"](s_val, i_val, r_val, *c_args_tuple)
    didt = lambdified_eqs["i"](s_val, i_val, r_val, *c_args_tuple)
    drdt = lambdified_eqs["r"](s_val, i_val, r_val, *c_args_tuple)

    # If an equation is constant (e.g., C0), lambdify will return a scalar.
    # Broadcast it to shape (N,) to match the other states.
    if np.ndim(dsdt) == 0:
        dsdt = np.full_like(s_val, dsdt)
    if np.ndim(didt) == 0:
        didt = np.full_like(i_val, didt)
    if np.ndim(drdt) == 0:
        drdt = np.full_like(r_val, drdt)

    return [dsdt, didt, drdt]


class RewardCalculator:
    """Reward calculator for MCGS."""

    def __init__(
        self,
        eta: float,
        system_structure: dict,
        state_keys: list[str],
        time_array: np.ndarray,
        prevalence: np.ndarray,
        incidence: np.ndarray,
        n_term: str = "max",
    ):
        """Initialise reward calculator.

        Args:
            eta: Parsimony penalty factor
            system_structure: Stoichiometry matrix
            state_keys: List of state variable names (e.g., ['s', 'i', 'r'])
            time_array: Shape (num_times,)
            prevalence: Infection prevalence data, Shape (num_times, num_realisations)
            incidence: Infection incidence data, Shape (num_times-1, num_realisations)
            n_term: Choose between "max" and "avg" for parsimony calculation
        """
        self.eta = eta
        self.n_term = n_term
        self.state_variables = {s, i, r}

        # Number of states and realisations
        self.num_states = len(state_keys)
        self.num_realisations = prevalence.shape[1]

        # System structure
        self.system_structure = system_structure
        self.state_keys = state_keys

        # Data for ODE solving
        self.prevalence = prevalence
        self.incidence = incidence
        self.time_array = time_array

        # Thread safety
        self.lock = threading.Lock()
        self.reward_cache = {}

    def calculate_reward(
        self,
        flux_action_lists: list[list[str]],
        return_details: bool = False,
    ) -> float:
        """Calculate the reward for a completed parse graph with multiple fluxes.

        Function includes constant estimation.

        Reward function: r = (eta^n) * exp(- [MSE_incidence + MSE_prevalence] )
        - Incidence compares with infection flux (S->I)
        - Prevalence compares with infection prevalence (I)

        Args:
            flux_action_lists (list[list[str]]): Actions taken for each of the fluxes.
            return_details (bool, optional): Whether to return details of the computed best equation. Defaults to False.

        Raises:
            NotImplementedError: if self.n_term is not max or avg.

        Returns:
            float: reward, r
        """  # noqa: E501
        try:
            # Get messy flux expressions from the expression builder
            flux_exprs = []
            const_count = 0

            # Build SymPy expressions
            for action_list in flux_action_lists:
                # Pass the current constant count as the offset
                eq_expr, consts = sympy_expression_builder(
                    action_list, const_offset=const_count
                )
                flux_exprs.append(eq_expr)
                const_count += len(consts)

            # Penalise fluxes that don't use a state variable
            for expr in flux_exprs:
                if not expr.free_symbols.intersection(self.state_variables):
                    # This flux is invalid and should use at least one state variable.
                    if return_details:
                        return 0.0, ([], [], [])
                    return 0.0

            # Constant folding: turn messy expressions into neat expressions
            # with as few constants as required.
            folded_flux_exprs = []
            flux_rule_count = []
            all_consts = []
            k_index = 0

            # Apply folding to each flux individually
            for expr in flux_exprs:
                # Pass k_index to ensure fluxes have unique constants
                folded_expr, k_index, k_consts = constant_folding(
                    expr=expr, state_vars=self.state_variables, start_index=k_index
                )
                # Determine the number of production rules in simplified equation
                flux_rules = rules_count(
                    expr=folded_expr, state_vars=self.state_variables
                )
                all_consts.extend(k_consts)
                folded_flux_exprs.append(folded_expr)
                flux_rule_count.append(flux_rules)

            # Create a unique key for this system of equations
            cache_key = tuple(str(expr) for expr in folded_flux_exprs)

            # Check if we have already evaluated these set of equations
            with self.lock:
                if cache_key in self.reward_cache:
                    c_reward, (c_exprs, c_strs, c_initial_conditions) = (
                        self.reward_cache[cache_key]
                    )

                    if return_details:
                        return c_reward, (c_exprs, c_strs, c_initial_conditions)
                    return c_reward

            # Build the full system equations from fluxes and system structure
            system_eqs = {}
            for state_key in self.state_keys:  # 's', 'i', 'r'
                state_eq = sympy.Float(0.0)
                for idx, flux_coeff in enumerate(self.system_structure[state_key]):
                    if flux_coeff != 0:
                        state_eq += flux_coeff * folded_flux_exprs[idx]
                system_eqs[state_key] = state_eq

            # Find the optimal constants, optimal initial conditions, and the resulting RMSE
            best_total_error, optimal_const_values, optimal_initial_conditions = (
                self.optimise_param(system_eqs, all_consts)
            )

            # Calculate parsimony-penalised reward
            if self.n_term == "max":
                parsimony_n = max(flux_rule_count)
            elif self.n_term == "avg":
                parsimony_n = sum(flux_rule_count) / len(flux_rule_count)
            else:
                raise NotImplementedError()
            reward = (self.eta**parsimony_n) * math.exp(-best_total_error)

            # Store the top result (as a system)
            if optimal_const_values.size > 0:
                subs_dict = dict(zip(all_consts, optimal_const_values))
                final_flux_strs = [
                    str(expr.subs(subs_dict)) for expr in folded_flux_exprs
                ]
            else:
                final_flux_strs = [str(expr) for expr in folded_flux_exprs]

            # Cache the result
            with self.lock:
                self.reward_cache[cache_key] = (
                    reward,
                    (folded_flux_exprs, final_flux_strs, optimal_initial_conditions),
                )

            if return_details:
                # Also return details of the equation form
                return reward, (
                    folded_flux_exprs,
                    final_flux_strs,
                    optimal_initial_conditions,
                )
            return reward

        except Exception as e:
            print(f"Reward exception: {e}")
            if return_details:
                return 0.0, ([], [], [])
            return 0.0

    def optimise_param(
        self,
        system_eqs: dict[str, sympy.Expr],
        consts: list[sympy.Symbol],
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Find the optimal values for constants and initial conditions to minimise error.

        NOTE: The initial values for the initial conditions will need to change for cascading tanks.
        Also, the bounds will also need to be specified for the new problem.

        Returns:
            tuple[float, np.ndarray, np.ndarray]:
            (best_rmse, optimal_const_values, optimal_initial_conditions)
                - best_rmse: float, the minimum error achieved
                - optimal_const_values: shape (num_consts,)
                - optimal_initial_conditions: shape (num_states, num_realisations)
        """
        # Lambdify all system equations
        # Functions are per state (not flux)
        lambdified_eqs = {}
        for state_key, eq_expr in system_eqs.items():
            lambdified_eqs[state_key] = sympy.lambdify(
                [s, i, r] + consts, eq_expr, modules="numpy"
            )

        # Total parameters
        num_consts = len(consts)
        num_init_conds = self.num_states * self.num_realisations
        total_params = num_consts + num_init_conds

        # Starting optimisation co-ordinates
        x0 = np.zeros(total_params)
        x0[:num_consts] = 0.01  # constants

        # Starting optimisation co-ords for initial conditions
        # TODO: This needs to change when we want to move towards the cascading tanks setup
        initial_s = 1.0 - self.prevalence[0, :]  # Initial susceptibles
        initial_i = self.prevalence[0, :]  # Initial infected
        initial_r = 0.0 * np.ones(self.num_realisations)  # Initial recovered

        for real_idx in range(self.num_realisations):
            # Check normalisation
            total = initial_s[real_idx] + initial_i[real_idx] + initial_r[real_idx]
            if not np.isclose(total, 1.0):
                print(f"Warning: Initial conditions do not sum to 1. Received {total}.")
                initial_s[real_idx] /= total
                initial_i[real_idx] /= total
                initial_r[real_idx] /= total
            base_idx = num_consts + real_idx * self.num_states
            x0[base_idx] = initial_s[real_idx]
            x0[base_idx + 1] = initial_i[real_idx]
            x0[base_idx + 2] = initial_r[real_idx]

        # Bounds: Reasonable constant bounds & initial conditions in (0,1)
        bounds = [(-10.0, 10.0)] * num_consts
        for _ in range(self.num_realisations):
            bounds.extend([(0.0, 1.0)] * self.num_states)

        # Arguments for the objective function
        args_tuple = (
            lambdified_eqs,
            self.prevalence,
            self.incidence,
            self.time_array,
            num_consts,
        )

        # Optimise
        best_total_error = math.inf
        optimal_const_values = np.array([])
        optimal_initial_conditions = np.array([])

        res = minimize(
            self.objective_function,
            x0,
            args=args_tuple,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-6},
        )

        if res.success:
            best_total_error = res.fun
            optimal_params = res.x
            optimal_const_values = optimal_params[:num_consts]
            optimal_initial_conditions = optimal_params[num_consts:]
        else:
            raise ValueError(f"Optimisation failed: {res.message, res}")

        # Reshape optimal initial conditions to (num_states, num_realisations)
        optimal_initial_conditions = optimal_initial_conditions.reshape(
            self.num_realisations, self.num_states
        ).T

        return best_total_error, optimal_const_values, optimal_initial_conditions

    def objective_function(
        self,
        params: np.ndarray,
        lambdified_eqs: dict,
        prevalence: np.ndarray,  # Shape (num_times, num_realisations)
        incidence: np.ndarray,  # Shape (num_times-1, num_realisations)
        time_array: np.ndarray,  # Shape (num_times,)
        num_consts: int,
    ) -> float:
        """
        Objective function used in optimising constants and initial conditions.

        MSE calculation uses incidence and prevalence data.
        MSE per realisation = MSE(prevalence) + MSE(incidence).

        Returns the average MSE over all realisations.
        """
        failure_penalty = 1e6

        # Split parameters
        c_values = params[:num_consts]
        c_args = tuple(c_values)
        flat_init_conds = params[num_consts:]
        initial_conditions_matrix = flat_init_conds.reshape(
            self.num_realisations, self.num_states
        ).T  # shape (num_states, num_realisations)

        # Create a callable function for the ODE solver
        ode_func = lambda t_val, y_val, *args: sympy_ode(  # noqa: E731
            t_val, y_val, lambdified_eqs, c_args
        )

        try:
            # Solve ODEs for all realisations at once
            # Returns tensor of shape: (num_times, num_states, num_realisations)
            pred_traj_tensor = euler_method(
                func=ode_func,
                initial_conditions_matrix=initial_conditions_matrix,
                times=time_array,
                args=c_args,
            )

            # Check for numerical explosion (NaN/Inf)
            if not np.all(np.isfinite(pred_traj_tensor)):
                print("Numerical explosion in prediction")
                return failure_penalty

            # Extract predicted prevalence I
            predicted_prevalence = pred_traj_tensor[:, 1, :]
            # shape (time, num_realisations)

            # Extract predicted incidence (new infections per interval)
            predicted_susceptibles = pred_traj_tensor[:, 0, :]
            # shape (time, num_realisations)
            pred_incidence = -np.diff(predicted_susceptibles, axis=0)
            # shape (time-1, num_realisations)

            # MSE over time per realisation
            mse_prev_per_real = np.mean(
                (prevalence - predicted_prevalence) ** 2, axis=0
            )
            mse_inc_per_real = np.mean((incidence - pred_incidence) ** 2, axis=0)

            # Total error = average over realisations of (MSE_prev + MSE_inc)
            total_mse_per_real = mse_prev_per_real + mse_inc_per_real
            avg_total_mse = np.mean(total_mse_per_real)

            if np.isnan(avg_total_mse) or np.isinf(avg_total_mse):
                print("Numerical explosion in MSE")
                return failure_penalty

            return avg_total_mse

        except Exception as e:
            print(f"Fail in objective function calc: {e}")
            return failure_penalty
