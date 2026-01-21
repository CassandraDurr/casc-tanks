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
        initial_conditions_matrix: np.ndarray,
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
            initial_conditions_matrix: Shape (3, num_realisations)
            prevalence: Infection prevalence data, Shape (num_times, num_realisations)
            incidence: Infection incidence data, Shape (num_times-1, num_realisations)
            n_term: Choose between "max" and "avg" for parsimony calculation
        """
        self.eta = eta
        self.n_term = n_term
        self.state_variables = {s, i, r}

        # System structure
        self.system_structure = system_structure
        self.state_keys = state_keys

        # Data for ODE solving
        self.prevalence = prevalence
        self.incidence = incidence
        self.time_array = time_array
        self.initial_conditions_matrix = initial_conditions_matrix

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
                        return 0.0, ([], [])
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
                    c_reward, (c_exprs, c_strs) = self.reward_cache[cache_key]
                    if return_details:
                        return c_reward, (c_exprs, c_strs)
                    return c_reward

            # Build the full system equations from fluxes and system structure
            system_eqs = {}
            for state_key in self.state_keys:  # 's', 'i', 'r'
                state_eq = sympy.Float(0.0)
                for idx, flux_coeff in enumerate(self.system_structure[state_key]):
                    if flux_coeff != 0:
                        state_eq += flux_coeff * folded_flux_exprs[idx]
                system_eqs[state_key] = state_eq

            # Find the optimal constants and the resulting RMSE
            best_total_error, optimal_const_values = self.optimise_constants(
                system_eqs, all_consts
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
                    (folded_flux_exprs, final_flux_strs),
                )

            if return_details:
                # Also return details of the equation form
                return reward, (folded_flux_exprs, final_flux_strs)
            return reward

        except Exception as e:
            print(f"Reward exception: {e}")
            if return_details:
                return 0.0, ([], [])
            return 0.0

    def optimise_constants(
        self,
        system_eqs: dict[str, sympy.Expr],
        consts: list[sympy.Symbol],
    ) -> tuple[float, np.ndarray]:
        """
        Find the optimal values for constants to minimise error.

        Returns:
            tuple[float, np.ndarray]: (best_rmse, optimal_const_values)
        """
        num_consts = len(consts)

        # Lambdify all system equations (dict of symbolic expressions -> dict of callable functions)
        # Functions are per state (not flux)
        lambdified_eqs = {}
        for state_key, eq_expr in system_eqs.items():
            lambdified_eqs[state_key] = sympy.lambdify(
                [s, i, r] + consts, eq_expr, modules="numpy"
            )

        # Arguments for the objective function
        args_tuple = (
            lambdified_eqs,
            self.prevalence,
            self.incidence,
            self.time_array,
            self.initial_conditions_matrix,
        )

        # Handle case with no constants
        if num_consts == 0:
            # Just evaluate the expression directly
            error = self.objective_function(np.array([]), *args_tuple)
            return error, np.array([])

        # Find the optimal constants
        starting_guesses = [0.001, 0.01, 0.1]
        best_total_error = math.inf
        optimal_const_values = np.array([])

        for start_val in starting_guesses:
            x0 = np.full(num_consts, start_val)

            try:
                # Minimise objective using start_val for constants
                res = minimize(
                    self.objective_function,
                    x0,
                    args=args_tuple,
                    method="Powell",
                    tol=1e-4,
                )

                # If this attempt is better than previous ones, save it
                if res.fun < best_total_error:
                    best_total_error = res.fun
                    optimal_const_values = res.x

            except Exception:
                continue

        # If all attempts failed, return high penalty
        if best_total_error == math.inf:
            return 1e6, np.array([0.01] * num_consts)

        return best_total_error, optimal_const_values

    @staticmethod
    def objective_function(
        c_values: np.ndarray,
        lambdified_eqs: dict,
        prevalence: np.ndarray,  # Shape (num_times, num_realisations)
        incidence: np.ndarray,  # Shape (num_times-1, num_realisations)
        time_array: np.ndarray,  # Shape (num_times,)
        initial_conditions_matrix: np.ndarray,  # Shape (3, num_realisations)
    ) -> float:
        """
        MSE calculation using incidence and prevalence data.

        MSE per realisation = MSE(prevalence) + MSE(incidence).

        Returns the average MSE over all realisations.
        """
        c_args = tuple(c_values)
        failure_penalty = 1e6

        # Create a callable function for the ODE solver
        ode_func = lambda t_val, y_val, *args: sympy_ode(  # noqa: E731
            t_val, y_val, lambdified_eqs, c_args
        )

        try:
            # Solve ODEs for all realisations at once
            # Returns tensor of shape: (num_times, 3, num_realisations)
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
