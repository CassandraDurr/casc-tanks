"""Monte Carlo Graph Search (MCGS) class."""

from __future__ import annotations

import csv
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sympy

from .const_folding import constant_folding, constant_folding_partial
from .expression_builder import sympy_expression_builder
from .grammar_probabilities import initialise_grammar_probabilities_util
from .reward_calculator import RewardCalculator
from .symbolic_graph_node import SymbolicNode

# State variables using sympy
s, i, r = sympy.symbols("s i r")


class MCGS:
    """MCGS (GBOP) for dynamical symbolic regression of coupled ODE systems."""

    def __init__(
        self,
        data_X: list[dict],  # time, incidence, prevalence
        system_structure: dict,  # Stoichiometry
        num_fluxes: int,
        grammar: dict,
        flux_priors: dict,
        initial_state: tuple,
        t_max: int,
        eta: float,
        gamma: float,
        epsilon: float,
        top_N: int,
        rollouts_per_leaf: int,
    ):
        """Initialise graph instance."""
        self.data_X = data_X
        self.grammar = grammar
        self.flux_priors = flux_priors
        self.system_structure = system_structure
        self.num_fluxes = num_fluxes
        self.state_keys = list(system_structure.keys())
        self.state_variables = {s, i, r}
        self.total_samples = 0

        # Hyperparameters
        self.t_max = t_max
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.top_N = top_N
        self.rollouts_per_leaf = rollouts_per_leaf

        # Update threshold
        self.update_threshold = max(
            ((1 - self.gamma) / self.gamma) * self.epsilon, 0.01
        )

        # Grammar helpers
        self.non_terminals = set(grammar.keys())
        self.non_terminals_regex = re.compile(f"({'|'.join(self.non_terminals)})")
        self.terminal_rules_for_M = ["M -> s", "M -> i", "M -> r", "M -> C"]

        # Prevent recalculating expesive reward function
        # Cache: {tuple(str_exprs): (reward, details)}
        self.reward_cache = {}

        # Intialise flux prior probabilities and rules
        self.initialise_grammar_probabilities()

        # Get time, data, and intial conditions (vectorised for replications)
        self.vectorise_data(data_X=data_X)

        # Initialise reward calculator
        self.reward_calculator = RewardCalculator(
            eta=self.eta,
            system_structure=self.system_structure,
            state_keys=self.state_keys,
            time_array=self.time_array,
            prevalence=self.true_prevalence,
            incidence=self.true_incidence,
        )

        # Transposition table: maps state hash -> SymbolicNode
        self.graph_nodes = {}

        # Initialise root (and add to graph)
        self.root = SymbolicNode(state=initial_state, graph=self)
        root_key = self.hash_state(self.root.state)
        self.graph_nodes[root_key] = self.root

    def vectorise_data(self, data_X):
        """Vectorise time, data, and intial conditions for multiple replications."""
        # Time vector: assumption = realisations share the same time steps.
        self.time_array = data_X[0]["time"]

        # Prevalence
        self.true_prevalence = np.stack(
            [data["prevalence"] for data in self.data_X], axis=-1
        )  # (num_times, realisations)

        # Incidence
        self.true_incidence = np.stack(
            [data["incidence"] for data in self.data_X], axis=-1
        )  # (num_times-1, realisations)

    def initialise_grammar_probabilities(self):
        """
        Calculate normalised probabilities for the flux production rules based on priors.

        Valid M rules used in rollout & getting untried actions, valid terminal M rules used in force-completion.
        Valid M terminal state rules used in rollout to ensure branches have at least one state.

        Define:
            - self.valid_M_rules: rules allowed per flux
            - self.M_rule_probs: probabilities of rules allowed per flux
            - self.valid_terminal_M_rules: terminal rules allowed per flux
            - self.terminal_M_rule_probs: probabilities of terminal rules allowed per flux
            - self.valid_state_terminal_M_rules: terminal rules allowed per flux containing state values
            - self.terminal_M_rule_probs: probabilities of terminal rules allowed per flux containing state values
        """  # noqa: E501
        (
            self.valid_M_rules,
            self.M_rule_probs,
            self.valid_terminal_M_rules,
            self.terminal_M_rule_probs,
            self.valid_state_terminal_M_rules,
            self.valid_state_terminal_M_rule_probs,
        ) = initialise_grammar_probabilities_util(
            grammar=self.grammar,
            flux_priors=self.flux_priors,
            terminal_rules_for_M=self.terminal_rules_for_M,
            num_fluxes=self.num_fluxes,
        )

    def hash_state(self, state) -> tuple:
        """
        Create a unique key for the node.

        SymPy simplifies both partial (internal nodes) and complete expressions (terminal nodes).
        """
        action_lists, stacks, _ = state

        # Check if terminal (all stacks empty)
        is_terminal = all(not stack for stack in stacks)

        if is_terminal:
            # Simplify full expression (fold constants, etc)
            try:
                canonical_form = self.get_simplified_exprs(
                    action_lists, partial_mode=False
                )
                return ("TERMINAL", canonical_form)
            except Exception as e:
                print(f"Hash exception for terminal node: {e}")
                return ("TERMINAL", tuple(tuple(x) for x in action_lists))
        else:
            # Internal node: Simplify partial expression
            try:
                partial_structure = self.get_simplified_exprs(
                    action_lists, partial_mode=True
                )
                return ("INTERNAL", partial_structure)
            except Exception as e:
                print(f"Hash exception for internal node: {e}")
                return ("INTERNAL", tuple(tuple(x) for x in action_lists))

    def get_simplified_exprs(
        self, flux_action_lists: list[list[str]], partial_mode: bool = False
    ) -> tuple:
        """Convert actions to a canonical, simplified string representations.

        Args:
            flux_action_lists (list[list[str]]): List of action lists per flux.
            partial_mode (bool, optional): Whether to simplify partial ODEs. Defaults to False.

        Returns:
            tuple: Tuple of simplified expression strings per flux.
        """
        folded_strs = []
        const_count = 0  # constant offset for initial, messy expressions
        k_index = 0  # constant offset for clean, folded expressions

        for action_list in flux_action_lists:
            # Build the messy expression
            eq_expr, consts = sympy_expression_builder(
                action_list,
                const_offset=const_count,
                partial_mode=partial_mode,
            )
            const_count += len(consts)

            if partial_mode:
                # Simplify and fold constants for partial expressions
                folded_expr, k_index, _ = constant_folding_partial(
                    expr=eq_expr,
                    state_vars_base=self.state_variables,
                    start_index=k_index,
                )
            else:
                # Simplify and fold constants for terminal expressions
                folded_expr, k_index, _ = constant_folding(
                    expr=eq_expr, state_vars=self.state_variables, start_index=k_index
                )
            folded_strs.append(str(folded_expr))

        return tuple(folded_strs)

    def get_or_create_node(self, state) -> SymbolicNode:
        """Check transposition table before creating a new node."""
        state_key = self.hash_state(state)

        if state_key in self.graph_nodes:
            # MERGE: Return existing node
            return self.graph_nodes[state_key]
        else:
            # CREATE: Return new node
            new_node = SymbolicNode(state=state, graph=self)
            self.graph_nodes[state_key] = new_node
            return new_node

    def propagate_upwards(self, start_node: SymbolicNode):
        """
        Efficiently propagate updates using Algorithm 4: queue-based implementation.

        Algorithm 4 is in the supplementary material of MCGS paper.
        """
        # Queue q initialised with s_n (q <- [s_n])
        processing_queue = [start_node]

        # Set of nodes in queue to avoid duplicates
        queue_set = {start_node}

        # Add max iterations to avoid infinite loops (should not happen in practice)
        max_iterations = 10000
        iterations = 0
        while processing_queue and iterations < max_iterations:
            iterations += 1
            if iterations == max_iterations:
                print("Warning: Max iterations reached in propagate_upwards.")
                break

            # Pop the first node s'
            node = processing_queue.pop(0)
            queue_set.remove(node)

            # Apply monotonicity here to only tighten bounds
            # Bounds proposed by Bellman operator
            new_U_candidate, new_L_candidate = node.calculate_bellman_values()

            # Enforce monotonicity of bounds: U is non-increasing; L is non-decreasing
            new_U = min(node.U, new_U_candidate)
            new_L = max(node.L, new_L_candidate)

            # Check stopping condition for the upper bound U
            # If |new U - old U| > threshold, we update and propagate.
            # We also always propagate if it's the start_node (expanded node),
            # or if the lower bound improves.
            should_propagate = False
            update_condition = (
                (abs(new_U - node.U) > self.update_threshold)
                or (new_L > node.L)
                or (node == start_node)
            )
            if update_condition:
                node.U = new_U
                node.L = new_L
                should_propagate = True

            # If updated, push predecessors s to queue q
            if should_propagate:
                for parent in node.parents:
                    if parent not in queue_set:
                        processing_queue.append(parent)
                        queue_set.add(parent)

    def run_search(
        self,
        episodes: int = 100,
        steps: int = 100,
        print_epi: int = 10,
    ) -> list:
        """Run the GBOP search algorithm (stochastic rewards, deterministic transitions)."""
        with ThreadPoolExecutor() as executor:
            for episode in range(episodes):
                # Restart at root node per epsisode
                node = self.root
                # Track nodes visited in this trajectory for a single final propagation
                trajectory_nodes = [node]

                # Print progress
                if episode % print_epi == 0:
                    print(f"Episode {episode}/{episodes}...")

                for step in range(steps):
                    # --- 0A. CHECK IF TERMINAL ---
                    if node.is_terminal_flag:
                        # No children possible for selection
                        print(f"   Reached terminal node at step {step + 1}")
                        break

                    # --- 0B. MAKE CHILDREN ---
                    # Check if the node actually has any children to select from
                    # If there are no children nodes get - create them so you can select them
                    if not node.children:
                        # Expand all untried actions to create all children
                        children = node.expand_all_children()
                        # Rollout each child once to initialise U/L and selection (warm start)
                        if children:
                            init_futures = [
                                executor.submit(child.rollout) for child in children
                            ]
                            init_results = [
                                init_future.result() for init_future in init_futures
                            ]
                            for child, (reward, data) in zip(children, init_results):
                                child.update_stats(reward, result_data=data)

                            # Propagate from parent (bellman update uses newly expanded children)
                            self.propagate_upwards(node)

                    # --- 1. SELECTION / SAMPLING ---
                    node = node.best_child()
                    trajectory_nodes.append(node)

                    # --- 2. SIMULATION ---
                    futures = [
                        executor.submit(node.rollout)
                        for _ in range(self.rollouts_per_leaf)
                    ]
                    results = [future.result() for future in futures]

                    # --- 3. UPDATE STATS ---
                    for reward, data in results:
                        node.update_stats(reward, result_data=data)

                # --- 4. GRAPH PROPAGATION ---
                # Propagate updates from the last node in the trajectory
                # (should include all ancestor nodes)
                self.propagate_upwards(trajectory_nodes[-1])

        return self.get_top_results_from_graph()

    def traverse_graph_and_collect_results(self):
        """
        Traverse the graph and collect all stored results.

        Returns:
            list[tuple]: A list of tuples containing (reward, result_data).
        """
        all_results = []
        visited = set()  # Required for graph merges (multiple parents)
        stack = [self.root]

        while stack:
            node = stack.pop()

            # Skip if we have already processed this node
            if node in visited:
                continue
            visited.add(node)

            # Collect result
            if node.best_result_data is not None:
                # node.best_result_data format: (flux_exprs, final_flux_strs)
                flux_exprs, final_flux_strs = node.best_result_data
                all_results.append((node.best_reward, flux_exprs, final_flux_strs))

            # Add children
            stack.extend(node.children.values())

        # Sort by reward descending
        all_results.sort(key=lambda x: x[0], reverse=True)

        return all_results

    def get_top_results_from_graph(self):
        """Get the top N results from the nodes."""
        all_results = self.traverse_graph_and_collect_results()
        return all_results[: self.top_N]

    def export_results_csv(self, filepath: str) -> None:
        """
        Write a CSV with all stored results in the graph.

        Columns: rank, reward, equations, equations_with_constants
        """
        all_results = self.traverse_graph_and_collect_results()

        # Prepare rows
        rows = []
        for rank, (reward, folded_exprs, final_strs) in enumerate(all_results, start=1):
            # folded_exprs may be sympy.Expr objects
            folded_strs = [str(e) for e in folded_exprs] if folded_exprs else []
            eq_symbolic = " ; ".join(folded_strs)
            eq_with_consts = " ; ".join(final_strs) if final_strs else ""
            rows.append((rank, float(reward), eq_symbolic, eq_with_consts))

        # Write CSV
        with open(filepath, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["rank", "reward", "equations", "equations_with_constants"])
            writer.writerows(rows)

    def export_graph_bounds(self, filepath: str) -> None:
        """Export the bounds (L, U) and hash of every node in the graph."""
        # Order by n_sa (visit counts) and then best_reward
        sorted_nodes = sorted(
            self.graph_nodes.items(),
            key=lambda item: (item[1].visit_counts, item[1].best_reward),
            reverse=True,
        )
        rows = []

        # Iterate over the transposition table (all registered nodes)
        for state_hash, node in sorted_nodes:
            hash_str = str(state_hash)
            rows.append(
                (
                    hash_str,
                    node.L,
                    node.U,
                    node.best_reward,
                    node.var_rewards if node.var_rewards is not None else "",
                    node.mean_reward,
                    node.visit_counts,
                    node.l_CI if node.l_CI is not None else 0.0,
                    node.u_CI if node.u_CI is not None else 1.0,
                    node.is_terminal_flag,
                )
            )

        with open(filepath, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "node_hash",
                    "L_bound",
                    "U_bound",
                    "best_reward_seen",
                    "variance_rewards",
                    "average_reward",
                    "visit_count",
                    "lower_confidence_interval",
                    "upper_confidence_interval",
                    "is_terminal",
                ]
            )
            writer.writerows(rows)
