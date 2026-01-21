"""Constant folding to simplify expressions prior to optimisation."""

import sympy


def constant_folding(expr, state_vars, start_index=0):
    """
    Simplifies an expression by folding constants.

    Returns: (folded_expression, next_constant_index)
    """
    # Expand expression (multiply out)
    expanded_exp = sympy.expand(expr, state_vars)

    # Define a list to store the folded-constant terms and a dictionary for constant mapping
    folded_terms = []
    constant_map = {}
    new_constant_index = start_index

    # Separate out terms at +
    terms_list = sympy.Add.make_args(expanded_exp)

    # --- Perform multiplicative constant folding ---
    for term in terms_list:
        # Isolate constants in term and symbols in term
        constant_part, state_part = term.as_independent(*state_vars)

        # If the constant part is a number, keep it as is.
        if not constant_part.free_symbols:
            updated_constant_part = constant_part

        # Fold the constants if they are symbolic
        elif constant_part in constant_map:
            # Use the existing mapped constant
            updated_constant_part = constant_map[constant_part]
        else:
            # Create a new symbol for the folded constant
            updated_constant_part = sympy.Symbol(f"K{new_constant_index}")

            # Store the constant mapping
            constant_map[constant_part] = updated_constant_part
            new_constant_index += 1

        # Put the folded constant and the state part back together
        folded_term = updated_constant_part * state_part

        # Add the new folded term to the list
        folded_terms.append(folded_term)

    # Put all the folded terms back together
    folded_exp = sum(folded_terms)

    # --- Perform additive constant folding ---
    coeff_accumulator = {}  # Key: state_part, Value: sum of K coefficients

    # Separate out terms at +
    terms_list = sympy.Add.make_args(folded_exp)

    for term in terms_list:
        # Split coefficient from state
        coeff, state_part = term.as_independent(*state_vars)

        # Add to dictionary
        if state_part in coeff_accumulator:
            coeff_accumulator[state_part] += coeff
        else:
            coeff_accumulator[state_part] = coeff

    # Storage for folded terms
    final_folded_terms = []

    for state_part, coefficient in coeff_accumulator.items():

        # If the combined coefficient is purely a number, don't fold it.
        if not coefficient.free_symbols:
            updated_coefficient = coefficient

        # Check if coefficient is already a pure K symbol
        elif coefficient in constant_map.values():
            updated_coefficient = coefficient

        # Create a new K-symbol if new coeff combination
        else:
            while sympy.Symbol(f"K{new_constant_index}") in constant_map.values():
                new_constant_index += 1

            new_K_symbol = sympy.Symbol(f"K{new_constant_index}")

            # Map the sum to the new symbol for tracking
            constant_map[coefficient] = new_K_symbol

            updated_coefficient = new_K_symbol

            # Increment const index
            new_constant_index += 1

        # Recombine the fully folded coefficient with the state part
        final_folded_term = updated_coefficient * state_part
        final_folded_terms.append(final_folded_term)

    # Sort the terms alphabetically by their state variable string (i, r, s...)
    final_folded_terms.sort(key=lambda x: str(x.as_independent(*state_vars)[1]))

    # Create the substitution map based on the sorted order
    rename_map = {}
    final_b_index = start_index

    for term in final_folded_terms:
        # Extract the coefficient (which is a K symbol or a number)
        coeff, state_part = term.as_independent(*state_vars)

        # Rename symbolic constants
        if coeff.free_symbols:
            # If we haven't renamed this K yet, give it the next B number
            if coeff not in rename_map:
                new_b = sympy.Symbol(f"B{final_b_index}")
                rename_map[coeff] = new_b
                final_b_index += 1

    # Construct the final expression
    temp_sum = sum(final_folded_terms)
    final_folded_exp_reindexed = temp_sum.subs(rename_map)

    # Calculate final outputs
    final_consts = list(rename_map.values())

    # Sort B-consts to look nice in the list [B0, B1, B2]
    final_consts.sort(key=lambda s: int(s.name[1:]))

    return final_folded_exp_reindexed, final_b_index, final_consts


def collect_partial_state_vars(expr, base_state_vars=None):
    """
    Collect state variables for partial expressions.

    State variables:
      - base state variables: {s, i, r} (if not provided)
      - any M{int} symbols present in the expression
    """
    if base_state_vars is None:
        s_sym, i_sym, r_sym = sympy.symbols("s i r")
        base_state_vars = {s_sym, i_sym, r_sym}
    else:
        base_state_vars = set(base_state_vars)

    # Add all M{int} symbols referenced in expr
    m_syms = {
        sym
        for sym in expr.free_symbols
        if sym.name.startswith("M") and sym.name[1:].isdigit()
    }
    return base_state_vars | m_syms


def constant_folding_partial(expr, state_vars_base=None, start_index=0):
    """
    Constant folding for partial expressions that may contain M0, M1, ... tokens.

    Args:
        expr (sympy.Expr): The partial expression to fold.
        state_vars_base (set[sympy.Symbol] | None):
            Base state variables (defaults to s, i, r if None).
        start_index (int): Starting index for introduced K/B symbols.

    Returns:
        tuple: (folded_expression, next_constant_index, folded_B_constants)
    """
    state_vars = collect_partial_state_vars(expr, state_vars_base)
    return constant_folding(expr=expr, state_vars=state_vars, start_index=start_index)
