"""Count the number of terminal and non-terminal production rules used in a sympy expression."""

import sympy


def rules_count(expr, state_vars) -> int:
    """Count the total number of production rules used to build an expression."""
    terms_list = sympy.Add.make_args(expr)

    num_operations = 0  # number of non-terminal rules
    num_variable_uses = 0  # number of terminal rules

    # Only count unique constants once as a variable use
    constants_seen = set()

    # Count addition operations (number of terms - 1)
    if len(terms_list) > 1:
        num_operations += len(terms_list) - 1

    for term in terms_list:
        # Isolate constants and symbols in term
        constant_part, state_part = term.as_independent(*state_vars)

        # Constant multiplied by states
        coeff = 1  # No repetition (i.e. 3s = s + s + s)
        if constant_part.is_Integer and state_part.free_symbols:
            coeff = abs(int(constant_part))
            if coeff > 1:
                num_operations += coeff - 1

        # Count variable use of constants
        if constant_part.free_symbols and (constant_part not in constants_seen):
            # Add constant to seen set
            constants_seen.add(constant_part)
            num_variable_uses += 1

        # Check multiplication operation between constants and states
        if constant_part.free_symbols and state_part.free_symbols:
            num_operations += 1

        # Deal with states which may be composed of various factors
        if state_part.free_symbols:
            # Count multiplication operation between states
            if state_part.is_Mul:
                # if coeff > 1: 3*i*s = i*s + i*s + i*s
                num_operations += (len(state_part.args) - 1) * coeff

            # Separate the state into separate factors
            state_factors = state_part.args if state_part.is_Mul else [state_part]
            for state_factor in state_factors:
                # Handle power terms
                if state_factor.is_Pow and state_factor.exp > 0:
                    # Apply coeff multiplier to both variable uses and operations
                    num_variable_uses += state_factor.exp * coeff
                    num_operations += (state_factor.exp - 1) * coeff

                # Count variable use of state factor (non-power term)
                elif state_factor.free_symbols:
                    num_variable_uses += 1 * coeff

    # print("\nnum_operations", num_operations)
    # print("num_variable_uses", num_variable_uses)
    # print("total rules", num_operations + num_variable_uses, "\n")

    return num_operations + num_variable_uses
