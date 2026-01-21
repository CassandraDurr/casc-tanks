"""Recursive expression builder."""

import sympy

# State variables using sympy
s, i, r = sympy.symbols("s i r")


class ExpressionBuilder:
    """Helper class to build a SymPy expression from a list of rules."""

    def __init__(
        self, action_list: list[str], const_offset: int = 0, partial_mode: bool = False
    ):
        """Initialise expression builder.

        Args:
            action_list (list[str]): List of grammar rules (strings).
            const_offset (int, optional): Integer offset for constant numbering.. Defaults to 0.
            partial_mode (bool, optional):
                - If True, returns symbol 'M' when rules run out.
                - If False, returns 1.0 (fallback for reward calc).
                - Defaults to False.
        """
        self.rules = iter(action_list)
        self.const_symbols = []  # State is stored here
        self.const_offset = const_offset  # Starting constant index
        self.partial_mode = partial_mode
        # Counter for non-terminal indexing
        self.m_counter = 0

    def build_expr(self) -> sympy.Expr:
        """Recursively work through rules iterator to build the expression."""
        try:
            rule = next(self.rules)
        except StopIteration:
            if self.partial_mode:
                # In partial mode, running out of rules means we hit a non-terminal
                # that hasn't been expanded yet. Return 'M{index}'.
                sym = sympy.Symbol(f"M{self.m_counter}")
                self.m_counter += 1
                return sym
            # Fallback for reward calculation (shouldn't happen)
            return sympy.Float(1.0)

        _, rhs = rule.split(" -> ")

        # Recursive mathematical expressions
        if rhs == "M + M":
            return self.build_expr() + self.build_expr()
        elif rhs == "M - M":
            return self.build_expr() - self.build_expr()
        elif rhs == "M * M":
            return self.build_expr() * self.build_expr()
        # Global sympy symbols (terminal rules)
        elif rhs == "s":
            return s
        elif rhs == "i":
            return i
        elif rhs == "r":
            return r
        # Constants (terminal rules)
        elif rhs == "C":
            # Create a unique constant symbol across fluxes
            c_idx = len(self.const_symbols) + self.const_offset
            c_sym = sympy.symbols(f"C{c_idx}")
            self.const_symbols.append(c_sym)
            return c_sym
        else:
            # Should not be reached
            return sympy.Float(1.0)


def sympy_expression_builder(
    action_list,
    const_offset: int = 0,
    partial_mode: bool = False,
) -> tuple[sympy.Expr, list[sympy.Symbol]]:
    """Convert a list of production rules into a SymPy expression for reward calculation."""
    # Use partial mode = False since these should be complete trees
    builder = ExpressionBuilder(
        action_list, const_offset=const_offset, partial_mode=partial_mode
    )
    # Call the recursive builder
    eq_expr = builder.build_expr()
    # Return the SymPy expression and constants
    return eq_expr, builder.const_symbols
