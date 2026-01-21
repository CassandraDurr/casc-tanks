"""Visualise graph as a .dot file."""

import textwrap


def export_mcgs_to_graphviz(mcgs_instance, filename="mcgs_graph.dot"):
    """
    Traverses the MCGS graph and exports it to a Graphviz DOT file.

    Visual Style:
    - Terminal Nodes (Leafs): Boxes, colored green. Show Reward & Equation.
    - Internal Nodes: Ovals. Show Bounds [L, U].
    - Edges: Labeled with the action taken.
    """
    nodes_visited = set()

    # Header for DOT file
    dot_lines = [
        "digraph MCGS {",
        "rankdir=TB;",
        'node [fontname="Arial", shape=ellipse, style=filled, fillcolor=white];',
        'edge [fontname="Arial", fontsize=10];',
    ]

    # Use a stack for DFS traversal
    stack = [mcgs_instance.root]

    while stack:
        node = stack.pop()

        # Use Python's built-in id() as a unique node identifier for the graph
        node_id = str(id(node))

        if node_id in nodes_visited:
            continue
        nodes_visited.add(node_id)

        # --- 1. Define Node Visuals ---
        if node.is_terminal_flag:
            # TERMINAL NODE
            reward = f"{node.best_reward:.4f}"

            # Extract equation string if available
            eq_str = "Invalid/Incomplete"
            if node.best_result_data:
                # best_result_data is (flux_exprs, final_flux_strs)
                _, final_strs = node.best_result_data
                # Join multiple fluxes with newline
                eq_str = "\\n".join(final_strs)

            # Wrap long equations for readability
            eq_str = "\\n".join(textwrap.wrap(eq_str, width=30))

            label = f"R: {reward}\\n{eq_str}"
            dot_lines.append(
                f'{node_id} [shape=box, fillcolor=lightgreen, label="{label}"];'
            )

        else:
            # INTERNAL NODE
            # Show bounds L (Lower) and U (Upper)
            label = f"L: {node.L:.3f}\\nU: {node.U:.3f}"

            # Check if this node is a merge point (has multiple parents)
            color = "lightblue" if len(node.parents) > 1 else "white"

            dot_lines.append(f'{node_id} [fillcolor={color}, label="{label}"];')

        # --- 2. Process Children (Edges) ---
        for action, child_node in node.children.items():
            child_id = str(id(child_node))

            # Clean up action label (remove ' -> ' for cleaner arrows if needed)
            # keeping it full here for clarity
            edge_label = action.replace('"', '\\"')  # Escape quotes

            dot_lines.append(f'{node_id} -> {child_id} [label="{edge_label}"];')

            # Add child to stack to continue traversal
            stack.append(child_node)

    dot_lines.append("}")

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(dot_lines))

    print(f"Graph exported to {filename}. Contains {len(nodes_visited)} nodes.")
