import networkx as nx
from common.types import ValidityReport


def check_validity(
    G: nx.Graph,
    L: dict[int, set[int]],
    coloring: dict[int, int],
) -> ValidityReport:
    """Independent validity checker - does not reuse any DSATUR internal logic.

    Checks:
    1. Every colored vertex has a color in its list
    2. No edge has both endpoints with the same color
    """
    conflict_edges: list[tuple[int, int]] = []
    violating_vertices: list[int] = []

    # Check list violations (track unique vertices with list violations only)
    list_violators: set[int] = set()
    for v, color in coloring.items():
        if v not in L or color not in L[v]:
            list_violators.add(v)

    # Check conflicts (monochromatic edges)
    conflict_edges: list[tuple[int, int]] = []
    for u, v in G.edges():
        if u in coloring and v in coloring and coloring[u] == coloring[v]:
            conflict_edges.append((u, v))

    # violating_vertices = union of list violators + endpoints of conflicts
    violating_vertices = list(list_violators | {x for e in conflict_edges for x in e})

    is_valid = len(conflict_edges) == 0 and len(list_violators) == 0

    return ValidityReport(
        is_valid=is_valid,
        num_conflicts=len(conflict_edges),
        num_list_violations=len(list_violators),
        conflict_edges=conflict_edges,
        violating_vertices=violating_vertices,
    )