import networkx as nx
from common.types import ColoringResult


def list_greedy(
    G: nx.Graph,
    L: dict[int, set[int]],
    order: list[int] | None = None,
) -> ColoringResult:
    """List greedy coloring algorithm.

    Process vertices in order (default: arbitrary/insertion order).
    Assign the lowest-indexed color in L(v) not used by an already-colored neighbor.
    Return explicit failure object if no such color exists.
    """
    if order is None:
        order = list(G.nodes())

    coloring: dict[int, int] = {}

    for v in order:
        if v not in G:
            continue
        neighbor_colors = {coloring[u] for u in G.neighbors(v) if u in coloring}
        available = [c for c in sorted(L[v]) if c not in neighbor_colors]
        if not available:
            return ColoringResult(success=False, coloring=coloring, failed_vertex=v)
        coloring[v] = available[0]

    return ColoringResult(success=True, coloring=coloring)