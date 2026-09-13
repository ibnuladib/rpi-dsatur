import networkx as nx
import heapq
from common.types import ColoringResult


def list_dsatur(
    G: nx.Graph,
    L: dict[int, set[int]],
    fixed: dict[int, int] | None = None,
) -> ColoringResult:
    """List DSATUR coloring algorithm with optional fixed vertices.

    Priority queue with dynamic recomputation:
    - Pick uncolored vertex with highest saturation degree (distinct colors among colored neighbors, including fixed)
    - Tie-break by largest |L(v)| / (deg(v) + 1)
    - Tie-break by vertex id (for determinism)
    - Assign lowest-indexed color in L(v) not used by colored neighbor
    - If none exists, fail and return partial coloring + failing vertex

    Fixed vertices are never recolored; if a fixed neighbor makes a vertex uncolorable, report failure.
    """
    if fixed is None:
        fixed = {}

    # Initialize coloring with fixed vertices
    coloring: dict[int, int] = dict(fixed)

    # Track uncolored vertices
    uncolored = set(G.nodes()) - set(fixed.keys())

    if not uncolored:
        return ColoringResult(success=True, coloring=coloring)

    # For tracking saturation degrees efficiently
    def compute_saturation(v: int) -> int:
        neighbor_colors = set()
        for u in G.neighbors(v):
            if u in coloring:
                neighbor_colors.add(coloring[u])
        return len(neighbor_colors)

    def list_ratio(v: int) -> float:
        deg = G.degree(v)
        return len(L.get(v, set())) / (deg + 1) if (deg + 1) > 0 else 0.0

    # Build initial priority queue: (-saturation, -list_ratio, vertex_id)
    # Using negative for max-heap behavior
    heap: list[tuple[int, float, int]] = []
    for v in uncolored:
        sat = compute_saturation(v)
        ratio = list_ratio(v)
        heapq.heappush(heap, (-sat, -ratio, v))

    while heap:
        neg_sat, neg_ratio, v = heapq.heappop(heap)

        if v not in uncolored:
            continue  # stale entry

        # Get available colors
        neighbor_colors = {coloring[u] for u in G.neighbors(v) if u in coloring}
        available = [c for c in sorted(L[v]) if c not in neighbor_colors]

        if not available:
            return ColoringResult(success=False, coloring=coloring, failed_vertex=v)

        coloring[v] = available[0]
        uncolored.remove(v)

        # Update saturation for uncolored neighbors
        for u in G.neighbors(v):
            if u in uncolored:
                new_sat = compute_saturation(u)
                new_ratio = list_ratio(u)
                heapq.heappush(heap, (-new_sat, -new_ratio, u))

    return ColoringResult(success=True, coloring=coloring)