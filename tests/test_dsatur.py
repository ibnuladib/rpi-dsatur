import pytest
import networkx as nx
import random
from baselines.dsatur import list_dsatur
from baselines.validity import check_validity
from common.types import ColoringResult


def test_dsatur_simple():
    """Simple graph - 3 vertices in a line."""
    G = nx.path_graph(3)
    L = {0: {1, 2}, 1: {1, 2}, 2: {1, 2}}
    result = list_dsatur(G, L)
    assert result.success
    validity = check_validity(G, L, result.coloring)
    assert validity.is_valid


def test_dsatur_complete_graph():
    """Complete graph K3 with 3 colors."""
    G = nx.complete_graph(3)
    L = {0: {1, 2, 3}, 1: {1, 2, 3}, 2: {1, 2, 3}}
    result = list_dsatur(G, L)
    assert result.success
    validity = check_validity(G, L, result.coloring)
    assert validity.is_valid


def test_dsatur_infeasible():
    """Infeasible: K3 with only 2 colors."""
    G = nx.complete_graph(3)
    L = {0: {1, 2}, 1: {1, 2}, 2: {1, 2}}
    result = list_dsatur(G, L)
    assert not result.success
    assert result.failed_vertex is not None


def test_dsatur_fixed_vertices():
    """Test that fixed vertices are never changed."""
    G = nx.path_graph(3)
    L = {0: {1, 2}, 1: {1, 2}, 2: {1, 2}}
    fixed = {0: 1}  # Fix vertex 0 to color 1
    result = list_dsatur(G, L, fixed=fixed)
    assert result.success
    assert result.coloring[0] == 1  # Fixed vertex unchanged
    validity = check_validity(G, L, result.coloring)
    assert validity.is_valid


def test_dsatur_fixed_conflict():
    """Test that fixed vertex causing conflict reports failure."""
    G = nx.path_graph(2)
    L = {0: {1}, 1: {1}}
    fixed = {0: 1}  # Fix vertex 0 to color 1
    result = list_dsatur(G, L, fixed=fixed)
    assert not result.success
    assert result.failed_vertex == 1  # Vertex 1 can't be colored


def test_dsatur_isolated():
    """Isolated vertex."""
    G = nx.Graph()
    G.add_node(0)
    L = {0: {1, 2}}
    result = list_dsatur(G, L)
    assert result.success
    assert result.coloring[0] in {1, 2}


def test_dsatur_empty():
    """Empty graph."""
    G = nx.Graph()
    L = {}
    result = list_dsatur(G, L)
    assert result.success
    assert result.coloring == {}


def test_dsatur_list_size_one():
    """Vertices with |L(v)|=1 that don't conflict."""
    G = nx.path_graph(3)
    L = {0: {1}, 1: {2}, 2: {1}}
    result = list_dsatur(G, L)
    assert result.success
    validity = check_validity(G, L, result.coloring)
    assert validity.is_valid


def test_dsatur_tiebreak_deterministic():
    """DSATUR should be deterministic with same inputs."""
    G = nx.path_graph(5)
    L = {i: {1, 2, 3} for i in range(5)}
    result1 = list_dsatur(G, L)
    result2 = list_dsatur(G, L)
    assert result1.coloring == result2.coloring


def test_dsatur_k_n_infeasible():
    """Complete graph K_n with n-1 colors - known infeasible."""
    for n in [3, 4, 5]:
        G = nx.complete_graph(n)
        L = {i: {1, 2} for i in range(n)}  # Only 2 colors for n>2
        result = list_dsatur(G, L)
        assert not result.success, f"K_{n} with 2 colors should be infeasible"


def test_dsatur_disconnected():
    """Disconnected graph."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (2, 3)])
    L = {0: {1, 2}, 1: {1, 2}, 2: {1, 2}, 3: {1, 2}}
    result = list_dsatur(G, L)
    assert result.success
    validity = check_validity(G, L, result.coloring)
    assert validity.is_valid


def test_dsatur_property_based():
    """Property-based test: 100 random small graphs, independently verified."""
    random.seed(42)
    for _ in range(100):
        n = random.randint(2, 10)
        p = random.uniform(0.1, 0.5)
        G = nx.erdos_renyi_graph(n, p, seed=random.randint(0, 10000))
        G = nx.convert_node_labels_to_integers(G)

        # Assign lists with slack
        max_deg = max((d for _, d in G.degree()), default=1)
        k = max_deg + 3
        L = {}
        for v in G.nodes():
            deg = G.degree(v)
            list_size = random.randint(deg, deg + 3)
            colors = random.sample(range(1, k + 1), min(list_size, k))
            L[v] = set(colors)

        result = list_dsatur(G, L)
        if result.success:
            validity = check_validity(G, L, result.coloring)
            assert validity.is_valid, f"Invalid coloring on graph {G.edges()}"
        else:
            # Verify it's actually infeasible by checking if any coloring exists
            # For now, just check that failed_vertex is set
            assert result.failed_vertex is not None


def test_dsatur_star():
    """Star graph."""
    G = nx.star_graph(5)  # center 0, leaves 1-5
    L = {0: {1, 2, 3, 4}, 1: {1, 2}, 2: {1, 2}, 3: {1, 2}, 4: {1, 2}, 5: {1, 2}}
    result = list_dsatur(G, L)
    assert result.success
    validity = check_validity(G, L, result.coloring)
    assert validity.is_valid


def test_dsatur_cycle():
    """Even cycle (2-colorable) and odd cycle (3-colorable)."""
    # Even cycle
    G = nx.cycle_graph(4)
    L = {i: {1, 2} for i in range(4)}
    result = list_dsatur(G, L)
    assert result.success
    validity = check_validity(G, L, result.coloring)
    assert validity.is_valid

    # Odd cycle - needs 3 colors
    G = nx.cycle_graph(5)
    L = {i: {1, 2, 3} for i in range(5)}
    result = list_dsatur(G, L)
    assert result.success
    validity = check_validity(G, L, result.coloring)
    assert validity.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])