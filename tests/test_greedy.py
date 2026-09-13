import pytest
import networkx as nx
from baselines.greedy import list_greedy
from baselines.validity import check_validity
from common.types import ColoringResult


def test_greedy_simple():
    """Simple graph - 3 vertices in a line."""
    G = nx.path_graph(3)
    L = {0: {1, 2}, 1: {1, 2}, 2: {1, 2}}
    result = list_greedy(G, L)
    assert result.success
    validity = check_validity(G, L, result.coloring)
    assert validity.is_valid


def test_greedy_complete_graph():
    """Complete graph K3 with 3 colors."""
    G = nx.complete_graph(3)
    L = {0: {1, 2, 3}, 1: {1, 2, 3}, 2: {1, 2, 3}}
    result = list_greedy(G, L)
    assert result.success
    validity = check_validity(G, L, result.coloring)
    assert validity.is_valid


def test_greedy_infeasible():
    """Infeasible: K3 with only 2 colors."""
    G = nx.complete_graph(3)
    L = {0: {1, 2}, 1: {1, 2}, 2: {1, 2}}
    result = list_greedy(G, L)
    assert not result.success
    assert result.failed_vertex is not None


def test_greedy_order():
    """Test that order parameter works."""
    G = nx.path_graph(3)
    L = {0: {1}, 1: {1, 2}, 2: {1, 2}}
    # If we color 0 first with 1, then 1 must use 2, then 2 can use 1
    result = list_greedy(G, L, order=[0, 1, 2])
    assert result.success
    assert result.coloring[0] == 1
    assert result.coloring[1] == 2
    assert result.coloring[2] == 1


def test_greedy_isolated():
    """Isolated vertex."""
    G = nx.Graph()
    G.add_node(0)
    L = {0: {1, 2}}
    result = list_greedy(G, L)
    assert result.success
    assert result.coloring[0] in {1, 2}


def test_greedy_empty():
    """Empty graph."""
    G = nx.Graph()
    L = {}
    result = list_greedy(G, L)
    assert result.success
    assert result.coloring == {}


def test_greedy_star():
    """Star graph - center connected to leaves."""
    G = nx.star_graph(4)  # center 0, leaves 1-4
    L = {0: {1, 2, 3}, 1: {1, 2}, 2: {1, 2}, 3: {1, 2}, 4: {1, 2}}
    result = list_greedy(G, L)
    assert result.success
    validity = check_validity(G, L, result.coloring)
    assert validity.is_valid


def test_greedy_returns_failure_object():
    """Ensure failure returns ColoringResult with failed_vertex, not None."""
    G = nx.complete_graph(3)
    L = {0: {1}, 1: {1}, 2: {1}}
    result = list_greedy(G, L)
    assert isinstance(result, ColoringResult)
    assert result.success is False
    assert result.failed_vertex is not None
    assert isinstance(result.coloring, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])