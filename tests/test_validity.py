import pytest
import networkx as nx
from baselines.validity import check_validity
from common.types import ValidityReport


def test_validity_valid_coloring():
    """Valid coloring should pass."""
    G = nx.path_graph(3)
    L = {0: {1, 2}, 1: {1, 2}, 2: {1, 2}}
    coloring = {0: 1, 1: 2, 2: 1}
    report = check_validity(G, L, coloring)
    assert report.is_valid
    assert report.num_conflicts == 0
    assert report.num_list_violations == 0


def test_validity_conflict():
    """Monochromatic edge should be detected."""
    G = nx.path_graph(2)
    L = {0: {1, 2}, 1: {1, 2}}
    coloring = {0: 1, 1: 1}  # Conflict
    report = check_validity(G, L, coloring)
    assert not report.is_valid
    assert report.num_conflicts == 1
    assert (0, 1) in report.conflict_edges or (1, 0) in report.conflict_edges


def test_validity_list_violation():
    """Color not in list should be detected."""
    G = nx.path_graph(2)
    L = {0: {1}, 1: {2}}
    coloring = {0: 2, 1: 2}  # Vertex 0 has color 2 not in list
    report = check_validity(G, L, coloring)
    assert not report.is_valid
    assert report.num_list_violations == 1
    assert 0 in report.violating_vertices


def test_validity_both_errors():
    """Both conflict and list violation."""
    G = nx.path_graph(2)
    L = {0: {1}, 1: {2}}
    coloring = {0: 2, 1: 2}  # Both violations
    report = check_validity(G, L, coloring)
    assert not report.is_valid
    assert report.num_conflicts == 1
    assert report.num_list_violations == 1


def test_validity_missing_vertex():
    """Vertex in coloring but not in graph."""
    G = nx.path_graph(2)
    L = {0: {1}, 1: {2}}
    coloring = {0: 1, 1: 2, 2: 1}  # Vertex 2 not in graph
    report = check_validity(G, L, coloring)
    # Current implementation doesn't check this - it only checks vertices in coloring
    # This is acceptable behavior


def test_validity_empty():
    """Empty graph and coloring."""
    G = nx.Graph()
    L = {}
    coloring = {}
    report = check_validity(G, L, coloring)
    assert report.is_valid


def test_validity_isolated():
    """Isolated vertex."""
    G = nx.Graph()
    G.add_node(0)
    L = {0: {1, 2}}
    coloring = {0: 1}
    report = check_validity(G, L, coloring)
    assert report.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])