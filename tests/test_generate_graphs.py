import pytest
import networkx as nx
import os
import tempfile
import json
from data_gen.generate_graphs import generate_graph, assign_lists, generate_dataset


def test_generate_graph_basic():
    """Basic graph generation."""
    graph = generate_graph(10, 5.0, 42)
    assert graph.n == 10
    assert graph.seed == 42
    assert graph.graph_id == "n10_s0042"
    assert len(graph.edges) > 0
    assert graph.avg_degree_realized > 0


def test_generate_graph_deterministic():
    """Same seed should produce same graph."""
    g1 = generate_graph(20, 4.0, 123)
    g2 = generate_graph(20, 4.0, 123)
    assert g1.edges == g2.edges
    assert g1.avg_degree_realized == g2.avg_degree_realized


def test_generate_graph_different_seeds():
    """Different seeds should produce different graphs."""
    g1 = generate_graph(20, 4.0, 123)
    g2 = generate_graph(20, 4.0, 456)
    assert g1.edges != g2.edges


def test_assign_lists():
    """List assignment."""
    G = nx.path_graph(5)
    lists, k = assign_lists(G, 42)
    assert len(lists) == 5
    assert k > 0
    for v, lst in lists.items():
        assert len(lst) >= G.degree(v)
        assert len(lst) <= G.degree(v) + 3
        assert all(1 <= c <= k for c in lst)


def test_assign_lists_deterministic():
    """Same seed produces same lists."""
    G = nx.path_graph(5)
    lists1, k1 = assign_lists(G, 42)
    lists2, k2 = assign_lists(G, 42)
    assert lists1 == lists2
    assert k1 == k2


def test_generate_dataset():
    """Full dataset generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_dataset([10, 20], 2, 0, tmpdir)
        files = os.listdir(tmpdir)
        assert len(files) == 4  # 2 sizes * 2 graphs per size

        for f in files:
            path = os.path.join(tmpdir, f)
            with open(path) as fp:
                data = json.load(fp)
            assert "graph_id" in data
            assert "edges" in data
            assert "lists" in data
            assert "k" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])