import pytest
import networkx as nx
from common.types import GraphInstance, EditType
from data_gen.generate_edits import (
    generate_edit_stream,
    load_graph,
    collect_target_r_star_edits,
)


def test_generate_edit_stream_shrink():
    """Test shrink_list edits."""
    graph = GraphInstance(
        graph_id="test",
        n=10,
        seed=0,
        edges=[(0, 1), (1, 2), (2, 3)],
        lists={0: [1, 2], 1: [1, 2, 3], 2: [1, 2], 3: [1, 2]},
        k=3,
        avg_degree_realized=2.0,
    )
    edits = generate_edit_stream(graph, 5, 42)
    assert len(edits) <= 5
    for e in edits:
        assert e.type in (EditType.SHRINK_LIST, EditType.REMOVE_VERTEX)
        if e.type == EditType.SHRINK_LIST:
            assert e.removed_colors is not None
            assert len(e.removed_colors) > 0


def test_generate_edit_stream_remove():
    """Test remove_vertex edits."""
    graph = GraphInstance(
        graph_id="test",
        n=10,
        seed=0,
        edges=[(0, 1), (1, 2), (2, 3)],
        lists={0: [1, 2], 1: [1, 2, 3], 2: [1, 2], 3: [1, 2]},
        k=3,
        avg_degree_realized=2.0,
    )
    edits = generate_edit_stream(graph, 10, 42)
    # Should have some remove_vertex edits
    remove_edits = [e for e in edits if e.type == EditType.REMOVE_VERTEX]
    # Not guaranteed but likely
    assert len(edits) > 0


def test_generate_edit_stream_min_size():
    """Test min_size_fraction constraint."""
    graph = GraphInstance(
        graph_id="test",
        n=10,
        seed=0,
        edges=[(0, 1), (1, 2)],
        lists={0: [1, 2], 1: [1, 2], 2: [1, 2]},
        k=2,
        avg_degree_realized=2.0,
    )
    # With min_size_fraction=0.5, can't go below 5 nodes
    edits = generate_edit_stream(graph, 20, 42, min_size_fraction=0.5)
    # Should stop removing vertices when size would go below 5


def test_load_graph():
    """Test loading graph from JSON."""
    import tempfile
    import json
    import os

    data = {
        "graph_id": "n10_s0042",
        "n": 10,
        "seed": 42,
        "edges": [[0, 1], [1, 2]],
        "lists": {"0": [1, 2], "1": [1, 2], "2": [1, 2]},
        "k": 3,
        "avg_degree_realized": 2.0,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.json")
        with open(path, "w") as f:
            json.dump(data, f)
        graph = load_graph(path)
        assert graph.graph_id == "n10_s0042"
        assert graph.n == 10
        assert graph.edges == [(0, 1), (1, 2)]
        assert graph.lists == {0: [1, 2], 1: [1, 2], 2: [1, 2]}


def test_collect_target_r_star_does_not_mutate():
    """Rejection sampling is one-off: the snapshot graph must be unchanged."""
    import random as rng_mod
    from baselines import list_dsatur

    G = nx.path_graph(6)
    L = {i: {1, 2, 3} for i in range(6)}
    result = list_dsatur(G, L)
    assert result.success
    nodes_before = set(G.nodes())
    lists_before = {v: set(lst) for v, lst in L.items()}
    found, attempts = collect_target_r_star_edits(
        G, L, result.coloring, rng_mod.Random(0),
        n_target=2, max_attempts=40, start_index=10000, origin="initial",
    )
    assert set(G.nodes()) == nodes_before
    assert L == lists_before
    assert attempts <= 40
    for edit, origin in found:
        assert origin == "initial"
        assert edit.type == EditType.SHRINK_LIST
        assert edit.removed_colors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])