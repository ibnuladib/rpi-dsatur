import pytest
import networkx as nx
from common.types import Edit, EditType
from inference.rpi_dsatur import (
    bucket_to_radius,
    next_bucket_up,
    subgraph_within_radius,
    fixed_radius_step,
    full_recompute_step,
)
from baselines import list_dsatur


def test_bucket_to_radius():
    """Test bucket to radius mapping."""
    assert bucket_to_radius(0) == 0
    assert bucket_to_radius(1) == 1
    assert bucket_to_radius(2) == 2
    assert bucket_to_radius(3) == 3
    assert bucket_to_radius(4) == 4
    assert bucket_to_radius(5) == "full"
    assert bucket_to_radius("full") == "full"


def test_next_bucket_up():
    """Test radius ladder."""
    assert next_bucket_up(0) == 1
    assert next_bucket_up(1) == 2
    assert next_bucket_up(2) == 3
    assert next_bucket_up(3) == 4
    assert next_bucket_up(4) == "full"
    assert next_bucket_up("full") == "full"


def test_subgraph_within_radius():
    """Test subgraph extraction."""
    G = nx.path_graph(5)  # 0-1-2-3-4

    # Radius 0
    S = subgraph_within_radius(G, 2, 0)
    assert set(S.nodes()) == {2}

    # Radius 1
    S = subgraph_within_radius(G, 2, 1)
    assert set(S.nodes()) == {1, 2, 3}

    # Radius 2
    S = subgraph_within_radius(G, 2, 2)
    assert set(S.nodes()) == {0, 1, 2, 3, 4}

    # Full
    S = subgraph_within_radius(G, 2, "full")
    assert set(S.nodes()) == {0, 1, 2, 3, 4}

    # Center not in graph
    S = subgraph_within_radius(G, 10, 1)
    assert S.number_of_nodes() == 0


def test_full_recompute_step():
    """Test full recompute baseline."""
    G = nx.path_graph(4)
    L = {0: {1, 2}, 1: {1, 2, 3}, 2: {1, 2}, 3: {1, 2}}
    c_prev = {0: 1, 1: 2, 2: 1, 3: 2}

    edit = Edit(index=0, type=EditType.SHRINK_LIST, vertex=1, removed_colors=[2])

    result, outcome = full_recompute_step(G, L, c_prev, edit)

    assert result.success
    assert outcome.system == "full_recompute"
    assert outcome.valid
    assert outcome.radius_used == "full_recompute"


def test_fixed_radius_step():
    """Test fixed radius baseline."""
    G = nx.path_graph(4)
    L = {0: {1, 2}, 1: {1, 2, 3}, 2: {1, 2}, 3: {1, 2}}
    c_prev = {0: 1, 1: 2, 2: 1, 3: 2}

    edit = Edit(index=0, type=EditType.SHRINK_LIST, vertex=1, removed_colors=[2])

    result, outcome = fixed_radius_step(G, L, c_prev, edit, radius=1)

    assert result.success
    assert outcome.system == "fixed_radius"
    assert outcome.valid


def test_fixed_radius_fallback():
    """Test fixed radius with fallback."""
    # Create a case where small radius fails
    G = nx.complete_graph(4)
    L = {0: {1, 2, 3}, 1: {1, 2, 3}, 2: {1, 2, 3}, 3: {1, 2, 3}}
    c_prev = {0: 1, 1: 2, 2: 3, 3: 1}  # Valid coloring

    # Shrink list on vertex 0 to only {1} - but 1 is used by neighbor 3
    edit = Edit(index=0, type=EditType.SHRINK_LIST, vertex=0, removed_colors=[2, 3])

    result, outcome = fixed_radius_step(G, L, c_prev, edit, radius=0, max_retries=3)

    # Should fallback to full recompute
    assert outcome.fallback_triggered
    assert outcome.radius_used == "full_recompute"


def test_remove_vertex():
    """Test remove_vertex edit."""
    G = nx.path_graph(4)  # 0-1-2-3
    L = {0: {1, 2}, 1: {1, 2, 3}, 2: {1, 2}, 3: {1, 2}}
    c_prev = {0: 1, 1: 2, 2: 1, 3: 2}

    edit = Edit(index=0, type=EditType.REMOVE_VERTEX, vertex=2)

    result, outcome = full_recompute_step(G, L, c_prev, edit)

    assert result.success
    assert 2 not in result.coloring
    assert outcome.valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])