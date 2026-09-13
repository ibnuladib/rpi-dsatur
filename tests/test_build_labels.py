import pytest
import networkx as nx
from common.types import Edit, EditType
from labels.build_labels import repair_radius, bucketize, BUCKET_EDGES, BUCKET_NAMES


def test_repair_radius_no_change():
    """No color changes -> radius 0."""
    G = nx.path_graph(3)
    c_prev = {0: 1, 1: 2, 2: 1}
    c_t = {0: 1, 1: 2, 2: 1}
    edit = Edit(index=0, type=EditType.SHRINK_LIST, vertex=0, removed_colors=[2])
    r = repair_radius(G, c_prev, c_t, edit)
    assert r == 0


def test_repair_radius_one_hop():
    """Non-edit vertex changes at distance 1 from edit."""
    G = nx.path_graph(3)
    c_prev = {0: 1, 1: 2, 2: 1}
    # Edit at vertex 0; vertex 1 (neighbor) changes color due to propagation.
    c_t = {0: 1, 1: 3, 2: 1}
    edit = Edit(index=0, type=EditType.SHRINK_LIST, vertex=0, removed_colors=[2])
    r = repair_radius(G, c_prev, c_t, edit)
    assert r == 1


def test_repair_radius_two_hop():
    """Non-edit vertex changes at distance 2 from edit."""
    G = nx.path_graph(4)
    c_prev = {0: 1, 1: 2, 2: 1, 3: 2}
    # Edit at vertex 0; vertex 2 (distance 2) changes color due to propagation.
    c_t = {0: 1, 1: 2, 2: 3, 3: 2}
    edit = Edit(index=0, type=EditType.SHRINK_LIST, vertex=0, removed_colors=[2])
    r = repair_radius(G, c_prev, c_t, edit)
    assert r == 2


def test_repair_radius_remove_vertex():
    """Remove vertex - distance from former neighbors."""
    G = nx.path_graph(4)
    c_prev = {0: 1, 1: 2, 2: 1, 3: 2}
    c_t = {0: 1, 1: 3, 2: 1, 3: 2}  # Vertex 1 changed after vertex 2 removed
    edit = Edit(index=0, type=EditType.REMOVE_VERTEX, vertex=2)
    r = repair_radius(G, c_prev, c_t, edit)
    # Former neighbors of 2 are 1 and 3. Vertex 1 changed, distance from 1 to 1 is 0.
    # Wait, BFS from former neighbors (1 and 3). Vertex 1 is a source with distance 0.
    # So max distance among changed vertices {1} is 0.
    assert r == 0  # The changed vertex is one of the sources


def test_repair_radius_remove_vertex_two_hop():
    """Remove vertex with change at distance 1 from former neighbor."""
    G = nx.path_graph(5)  # 0-1-2-3-4
    c_prev = {0: 1, 1: 2, 2: 1, 3: 2, 4: 1}
    # Remove vertex 2. Former neighbors: 1 and 3.
    # Suppose vertex 0 changes color (distance 1 from neighbor 1)
    c_t = {0: 2, 1: 2, 3: 2, 4: 1}  # Vertex 0 changed
    edit = Edit(index=0, type=EditType.REMOVE_VERTEX, vertex=2)
    r = repair_radius(G, c_prev, c_t, edit)
    # BFS from {1, 3}. Distance to 0: dist(1,0)=1. Max = 1.
    assert r == 1


def test_bucketize():
    """Test bucket assignment."""
    assert bucketize(0) == 0  # bucket "0"
    assert bucketize(1) == 1  # bucket "1"
    assert bucketize(2) == 2  # bucket "2"
    assert bucketize(3) == 3  # bucket "3"
    assert bucketize(4) == 4  # bucket "4"
    assert bucketize(5) == 5  # bucket "full" (ceiling)
    assert bucketize(8) == 5  # bucket "full"
    assert bucketize(100) == 5  # bucket "full"


def test_bucket_names():
    """Bucket names match expected."""
    assert BUCKET_NAMES == ["0", "1", "2", "3", "4", "full"]


def test_bucket_edges():
    """Bucket edges match expected."""
    assert BUCKET_EDGES == [0, 1, 2, 3, 4]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])