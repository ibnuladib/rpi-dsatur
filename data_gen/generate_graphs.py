import networkx as nx
import random
import json
import os
from common.types import GraphInstance


def generate_graph(n: int, target_avg_degree: float, seed: int) -> GraphInstance:
    """Generate an Erdos-Renyi graph with specified average degree."""
    random.seed(seed)
    p = target_avg_degree / (n - 1) if n > 1 else 0.0
    G = nx.erdos_renyi_graph(n, p, seed=seed)

    # Relabel nodes to be 0..n-1 (erdos_renyi_graph already does this)
    # But ensure we have exactly n nodes even if isolated
    G = nx.convert_node_labels_to_integers(G, first_label=0)

    edges = list(G.edges())
    avg_degree_realized = 2 * len(edges) / n if n > 0 else 0.0

    graph_id = f"n{n}_s{seed:04d}"

    return GraphInstance(
        graph_id=graph_id,
        n=n,
        seed=seed,
        edges=edges,
        lists={},  # will be filled by assign_lists
        k=0,  # will be filled by assign_lists
        avg_degree_realized=avg_degree_realized,
    )


def assign_lists(G: nx.Graph, seed: int) -> tuple[dict[int, list[int]], int]:
    """Assign color lists to vertices.

    |L(v)| ~ Uniform{deg(v), deg(v)+1, deg(v)+2, deg(v)+3}
    Colors drawn uniformly without replacement from palette {1,...,k}
    where k = max_degree(G) + 5
    """
    random.seed(seed)

    max_degree = max((d for _, d in G.degree()), default=0)
    k = max_degree + 5
    palette = list(range(1, k + 1))

    lists = {}
    for v in G.nodes():
        deg = G.degree(v)
        list_size = random.randint(deg, deg + 3)
        # Sample without replacement from palette
        lists[v] = random.sample(palette, min(list_size, len(palette)))

    return lists, k


def generate_dataset(
    sizes: list[int],
    graphs_per_size: int,
    base_seed: int,
    out_dir: str,
) -> None:
    """Generate dataset and write one JSON file per graph."""
    os.makedirs(out_dir, exist_ok=True)

    for size in sizes:
        # Context.md Section 4.1: average degree must land in [4, 10].
        # Pick a target within that band independent of size.
        target_avg_degree = 7.0
        for i in range(graphs_per_size):
            seed = base_seed + size * 1000 + i
            graph = generate_graph(size, target_avg_degree, seed)
            lists, k = assign_lists(nx.Graph(graph.edges), seed)
            graph.lists = lists
            graph.k = k

            # Write to file
            out_path = os.path.join(out_dir, f"{graph.graph_id}.json")
            data = {
                "graph_id": graph.graph_id,
                "n": graph.n,
                "seed": graph.seed,
                "edges": [list(e) for e in graph.edges],
                "lists": {str(k): v for k, v in graph.lists.items()},
                "k": graph.k,
                "avg_degree_realized": graph.avg_degree_realized,
            }
            with open(out_path, "w") as f:
                json.dump(data, f)

    print(f"Generated {len(sizes) * graphs_per_size} graphs in {out_dir}")