"""Hop-window coverage fraction vs graph size n.

Substantiates the claim that a 4-hop window nearly covers the whole graph
at every tested n, with measured coverage_fraction data.
"""

from __future__ import annotations

import logging
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiments.run_matrix import load_graph, load_held_out_test_graph_ids  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TEST_GRAPH_IDS = "model/reports/hop_shell/test_graph_ids.json"


def compute_hop_coverage_fractions(
    graphs_dir: str = "data/raw/graphs",
    test_graph_ids_path: str = DEFAULT_TEST_GRAPH_IDS,
    n2000_graphs_dir: str = "data/raw/graphs_n2000",
    hop_radius: int = 4,
    sample_vertices_per_graph: int = 20,
) -> pd.DataFrame:
    """
    For every held-out test graph (n=100/200/500) AND every n=2000 OOD graph:
      1. Load the graph via networkx (reuse whatever graph-loading helper
         experiments/run_matrix.py already has -- do not reimplement JSON
         parsing).
      2. Sample up to sample_vertices_per_graph vertices uniformly at random
         (fixed seed=0 for reproducibility).
      3. For each sampled vertex, run nx.single_source_shortest_path_length
         with cutoff=hop_radius, count reachable nodes.
      4. coverage_fraction = reachable_count / graph.number_of_nodes()
    Return a long-format DataFrame: columns [graph_id, n, vertex, coverage_fraction].
    """
    rows: list[dict] = []
    rng = np.random.default_rng(0)

    test_graph_ids = load_held_out_test_graph_ids(test_graph_ids_path)
    holdout_files = sorted(
        f for f in os.listdir(graphs_dir) if f.endswith(".json")
    )
    graphs_to_run: list[tuple[str, object]] = []
    for gf in holdout_files:
        gpath = os.path.join(graphs_dir, gf)
        graph = load_graph(gpath)
        if graph.graph_id in test_graph_ids:
            graphs_to_run.append((gpath, graph))

    if os.path.isdir(n2000_graphs_dir):
        for gf in sorted(os.listdir(n2000_graphs_dir)):
            if not gf.endswith(".json"):
                continue
            gpath = os.path.join(n2000_graphs_dir, gf)
            graph = load_graph(gpath)
            graphs_to_run.append((gpath, graph))
    else:
        logger.warning(
            "n2000 graphs dir %s missing -- OOD coverage rows will be absent",
            n2000_graphs_dir,
        )

    logger.info(
        "Computing %d-hop coverage on %d graphs (sample=%d verts/graph)",
        hop_radius,
        len(graphs_to_run),
        sample_vertices_per_graph,
    )

    for gpath, graph in graphs_to_run:
        G = nx.Graph(graph.edges)
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            continue
        nodes = list(G.nodes())
        k = min(sample_vertices_per_graph, len(nodes))
        # Independent draw per graph but shared RNG stream (seed=0 overall).
        sampled = rng.choice(nodes, size=k, replace=False)
        for v in sampled:
            lengths = nx.single_source_shortest_path_length(
                G, int(v), cutoff=hop_radius
            )
            coverage = len(lengths) / float(n_nodes)
            rows.append(
                {
                    "graph_id": graph.graph_id,
                    "n": int(graph.n),
                    "vertex": int(v),
                    "coverage_fraction": float(coverage),
                }
            )
        logger.info(
            "  %s n=%d: mean coverage=%.4f over %d verts",
            graph.graph_id,
            graph.n,
            float(np.mean([r["coverage_fraction"] for r in rows[-k:]])),
            k,
        )

    return pd.DataFrame(rows, columns=["graph_id", "n", "vertex", "coverage_fraction"])


def main(
    out_path: str = "results/hop_coverage.csv",
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df = compute_hop_coverage_fractions()
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
    if not (df["n"] == 2000).any():
        logger.warning(
            "OOD n=2000 coverage could not be computed (no rows with n=2000)."
        )


if __name__ == "__main__":
    main()
