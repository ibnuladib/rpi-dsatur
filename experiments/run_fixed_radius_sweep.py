"""Fixed-radius sweep at r=1,3,4 on the held-out test graphs.

Reuses experiments.run_matrix.run_system / load helpers and
inference.rpi_dsatur.fixed_radius_step (via run_system). Does not re-run r=2
(those rows already exist in results/matrix_static_systems.jsonl).
"""

from __future__ import annotations

import json
import logging
import os
import sys

# Allow `python experiments/run_fixed_radius_sweep.py` from repo root.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiments.run_matrix import (  # noqa: E402
    load_edits,
    load_graph,
    load_held_out_test_graph_ids,
    run_system,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Actual holdout artifact lives under hop_shell/ (see model/train.py output).
DEFAULT_TEST_GRAPH_IDS = "model/reports/hop_shell/test_graph_ids.json"


def run_fixed_radius_sweep(
    radii: list[int],
    graphs_dir: str = "data/raw/graphs",
    edits_dir: str = "data/raw/edits",
    test_graph_ids_path: str = DEFAULT_TEST_GRAPH_IDS,
    out_path: str = "results/fixed_radius_sweep.jsonl",
) -> None:
    """
    For each radius in radii, for every held-out test graph (filtered via
    test_graph_ids_path, same restriction as run_matrix.py), run
    fixed_radius_step over its full edit stream. Write one EditOutcome per
    (radius, graph, edit) row to out_path as JSONL, with an added "radius"
    field so rows across different sweep values can be distinguished
    (system field should read "fixed_radius" for all rows; disambiguate by
    the added "radius" column, not by overloading "system").
    """
    test_graph_ids = load_held_out_test_graph_ids(test_graph_ids_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    graph_files = sorted(f for f in os.listdir(graphs_dir) if f.endswith(".json"))
    held_out = []
    for gf in graph_files:
        gpath = os.path.join(graphs_dir, gf)
        graph = load_graph(gpath)
        if graph.graph_id in test_graph_ids:
            held_out.append((gf, graph))

    logger.info(
        "Fixed-radius sweep radii=%s over %d held-out graphs -> %s",
        radii,
        len(held_out),
        out_path,
    )
    if not held_out:
        raise RuntimeError(
            f"No held-out graphs found under {graphs_dir} matching {test_graph_ids_path}"
        )

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for radius in radii:
            logger.info("=== radius=%d ===", radius)
            for gf, graph in held_out:
                edit_path = os.path.join(edits_dir, gf.replace(".json", "_edits.json"))
                if not os.path.exists(edit_path):
                    logger.warning("Missing edits for %s -- skipping", graph.graph_id)
                    continue
                edits = load_edits(edit_path)
                outcomes = run_system(
                    "fixed_radius",
                    graph,
                    edits,
                    model=None,
                    device="cpu",
                    fixed_radius=radius,
                )
                for o in outcomes:
                    o["seed"] = 0
                    o["size"] = graph.n
                    o["radius"] = radius
                    out_f.write(json.dumps(o) + "\n")
                    n_written += 1
                logger.info(
                    "  r=%d %s: %d edits", radius, graph.graph_id, len(outcomes)
                )

    logger.info("Wrote %d rows to %s", n_written, out_path)


if __name__ == "__main__":
    run_fixed_radius_sweep(radii=[1, 3, 4])
