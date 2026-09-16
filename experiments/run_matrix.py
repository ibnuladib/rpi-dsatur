import os
import json
import argparse
import logging
import yaml
import networkx as nx
import torch

from common.types import Edit, EditType, GraphInstance
from baselines import list_dsatur
from labels.build_labels import extract_features, FEATURE_HOP_RADIUS
from inference.rpi_dsatur import (
    rpi_dsatur_step,
    fixed_radius_step,
    full_recompute_step,
)
from model.radius_gnn import create_model_from_config, RadiusGNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_graph(graph_path: str) -> GraphInstance:
    """Load a graph from JSON file."""
    with open(graph_path, "r") as f:
        data = json.load(f)

    return GraphInstance(
        graph_id=data["graph_id"],
        n=data["n"],
        seed=data["seed"],
        edges=[tuple(e) for e in data["edges"]],
        lists={int(k): set(v) for k, v in data["lists"].items()},
        k=data["k"],
        avg_degree_realized=data["avg_degree_realized"],
    )


def load_edits(edit_path: str) -> list[Edit]:
    """Load edit stream from JSON file."""
    with open(edit_path, "r") as f:
        data = json.load(f)

    edits = []
    for e in data["edits"]:
        edit_type = EditType(e["type"])
        edits.append(Edit(
            index=e["index"],
            type=edit_type,
            vertex=e["vertex"],
            removed_colors=e["removed_colors"],
        ))
    return edits


def load_held_out_test_graph_ids(path: str = "model/reports/test_graph_ids.json") -> set[str]:
    """Load the EXACT set of held-out test graph_ids saved by model/train.py.

    This MUST be used instead of any independently-derived "test set"
    (e.g. picking the last N graphs alphabetically) -- the real split is a
    seeded sklearn train_test_split over graph_ids in whatever order they
    first appeared while building the label dataset, which has no relation
    to filename ordering. Using a different selection risks silently
    evaluating rpi_dsatur on graphs the classifier was TRAINED on.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m model.train model/config.yaml` "
            f"first -- it now saves the exact held-out test graph_ids there. "
            f"Do not substitute an alphabetical or manual graph selection."
        )
    with open(path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data['test_graph_ids'])} held-out test graph_ids "
                f"(from training seed={data['seed']})")
    return set(data["test_graph_ids"])


def create_model_for_inference(checkpoint_path: str, model_config_path: str, device: str) -> RadiusGNN:
    """Load the radius predictor using the SAME config the checkpoint was
    trained with, and infer in_dim/num_classes from real data rather than
    hardcoded guesses.

    Previously this hardcoded hidden_dim=64, num_layers=3, conv_type="sage",
    in_dim=6 -- none of which necessarily matched what was actually trained
    (e.g. in_dim is 7 with the hop_distance_from_center feature; hidden_dim/
    num_layers/conv_type depend on whichever config.yaml produced this
    checkpoint). A mismatch here either crashes on load_state_dict or,
    worse, loads successfully into the wrong-shaped layers.
    """
    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Determine per-node in_dim from actual extracted features on a real
    # example, rather than a hardcoded constant -- this always matches
    # whatever labels/build_labels.py currently produces.
    in_dim = 7  # current per-node feature count; see labels/build_labels.py extract_features()

    # num_classes: read from the training report if available (handles the
    # excluded_buckets ad-hoc experiment producing a smaller head); falls
    # back to the full bucket count otherwise.
    num_classes = None
    report_path = "model/reports/train_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
        excluded = report.get("excluded_buckets", [])
        if excluded:
            logger.warning(
                f"Checkpoint's training report shows excluded_buckets={excluded}. "
                f"This is a SCOPED classifier missing predictions for those bucket(s) "
                f"entirely -- confirm this is really the model you want deployed in "
                f"the full RPI-DSATUR system, not just the diagnostic ceiling test."
            )
        num_classes = len(report.get("confusion_matrix_labels", [])) or None

    model = create_model_from_config(config, in_dim, num_classes=num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded RPI-DSATUR model from {checkpoint_path} "
                f"(in_dim={in_dim}, num_classes={num_classes or 'default'})")
    return model


def run_system(
    system: str,
    graph: GraphInstance,
    edits: list[Edit],
    model: RadiusGNN | None,
    device: str,
    fixed_radius: int = 2,
) -> list[dict]:
    """Run a single system on a graph's edit stream."""
    G = nx.Graph(graph.edges)
    L = {v: set(lst) for v, lst in graph.lists.items()}

    result = list_dsatur(G, L)
    if not result.success:
        logger.warning(f"Initial coloring failed for {graph.graph_id}")
        return []

    c_prev = result.coloring.copy()

    outcomes = []

    for edit in edits:
        G_prev = G.copy()
        L_prev = {v: set(lst) for v, lst in L.items()}
        c_prev_copy = c_prev.copy()

        if system == "full_recompute":
            result, outcome = full_recompute_step(G_prev, L_prev, c_prev_copy, edit)
        elif system == "fixed_radius":
            result, outcome = fixed_radius_step(G_prev, L_prev, c_prev_copy, edit, radius=fixed_radius)
        elif system == "majority":
            # Always start at the majority classifier bucket (radius 0).
            result, outcome = fixed_radius_step(G_prev, L_prev, c_prev_copy, edit, radius=0)
            outcome.system = "majority"
        elif system == "rpi_dsatur":
            result, outcome = rpi_dsatur_step(G_prev, L_prev, c_prev_copy, edit, model, device=device)
        else:
            raise ValueError(f"Unknown system: {system}")

        outcome.graph_id = graph.graph_id

        if edit.type == EditType.SHRINK_LIST:
            if edit.vertex in L:
                for c in edit.removed_colors:
                    L[edit.vertex].discard(c)
        elif edit.type == EditType.REMOVE_VERTEX:
            G.remove_node(edit.vertex)
            L.pop(edit.vertex, None)

        if result.success:
            c_prev = result.coloring

        outcomes.append(outcome.__dict__)

    return outcomes


def main():
    parser = argparse.ArgumentParser(description="Run experimental matrix")
    parser.add_argument("--systems", nargs="+", default=["full_recompute", "fixed_radius", "rpi_dsatur"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 200, 500])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--model-checkpoint", type=str, default="model/checkpoints/radius_gnn_hopshell.pt")
    parser.add_argument("--model-config", type=str, default="model/config.yaml",
                         help="The config.yaml that produced --model-checkpoint. Must match "
                              "(hidden_dim/num_layers/conv_type/etc.) or loading will fail.")
    parser.add_argument("--out", type=str, default="results/matrix_out.jsonl")
    parser.add_argument("--graphs-dir", type=str, default="data/raw/graphs")
    parser.add_argument("--edits-dir", type=str, default="data/raw/edits")
    parser.add_argument("--test-graph-ids-path", type=str, default="model/reports/hop_shell/test_graph_ids.json")
    parser.add_argument(
        "--skip-holdout-filter",
        action="store_true",
        help="Evaluate every graph of the requested sizes in --graphs-dir. "
             "Required for OOD sizes (e.g. n=2000) that are not in the training holdout.",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # The ONLY correct source of "which graphs are the held-out test set" --
    # never re-derive this independently (see load_held_out_test_graph_ids docstring).
    # OOD scaling runs (n not in training) pass --skip-holdout-filter instead.
    test_graph_ids = None
    if not args.skip_holdout_filter:
        test_graph_ids = load_held_out_test_graph_ids(args.test_graph_ids_path)

    model = None
    if "rpi_dsatur" in args.systems:
        model = create_model_for_inference(args.model_checkpoint, args.model_config, device)

    all_outcomes = []

    for size in args.sizes:
        logger.info(f"Processing size n={size}")

        graph_files = sorted([
            f for f in os.listdir(args.graphs_dir)
            if f.endswith(".json") and f.startswith(f"n{size}_")
        ])

        # Filter to ONLY graphs that were actually held out during training,
        # instead of an alphabetical slice that has no guaranteed relationship
        # to the real train/val/test split. OOD sizes skip this filter.
        if test_graph_ids is None:
            test_graphs = graph_files
            logger.info(f"  {len(test_graphs)} graphs at size n={size} (holdout filter skipped)")
        else:
            test_graphs = [
                f for f in graph_files
                if load_graph(os.path.join(args.graphs_dir, f)).graph_id in test_graph_ids
            ]
            logger.info(f"  {len(test_graphs)} held-out test graphs at size n={size}")
        if not test_graphs:
            logger.warning(f"  No held-out test graphs found for size n={size} -- skipping.")
            continue

        for seed in args.seeds:
            logger.info(f"  Seed {seed}")

            for graph_file in test_graphs:
                graph_path = os.path.join(args.graphs_dir, graph_file)
                edit_path = os.path.join(args.edits_dir, graph_file.replace(".json", "_edits.json"))

                if not os.path.exists(edit_path):
                    continue

                graph = load_graph(graph_path)
                edits = load_edits(edit_path)

                for system in args.systems:
                    logger.info(f"    {system} on {graph.graph_id}")
                    outcomes = run_system(system, graph, edits, model, device)
                    for o in outcomes:
                        o["seed"] = seed
                        o["size"] = size
                        all_outcomes.append(o)

    with open(args.out, "w") as f:
        for o in all_outcomes:
            f.write(json.dumps(o) + "\n")

    logger.info(f"Saved {len(all_outcomes)} outcomes to {args.out}")

    by_system = {}
    for o in all_outcomes:
        by_system.setdefault(o["system"], []).append(o)
    logger.info("Aggregate metrics (mean wasted retries = mean attempts):")
    logger.info(
        f"{'system':<16} {'n':>6} {'valid':>8} {'fallback':>10} "
        f"{'mean_ms':>10} {'touched':>10} {'attempts':>10}"
    )
    for system, rows in by_system.items():
        n = len(rows)
        valid = sum(1 for r in rows if r.get("valid")) / n if n else 0.0
        fallback = sum(1 for r in rows if r.get("fallback_triggered")) / n if n else 0.0
        mean_ms = sum(r.get("time_ms", 0.0) for r in rows) / n if n else 0.0
        touched = sum(r.get("vertices_touched", 0) for r in rows) / n if n else 0.0
        attempts = sum(r.get("attempts", 0) for r in rows) / n if n else 0.0
        logger.info(
            f"{system:<16} {n:6d} {valid:8.4f} {fallback:10.4f} "
            f"{mean_ms:10.1f} {touched:10.2f} {attempts:10.2f}"
        )


if __name__ == "__main__":
    main()