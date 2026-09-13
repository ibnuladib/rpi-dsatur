import time
import logging
import torch
import networkx as nx
from typing import Union

from common.types import (
    Edit, EditType, ColoringResult, EditOutcome,
    GraphInstance
)
from baselines import list_dsatur, check_validity
from labels.build_labels import extract_features, BUCKET_NAMES, FEATURE_HOP_RADIUS, apply_edit
from model.radius_gnn import RadiusGNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Radius ladder, DERIVED from labels.build_labels.BUCKET_NAMES rather than
# hardcoded here -- if the bucket scheme ever changes again (as it already
# has once, geometric -> linear), this file updates automatically instead
# of silently drifting out of sync the way it did before.
RADIUS_LADDER = [int(n) if n != "full" else "full" for n in BUCKET_NAMES]
BUCKET_TO_RADIUS = {i: r for i, r in enumerate(RADIUS_LADDER)}


def bucket_to_radius(bucket: Union[int, str]) -> Union[int, str]:
    """Map bucket index to radius value."""
    if isinstance(bucket, str):
        if bucket == "full":
            return "full"
        bucket = int(bucket)
    return BUCKET_TO_RADIUS.get(bucket, "full")


def next_bucket_up(bucket: Union[int, str]) -> Union[int, str]:
    """Move to next bucket in ladder."""
    if isinstance(bucket, str):
        if bucket == "full":
            return "full"
        bucket = int(bucket)

    idx = RADIUS_LADDER.index(bucket) if bucket in RADIUS_LADDER else 0
    if idx + 1 < len(RADIUS_LADDER):
        return RADIUS_LADDER[idx + 1]
    return "full"


def subgraph_within_radius(G: nx.Graph, center: int, radius: Union[int, str]) -> nx.Graph:
    """Extract subgraph within radius of center.

    For radius="full", return entire graph.
    For remove_vertex case, center is the former vertex's neighbor (handled by caller).
    """
    if radius == "full":
        return G.copy()

    if center not in G:
        return nx.Graph()

    lengths = nx.single_source_shortest_path_length(G, center, cutoff=radius)
    nodes = set(lengths.keys())

    if not nodes:
        return nx.Graph()

    return G.subgraph(nodes).copy()


def get_center_for_subgraph(edit: Edit, G_prev: nx.Graph) -> list[int]:
    """Get center node(s) for subgraph extraction.

    For shrink_list: the edited vertex.
    For remove_vertex: former neighbors of the removed vertex.
    """
    if edit.type == EditType.REMOVE_VERTEX:
        return list(G_prev.neighbors(edit.vertex))
    return [edit.vertex]


def rpi_dsatur_step(
    G_prev: nx.Graph,
    L_prev: dict[int, set[int]],
    c_prev: dict[int, int],
    edit: Edit,
    model: RadiusGNN,
    max_retries: int = 3,
    device: str = "cpu",
) -> tuple[ColoringResult, EditOutcome]:
    """RPI-DSATUR inference step - implements Section 7 pseudocode exactly."""
    start_time = time.perf_counter()

    G_t, L_t = apply_edit(G_prev, L_prev, edit)

    # IMPORTANT: hop_radius here MUST match FEATURE_HOP_RADIUS from
    # labels/build_labels.py -- that's the window size the model was
    # actually TRAINED on. Previously this was hardcoded to 3, silently
    # mismatched against whatever hop_radius was actually used for training
    # (4, at time of writing) -- feeding the model out-of-distribution
    # window sizes at inference. Importing the constant instead of
    # hardcoding it means this can never drift out of sync again.
    data = extract_features(G_prev, L_prev, c_prev, edit, hop_radius=FEATURE_HOP_RADIUS)
    if data is None:
        # Feature extraction failed -- fall back to a moderate radius
        bucket_pred = len(RADIUS_LADDER) // 2
    else:
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([data]).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        bucket_pred = int(logits.argmax(dim=1).item())

    r_hat = bucket_to_radius(bucket_pred)

    attempts = 0
    fallback_triggered = False

    while attempts < max_retries:
        centers = get_center_for_subgraph(edit, G_prev)

        if len(centers) == 1:
            S = subgraph_within_radius(G_t, centers[0], r_hat)
        else:
            S = nx.Graph()
            for c in centers:
                S = nx.compose(S, subgraph_within_radius(G_t, c, r_hat))

        boundary_fixed = {}
        for u in S.nodes():
            for v in G_t.neighbors(u):
                if v not in S and v in c_prev:
                    boundary_fixed[v] = c_prev[v]

        L_S = {v: L_t[v] for v in S.nodes() if v in L_t}

        c_S_result = list_dsatur(S, L_S, fixed=boundary_fixed)

        if not c_S_result.success:
            r_hat = next_bucket_up(r_hat)
            attempts += 1
            continue

        c_S = c_S_result.coloring

        c_t = c_prev.copy()
        c_t.update(c_S)

        validity = check_validity(G_t, L_t, c_t)
        if validity.is_valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            vertices_touched = sum(1 for v in c_t if v in c_prev and c_t[v] != c_prev[v])

            outcome = EditOutcome(
                system="rpi_dsatur",
                graph_id="",
                edit_index=edit.index,
                time_ms=elapsed_ms,
                vertices_touched=vertices_touched,
                radius_used=r_hat if r_hat != "full" else "full_recompute",
                attempts=attempts,
                fallback_triggered=fallback_triggered,
                valid=True,
            )
            return ColoringResult(success=True, coloring=c_t), outcome

        r_hat = next_bucket_up(r_hat)
        attempts += 1

    fallback_triggered = True
    result = list_dsatur(G_t, L_t)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    vertices_touched = 0
    if result.success:
        vertices_touched = sum(1 for v in result.coloring if v in c_prev and result.coloring[v] != c_prev[v])

    outcome = EditOutcome(
        system="rpi_dsatur",
        graph_id="",
        edit_index=edit.index,
        time_ms=elapsed_ms,
        vertices_touched=vertices_touched,
        radius_used="full_recompute",
        attempts=attempts,
        fallback_triggered=True,
        valid=result.success,
    )
    return result, outcome


def fixed_radius_step(
    G_prev: nx.Graph,
    L_prev: dict[int, set[int]],
    c_prev: dict[int, int],
    edit: Edit,
    radius: int = 2,
    max_retries: int = 3,
) -> tuple[ColoringResult, EditOutcome]:
    """Fixed-radius baseline - identical control flow but with constant radius."""
    start_time = time.perf_counter()

    G_t, L_t = apply_edit(G_prev, L_prev, edit)

    r_hat = radius
    attempts = 0
    fallback_triggered = False

    while attempts < max_retries:
        centers = get_center_for_subgraph(edit, G_prev)

        if len(centers) == 1:
            S = subgraph_within_radius(G_t, centers[0], r_hat)
        else:
            S = nx.Graph()
            for c in centers:
                S = nx.compose(S, subgraph_within_radius(G_t, c, r_hat))

        boundary_fixed = {}
        for u in S.nodes():
            for v in G_t.neighbors(u):
                if v not in S and v in c_prev:
                    boundary_fixed[v] = c_prev[v]

        L_S = {v: L_t[v] for v in S.nodes() if v in L_t}

        c_S_result = list_dsatur(S, L_S, fixed=boundary_fixed)

        if not c_S_result.success:
            r_hat = next_bucket_up(r_hat)
            attempts += 1
            continue

        c_S = c_S_result.coloring
        c_t = c_prev.copy()
        c_t.update(c_S)

        validity = check_validity(G_t, L_t, c_t)
        if validity.is_valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            vertices_touched = sum(1 for v in c_t if v in c_prev and c_t[v] != c_prev[v])

            outcome = EditOutcome(
                system="fixed_radius",
                graph_id="",
                edit_index=edit.index,
                time_ms=elapsed_ms,
                vertices_touched=vertices_touched,
                radius_used=r_hat if r_hat != "full" else "full_recompute",
                attempts=attempts,
                fallback_triggered=fallback_triggered,
                valid=True,
            )
            return ColoringResult(success=True, coloring=c_t), outcome

        r_hat = next_bucket_up(r_hat)
        attempts += 1

    fallback_triggered = True
    result = list_dsatur(G_t, L_t)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    vertices_touched = 0
    if result.success:
        vertices_touched = sum(1 for v in result.coloring if v in c_prev and result.coloring[v] != c_prev[v])

    outcome = EditOutcome(
        system="fixed_radius",
        graph_id="",
        edit_index=edit.index,
        time_ms=elapsed_ms,
        vertices_touched=vertices_touched,
        radius_used="full_recompute",
        attempts=attempts,
        fallback_triggered=True,
        valid=result.success,
    )
    return result, outcome


def full_recompute_step(
    G_prev: nx.Graph,
    L_prev: dict[int, set[int]],
    c_prev: dict[int, int],
    edit: Edit,
) -> tuple[ColoringResult, EditOutcome]:
    """Full recompute baseline - run DSATUR on entire graph from scratch."""
    start_time = time.perf_counter()

    G_t, L_t = apply_edit(G_prev, L_prev, edit)

    result = list_dsatur(G_t, L_t)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    vertices_touched = 0
    if result.success:
        vertices_touched = sum(1 for v in result.coloring if v in c_prev and result.coloring[v] != c_prev[v])

    validity = check_validity(G_t, L_t, result.coloring) if result.success else None

    outcome = EditOutcome(
        system="full_recompute",
        graph_id="",
        edit_index=edit.index,
        time_ms=elapsed_ms,
        vertices_touched=vertices_touched,
        radius_used="full_recompute",
        attempts=0,
        fallback_triggered=False,
        valid=validity.is_valid if validity else False,
    )
    return result, outcome