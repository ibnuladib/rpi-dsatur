"""
Label generation for the RPI-DSATUR radius predictor (methodology Section 5,
Appendix A.4).

BUCKET SCHEME -- DELIBERATE DEVIATION FROM THE METHODOLOGY DOC, DOCUMENTED HERE:
The methodology's Section 5 specifies buckets {0,1,2,4,8,full} (geometric),
calibrated for graphs where r* can plausibly exceed 8. Empirically, on our
actual ER(n, p=7/(n-1)) graphs at n in {100,200,500}, r* maxes out around 5
(bounded by graph diameter at these densities), so the geometric scheme wastes
two of its six classes ("4" and "8") on radii that almost never occur while
under-resolving the common small-radius cases. We use a LINEAR scheme instead:
{0,1,2,3,4,full}. "full" still means "treat as a whole-graph recompute" at
inference time -- that semantic is unchanged.

IMPORTANT: inference/rpi_dsatur.py's retry ladder (bucket_to_radius /
next_bucket_up) MUST be updated to match this exact ladder:
    0 -> 1 -> 2 -> 3 -> 4 -> full
NOT the methodology's original 0 -> 1 -> 2 -> 4 -> 8 -> full. If that file
still uses the old ladder, predicted bucket "3" will be silently
misinterpreted. Do not build/run Section 7 against this label set until that
file is updated to match.

HOP RADIUS -- CALIBRATED EMPIRICALLY:
FEATURE_HOP_RADIUS was tested at 3, 4, 5, and 8. hop=8 was found to nearly
saturate the whole graph on n=100 instances (windows of 93-100 nodes out of
100), giving no additional signal over hop=5 there, while wasting compute.
hop=4/5 empirically gave the best results and is what's set below. This may
need revisiting per graph size in a later pass (n=500 graphs likely still
have real headroom above hop=5, unlike n=100).

GLOBAL FEATURES -- ADDED to attack the two classes that widening the hop
window alone did NOT fix ("1" and "full"). Diagnosis: "full" cases are
hypothesized to depend on GLOBAL structural criticality of the edited vertex
(how central/bottleneck-like it is to the whole graph), which a bounded local
window cannot see regardless of radius. These are stored as a SEPARATE
per-example tensor (data.global_features), not folded into the per-node `x`
matrix, since they are graph-level, not per-node, quantities -- the model
concatenates them in after pooling (see model/radius_gnn.py).
"""

import os
import json
import logging
from collections import deque

import networkx as nx
import torch

from common.types import Edit, EditType
from baselines import list_dsatur

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LINEAR 6-class scheme {0,1,2,3,4,full}. Classes "1" and "full" are rare
# but KEPT as training targets: global features (this file) + minority
# oversampling (model/train.py) are the current attack on those two
# classes. Do not drop them here -- that hides the metric we are trying
# to move.
#
# Ceiling assignment, rounding UP (never down):
#   r*=0     -> "0"
#   r*=1     -> "1"
#   r*=2     -> "2"
#   r*=3     -> "3"
#   r*=4     -> "4"
#   r*>=5    -> "full"
BUCKET_EDGES = [0, 1, 2, 3, 4]
BUCKET_NAMES = ["0", "1", "2", "3", "4", "full"]
NUM_BUCKETS = len(BUCKET_NAMES)

# Feature-extraction hop radius -- empirically calibrated, see module
# docstring. Explicit module-level constant so it is never silently
# defaulted deep inside a function signature. If you change it, you MUST
# regenerate labels AND update model/config.yaml's num_layers to match (a
# GNN needs at least as many message-passing layers as hops it should see).
FEATURE_HOP_RADIUS = 4

# Number of GNN message-passing layers this hop radius requires, at minimum.
# Cross-checked in build_label_dataset() and worth keeping in sync with
# model/config.yaml's model.num_layers.
MIN_REQUIRED_GNN_LAYERS = FEATURE_HOP_RADIUS

# Number of scalar global-graph features concatenated onto each example.
# Structural (4) + post-edit slack (6) = 10. Optional +5 shell tightness
# means (see include_shell_density) for the §18 fallback experiment.
NUM_STRUCTURAL_FEATURES = 4
NUM_SLACK_FEATURES = 6
NUM_SHELL_FEATURES = 5  # hop shells 0..4
NUM_GLOBAL_FEATURES = NUM_STRUCTURAL_FEATURES + NUM_SLACK_FEATURES  # 10


def repair_radius(
    G_prev: nx.Graph,
    c_prev: dict[int, int],
    c_t: dict[int, int],
    edit: Edit,
) -> int:
    """Compute repair radius r* via BFS from the edited vertex.

    r* = max_{v: c_t(v) != c_prev(v)} dist_{G_prev}(v_edit, v)
    For remove_vertex, BFS sources are the removed vertex's former neighbors
    in G_prev, and distances are computed in G_prev with that vertex removed.
    """
    changed = {v for v in c_t if v in c_prev and c_t[v] != c_prev[v]}
    if not changed:
        return 0

    if edit.type == EditType.REMOVE_VERTEX:
        sources = list(G_prev.neighbors(edit.vertex))
    else:
        sources = [edit.vertex]

    if not sources:
        return 0

    G_bfs = G_prev.copy()
    if edit.type == EditType.REMOVE_VERTEX:
        G_bfs.remove_node(edit.vertex)

    # Multi-source BFS
    distances = {s: 0 for s in sources}
    queue = deque(sources)
    while queue:
        u = queue.popleft()
        d = distances[u]
        for v in G_bfs.neighbors(u):
            if v not in distances:
                distances[v] = d + 1
                queue.append(v)

    max_changed_dist = 0
    for v in changed:
        if v in distances:
            max_changed_dist = max(max_changed_dist, distances[v])
    return max_changed_dist


def bucketize(r_star: int) -> int:
    """Ceiling assignment into a bucket index, per BUCKET_EDGES/BUCKET_NAMES."""
    for i, edge in enumerate(BUCKET_EDGES):
        if r_star <= edge:
            return i
    return len(BUCKET_EDGES)  # "full" bucket


def compute_global_features(G: nx.Graph, edit_vertex: int) -> list[float]:
    """Whole-graph structural features, independent of hop window size.

    Targets the 'full' bucket specifically: cases where damage spreads
    further than any local window can see are hypothesized to correlate with
    how structurally CRITICAL the edited vertex is to the graph overall, not
    with anything visible in a bounded neighborhood.

    Cost note: betweenness centrality is O(V*E) and too expensive to compute
    exactly per edit across a full dataset. We use O(1)/O(V+E) proxies
    instead: degree centrality and articulation-point membership.

    Returns exactly NUM_GLOBAL_FEATURES values, in this fixed order:
      [degree_centrality, is_articulation_point, avg_degree_norm, density]
    """
    n = G.number_of_nodes()
    if n == 0 or edit_vertex not in G:
        return [0.0] * NUM_STRUCTURAL_FEATURES

    degree_centrality = G.degree(edit_vertex) / max(1, n - 1)

    try:
        artic_points = set(nx.articulation_points(G)) if n > 2 else set()
    except nx.NetworkXError:
        artic_points = set()
    is_articulation = 1.0 if edit_vertex in artic_points else 0.0

    avg_degree = sum(d for _, d in G.degree()) / max(1, n)
    avg_degree_norm = avg_degree / max(1, n)
    density = nx.density(G)

    return [degree_centrality, is_articulation, avg_degree_norm, density]


def compute_shell_conflict_density(x, hop_radius: int = FEATURE_HOP_RADIUS) -> list[float]:
    """Mean node tightness per hop shell d=0..4. Empty shells contribute 0.

    x columns: [:, 2]=tightness, [:, 6]=hop_distance/hop_radius.
    """
    tightness = x[:, 2]
    shell = (x[:, 6] * hop_radius).round().long().clamp(0, NUM_SHELL_FEATURES - 1)
    out = []
    for d in range(NUM_SHELL_FEATURES):
        mask = shell == d
        out.append(float(tightness[mask].mean()) if bool(mask.any()) else 0.0)
    return out


def append_shell_density_to_example(data, hop_radius: int = FEATURE_HOP_RADIUS):
    """Append 5 shell-density dims to an existing hop-4 labeled example."""
    extra = compute_shell_conflict_density(data.x, hop_radius)
    extra_t = torch.tensor(extra, dtype=data.global_features.dtype).view(1, -1)
    data.global_features = torch.cat([data.global_features, extra_t], dim=-1)
    return data


def apply_edit(G: nx.Graph, L: dict[int, set[int]], edit: Edit) -> tuple[nx.Graph, dict[int, set[int]]]:
    """Apply an edit to graph and lists, returning new copies.

    Canonical helper -- inference/rpi_dsatur.py re-exports this. The label
    stream mutates in place for speed; callers that need an isolated post-edit
    snapshot (feature extraction, inference) use this.
    """
    G_new = G.copy()
    L_new = {u: set(lst) for u, lst in L.items()}

    if edit.type == EditType.SHRINK_LIST:
        if edit.vertex in L_new:
            for c in (edit.removed_colors or []):
                L_new[edit.vertex].discard(c)
    elif edit.type == EditType.REMOVE_VERTEX:
        if edit.vertex in G_new:
            G_new.remove_node(edit.vertex)
        L_new.pop(edit.vertex, None)

    return G_new, L_new


def compute_slack_features(
    G_post: nx.Graph,
    L_post: dict[int, set[int]],
    c_prev: dict[int, int],
    v: int,
) -> list[float]:
    """Edit-vertex color-slack features on the POST-edit graph/lists.

    Targets class "1" vs "0": whether the edit leaves a free color at v
    (radius-0 recolor) or forces a neighbor recolor (radius 1).

    Computed only on {v} ∪ N(v) ∪ N(N(v)) -- O(deg(v) * max_deg), no
    whole-graph pass. If v is absent from G_post (remove_vertex), all zeros.

    Returns NUM_SLACK_FEATURES floats, in this fixed order:
      slack_v, slack_v_is_zero, list_size_v,
      n_zero_slack_nbrs, min_nbr_slack, mean_nbr_slack
    """
    zeros = [0.0] * NUM_SLACK_FEATURES
    if v not in G_post:
        return zeros

    L_v = L_post.get(v, set())
    nbrs = list(G_post.neighbors(v))
    deg = len(nbrs)

    used_by_nbrs: set[int] = set()
    for u in nbrs:
        cu = c_prev.get(u)
        if cu is not None:
            used_by_nbrs.add(cu)
    slack_v = sum(1 for c in L_v if c not in used_by_nbrs)

    max_list = len(L_v)
    nbr_slacks: list[int] = []
    for u in nbrs:
        L_u = L_post.get(u, set())
        if len(L_u) > max_list:
            max_list = len(L_u)
        used: set[int] = set()
        for w in G_post.neighbors(u):
            if w == v:
                continue
            cw = c_prev.get(w)
            if cw is not None:
                used.add(cw)
        nbr_slacks.append(sum(1 for c in L_u if c not in used))

    max_list = max(max_list, 1)
    slack_v_is_zero = 1.0 if slack_v == 0 else 0.0
    list_size_v = len(L_v) / max_list
    if deg == 0:
        n_zero_slack_nbrs = 0.0
        min_nbr_slack = 0.0
        mean_nbr_slack = 0.0
    else:
        n_zero_slack_nbrs = sum(1 for s in nbr_slacks if s == 0) / deg
        min_nbr_slack = min(nbr_slacks) / max_list
        mean_nbr_slack = (sum(nbr_slacks) / deg) / max_list

    return [
        float(slack_v),
        slack_v_is_zero,
        list_size_v,
        n_zero_slack_nbrs,
        min_nbr_slack,
        mean_nbr_slack,
    ]


def extract_features(
    G_prev: nx.Graph,
    L_prev: dict[int, set[int]],
    c_prev: dict[int, int],
    edit: Edit,
    hop_radius: int = FEATURE_HOP_RADIUS,
    include_shell_density: bool = False,
):
    """Extract a hop_radius-hop neighborhood window as GNN input features,
    plus a separate whole-graph global-feature vector.

    Returns a PyG Data object with:
      x: per-node features [num_nodes, 7]
         columns: current_color, list_size, tightness, degree_norm,
                  is_edited, edit_type, hop_distance_from_center
      edge_index: [2, num_edges]
      edit_type: 0 for shrink_list, 1 for remove_vertex
      global_features: [1, 10] by default -- [4 structural] + [6 post-edit slack].
          Pass include_shell_density=True to append 5 shell-tightness means ([1, 15]).
          shape [1, k] (not [k]) so PyG's Batch.from_data_list concatenates
          correctly along dim 0 into [batch_size, k] after batching.

    NOTE: hop_radius here is a FEATURE-EXTRACTION window size, unrelated to
    the predicted repair radius r_hat used at inference time (methodology
    Section 6.1). Caller must pass hop_radius explicitly if diverging from
    the module default -- do not rely on an implicit default.
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        logger.warning("torch_geometric not available, returning None")
        return None

    if edit.type == EditType.REMOVE_VERTEX:
        center_nodes = list(G_prev.neighbors(edit.vertex))
    else:
        center_nodes = [edit.vertex]

    # Collect nodes within hop_radius, AND their exact hop distance from the
    # nearest center node -- needed for the hop_distance_from_center feature.
    hop_distance: dict[int, int] = {}
    for center in center_nodes:
        if center not in G_prev:
            continue
        if hop_distance.get(center, hop_radius + 1) > 0:
            hop_distance[center] = 0
        lengths = nx.single_source_shortest_path_length(G_prev, center, cutoff=hop_radius)
        for node, dist in lengths.items():
            if node not in hop_distance or dist < hop_distance[node]:
                hop_distance[node] = dist

    if not hop_distance:
        # Edited vertex had no valid center (e.g. isolated vertex removed
        # with no neighbors) -- return a minimal single-node window.
        hop_distance = {edit.vertex: 0} if edit.vertex in G_prev else {}
        if not hop_distance:
            logger.debug(f"extract_features: empty window for edit on vertex {edit.vertex}")
            return None

    node_list = sorted(hop_distance.keys())
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    edges = []
    for u, v in G_prev.edges():
        if u in node_to_idx and v in node_to_idx:
            edges.append([node_to_idx[u], node_to_idx[v]])
            edges.append([node_to_idx[v], node_to_idx[u]])
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges else torch.empty((2, 0), dtype=torch.long)
    )

    max_degree = max((d for _, d in G_prev.degree()), default=1)
    edit_type_val = 1 if edit.type == EditType.REMOVE_VERTEX else 0

    features = []
    for node in node_list:
        current_color = c_prev.get(node, 0)
        list_size = len(L_prev.get(node, set()))
        degree = G_prev.degree(node)
        tightness = list_size / (degree + 1) if (degree + 1) > 0 else 0.0
        degree_norm = degree / max_degree if max_degree > 0 else 0.0
        is_edited = 1.0 if node == edit.vertex else 0.0
        dist_norm = hop_distance[node] / hop_radius if hop_radius > 0 else 0.0

        features.append([
            float(current_color),
            float(list_size),
            tightness,
            degree_norm,
            is_edited,
            float(edit_type_val),
            dist_norm,
        ])

    x = torch.tensor(features, dtype=torch.float)

    global_feats = compute_global_features(G_prev, edit.vertex)

    # Slack on the POST-edit state. apply_edit copies the whole graph
    # (O(n+m)); shrink_list does not change topology, so overlay L(v) only.
    # remove_vertex drops v from G_post -> all-zero slack (v absent).
    v = edit.vertex
    if edit.type == EditType.REMOVE_VERTEX or v not in G_prev:
        slack_feats = [0.0] * NUM_SLACK_FEATURES
    else:
        L_v_post = set(L_prev.get(v, set()))
        for c in (edit.removed_colors or []):
            L_v_post.discard(c)
        L_post = {v: L_v_post}
        for u in G_prev.neighbors(v):
            if u in L_prev:
                L_post[u] = L_prev[u]
        slack_feats = compute_slack_features(G_prev, L_post, c_prev, v)

    shell_feats = (
        compute_shell_conflict_density(x, hop_radius) if include_shell_density else []
    )
    global_feats = global_feats + slack_feats + shell_feats
    expected = NUM_GLOBAL_FEATURES + (NUM_SHELL_FEATURES if include_shell_density else 0)
    if len(global_feats) != expected:
        raise RuntimeError(
            f"global_features has {len(global_feats)} values, expected {expected}"
        )

    data = Data(
        x=x,
        edge_index=edge_index,
        edit_type=edit_type_val,
        num_nodes=len(node_list),
    )
    # Shape [1, NUM_GLOBAL_FEATURES]: PyG's Batch.from_data_list concatenates
    # per-example tensors along dim 0, so each example needs a leading
    # singleton dim to become [batch_size, NUM_GLOBAL_FEATURES] after batching.
    data.global_features = torch.tensor(global_feats, dtype=torch.float).unsqueeze(0)

    # Metadata populated by the caller (graph_id, edit_index, bucket, r_star).
    data.graph_id = ""
    data.edit_index = 0
    data.bucket = 0
    data.r_star = 0
    return data


def _pack_labeled_data(data, graph_id: str, edit: Edit, r_star: int, bucket: int):
    data.graph_id = graph_id
    data.edit_index = edit.index
    data.bucket = bucket
    data.r_star = r_star
    return data


def label_one_off_edit(
    G_prev: nx.Graph,
    L_prev: dict[int, set[int]],
    c_prev: dict[int, int],
    edit: Edit,
    graph_id: str,
    hop_radius: int,
):
    """Label a one-off edit without mutating the sequential stream state.

    Returns (data, discarded). discarded=True if the post-edit instance is
    infeasible or feature extraction failed.
    """
    G_t, L_t = apply_edit(G_prev, L_prev, edit)
    result = list_dsatur(G_t, L_t)
    if not result.success:
        return None, True
    r_star = repair_radius(G_prev, c_prev, result.coloring, edit)
    bucket = bucketize(r_star)
    data = extract_features(G_prev, L_prev, c_prev, edit, hop_radius=hop_radius)
    if data is None:
        return None, True
    return _pack_labeled_data(data, graph_id, edit, r_star, bucket), False


def build_label_dataset(
    graphs_dir: str,
    edits_dir: str,
    out_path: str,
    hop_radius: int = FEATURE_HOP_RADIUS,
    skip_targeted: bool = False,
) -> None:
    """Build the labeled dataset from generated graphs and edit streams.

    For each (graph, edit) pair, in stream order:
      1. Run list_dsatur on (G_{t-1}, L_{t-1}); this must match the running c_prev.
      2. Apply the edit -> (G_t, L_t).
      3. Run list_dsatur from scratch on (G_t, L_t) -> c_t.
      4. If infeasible, discard this edit and continue the stream from the
         PRE-edit state (do not let one bad edit corrupt the rest of the stream).
      5. Compute r* via BFS in G_{t-1}, bucketize it.
      6. Extract features (per-node window + global features) from the
         PRE-edit state and store, tagged with graph_id / edit_index /
         bucket / r_star.
    """
    if MIN_REQUIRED_GNN_LAYERS != hop_radius:
        logger.warning(
            f"hop_radius={hop_radius} but MIN_REQUIRED_GNN_LAYERS={MIN_REQUIRED_GNN_LAYERS} "
            f"-- if these diverge, update model/config.yaml's num_layers to match "
            f"whichever hop_radius is actually used, or the GNN will not be able "
            f"to see the full extracted window."
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    graph_files = sorted(f for f in os.listdir(graphs_dir) if f.endswith(".json"))
    logger.info(f"Processing {len(graph_files)} graphs (hop_radius={hop_radius}, "
                f"num_global_features={NUM_GLOBAL_FEATURES}, skip_targeted={skip_targeted})...")

    all_data = []
    discard_count = 0
    total_edits = 0
    infeasible_initial = 0
    bucket_counts = {name: 0 for name in BUCKET_NAMES}

    for graph_file in graph_files:
        graph_path = os.path.join(graphs_dir, graph_file)
        edit_path = os.path.join(edits_dir, graph_file.replace(".json", "_edits.json"))

        if not os.path.exists(edit_path):
            logger.warning(f"Edit file not found for {graph_file}, skipping")
            continue

        with open(graph_path, "r") as f:
            graph_data = json.load(f)
        with open(edit_path, "r") as f:
            edit_data = json.load(f)

        graph_id = graph_data["graph_id"]
        edges = [tuple(e) for e in graph_data["edges"]]
        L = {int(k): set(v) for k, v in graph_data["lists"].items()}
        G = nx.Graph(edges)
        # Ensure isolated / list-only vertices from the JSON are present as nodes.
        G.add_nodes_from(L.keys())

        result = list_dsatur(G, L)
        if not result.success:
            infeasible_initial += 1
            logger.warning(f"Initial coloring failed for {graph_id}, skipping this graph")
            continue
        c_prev = result.coloring.copy()
        G0 = G.copy()
        L0 = {v: set(lst) for v, lst in L.items()}
        c0 = c_prev.copy()

        edits = [
            Edit(
                index=e["index"],
                type=EditType(e["type"]),
                vertex=e["vertex"],
                removed_colors=e["removed_colors"],
            )
            for e in edit_data["edits"]
        ]

        for edit in edits:
            total_edits += 1

            G_prev = G.copy()
            L_prev = {v: set(lst) for v, lst in L.items()}
            c_prev_snapshot = c_prev.copy()

            # Apply edit to the running (mutable) G, L.
            if edit.type == EditType.SHRINK_LIST:
                if edit.vertex in L:
                    for c in (edit.removed_colors or []):
                        L[edit.vertex].discard(c)
            elif edit.type == EditType.REMOVE_VERTEX:
                if edit.vertex in G:
                    G.remove_node(edit.vertex)
                L.pop(edit.vertex, None)

            result = list_dsatur(G, L)

            if not result.success:
                # Discard this single edit; REVERT G, L, c_prev to the
                # pre-edit snapshot so the rest of the stream continues
                # cleanly from a known-valid state, per Section 5 step 3.
                discard_count += 1
                G = G_prev
                L = L_prev
                c_prev = c_prev_snapshot
                logger.debug(f"Discarded edit {edit.index} for {graph_id} (infeasible after edit)")
                continue

            c_t = result.coloring

            r_star = repair_radius(G_prev, c_prev_snapshot, c_t, edit)
            bucket = bucketize(r_star)
            bucket_counts[BUCKET_NAMES[bucket]] += 1

            data = extract_features(G_prev, L_prev, c_prev_snapshot, edit, hop_radius=hop_radius)
            if data is not None:
                data.graph_id = graph_id
                data.edit_index = edit.index
                data.bucket = bucket
                data.r_star = r_star
                all_data.append(data)
            else:
                logger.debug(f"extract_features returned None for {graph_id} edit {edit.index}")

            # Advance state for the next edit in this stream.
            c_prev = c_t

        # One-off r*=1 (etc.) extras: do not mutate the sequential stream.
        # Skip for sequential 6-class labels (test n=2007); r1 extras make n=2457.
        targeted_iter = () if skip_targeted else (edit_data.get("targeted_edits") or [])
        for te in targeted_iter:
            total_edits += 1
            origin = te.get("from", "initial")
            targeted = Edit(
                index=te["index"],
                type=EditType(te["type"]),
                vertex=te["vertex"],
                removed_colors=te["removed_colors"],
            )
            if origin == "final":
                data, discarded = label_one_off_edit(
                    G, L, c_prev, targeted, graph_id, hop_radius,
                )
            else:
                data, discarded = label_one_off_edit(
                    G0, L0, c0, targeted, graph_id, hop_radius,
                )
            if discarded or data is None:
                discard_count += 1
                continue
            bucket_counts[BUCKET_NAMES[data.bucket]] += 1
            all_data.append(data)

        if total_edits % 500 == 0:
            logger.info(f"Processed {total_edits} edits, {len(all_data)} labeled, {discard_count} discarded")

    if infeasible_initial:
        logger.warning(f"{infeasible_initial} graphs skipped entirely (infeasible initial coloring)")

    discard_rate = discard_count / total_edits * 100 if total_edits else 0.0
    logger.info(f"Total edits processed: {total_edits}")
    logger.info(f"Discard rate: {discard_rate:.1f}%")
    logger.info(f"Class balance: {bucket_counts}")

    if discard_rate > 5.0:
        logger.warning(
            f"Discard rate {discard_rate:.1f}% exceeds the 5% guideline in the "
            f"Section 5 validation gate -- consider increasing k's slack constant "
            f"in data_gen before trusting this label set."
        )
    min_class_frac = min(bucket_counts.values()) / max(1, sum(bucket_counts.values()))
    if min_class_frac < 0.01:
        logger.warning(
            f"Smallest bucket is only {min_class_frac*100:.2f}% of examples -- "
            f"per the Section 5 gate, decide explicitly whether to accept this "
            f"imbalance (and expect low recall on it) or adjust edit generation "
            f"to produce more of that case."
        )

    torch.save(all_data, out_path)
    logger.info(f"Saved {len(all_data)} labeled examples to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the RPI-DSATUR label dataset.")
    parser.add_argument("--graphs-dir", required=True)
    parser.add_argument("--edits-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--hop-radius", type=int, default=FEATURE_HOP_RADIUS,
        help=f"Feature-extraction hop window (default: {FEATURE_HOP_RADIUS}). "
             f"Must match model.num_layers in model/config.yaml.",
    )
    parser.add_argument(
        "--skip-targeted", action="store_true",
        help="Omit targeted_edits extras (sequential 6-class, test n=2007).",
    )
    args = parser.parse_args()
    build_label_dataset(
        args.graphs_dir, args.edits_dir, args.out,
        hop_radius=args.hop_radius, skip_targeted=args.skip_targeted,
    )