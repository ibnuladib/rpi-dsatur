import json
import logging
import random
import os
import networkx as nx
from common.types import Edit, EditType, GraphInstance
from baselines import list_dsatur
from labels.build_labels import apply_edit, repair_radius

logger = logging.getLogger(__name__)

# Targeted r*=1 extras live in a sidecar list on the same JSON so the
# original sequential stream is unchanged (Section 8 eval protocol).
TARGETED_INDEX_BASE = 10000
TARGET_R_STAR = 1


def generate_edit_stream(
    graph: GraphInstance,
    num_edits: int,
    seed: int,
    min_size_fraction: float = 0.2,
) -> list[Edit]:
    """Generate a stream of edits for a graph.

    60% SHRINK_LIST / 40% REMOVE_VERTEX per Section 4.4 rules.
    Applies edits sequentially to the same evolving graph.

    Rebalanced (per Section 5 gate: every bucket >=5%):
    - REMOVE_VERTEX samples proportional to degree+1 (avoids isolated vertices).
    - SHRINK_LIST removes 1 color w.p. 0.5, 2 colors w.p. 0.5 (was 0.7/0.3) to
      produce more propagation events.
    """
    random.seed(seed)

    G = nx.Graph(graph.edges)
    lists = {v: set(lst) for v, lst in graph.lists.items()}
    original_size = G.number_of_nodes()

    edits = []
    for i in range(num_edits):
        current_size = G.number_of_nodes()
        if current_size == 0:
            break

        # Determine if we can do remove_vertex (need at least min_size_fraction of original)
        can_remove = current_size > max(1, int(original_size * min_size_fraction))

        # 60% shrink_list, 40% remove_vertex (if allowed)
        if can_remove and random.random() < 0.4:
            # REMOVE_VERTEX — bias by degree+1 so isolated vertices are rarely picked
            nodes = list(G.nodes())
            weights = [G.degree(v) + 1 for v in nodes]
            v = random.choices(nodes, weights=weights, k=1)[0]
            G.remove_node(v)
            lists.pop(v, None)
            edits.append(Edit(index=i, type=EditType.REMOVE_VERTEX, vertex=v, removed_colors=None))
        else:
            # SHRINK_LIST
            # Pick v with |L(v)| >= 2
            candidates = [v for v, lst in lists.items() if len(lst) >= 2]
            if not candidates:
                # Fallback: if no candidates, do remove_vertex instead
                if can_remove:
                    nodes = list(G.nodes())
                    weights = [G.degree(v) + 1 for v in nodes]
                    v = random.choices(nodes, weights=weights, k=1)[0]
                    G.remove_node(v)
                    lists.pop(v, None)
                    edits.append(Edit(index=i, type=EditType.REMOVE_VERTEX, vertex=v, removed_colors=None))
                else:
                    # No valid edits possible
                    break
                continue

            v = random.choice(candidates)
            current_list = list(lists[v])
            # Remove 1 color w.p. 0.5, 2 colors w.p. 0.5 (was 0.7/0.3) to bias toward propagation
            num_remove = 1 if random.random() < 0.5 else 2
            num_remove = min(num_remove, len(current_list) - 1)  # keep at least 1 color
            removed = random.sample(current_list, num_remove)
            for c in removed:
                lists[v].remove(c)
            edits.append(Edit(index=i, type=EditType.SHRINK_LIST, vertex=v, removed_colors=removed))

    return edits


def load_graph(graph_path: str) -> GraphInstance:
    """Load a graph from JSON file."""
    with open(graph_path, "r") as f:
        data = json.load(f)

    return GraphInstance(
        graph_id=data["graph_id"],
        n=data["n"],
        seed=data["seed"],
        edges=[tuple(e) for e in data["edges"]],
        lists={int(k): v for k, v in data["lists"].items()},
        k=data["k"],
        avg_degree_realized=data["avg_degree_realized"],
    )


def generate_edits_for_graphs(
    graphs_dir: str,
    num_edits: int,
    seed: int,
    out_dir: str,
) -> None:
    """Generate edit streams for all graphs in a directory."""
    os.makedirs(out_dir, exist_ok=True)

    graph_files = sorted([f for f in os.listdir(graphs_dir) if f.endswith(".json")])

    for i, graph_file in enumerate(graph_files):
        graph_path = os.path.join(graphs_dir, graph_file)
        graph = load_graph(graph_path)
        edit_seed = seed + i * 1000
        edits = generate_edit_stream(graph, num_edits, edit_seed)

        out_path = os.path.join(out_dir, f"{graph.graph_id}_edits.json")
        data = {
            "graph_id": graph.graph_id,
            "edits": [
                {
                    "index": e.index,
                    "type": e.type.value,
                    "vertex": e.vertex,
                    "removed_colors": e.removed_colors,
                }
                for e in edits
            ],
        }
        # Preserve r*=1 one-offs if this stream is being regenerated.
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                prev = json.load(f)
            if prev.get("targeted_edits"):
                data["targeted_edits"] = prev["targeted_edits"]
        with open(out_path, "w") as f:
            json.dump(data, f)

    print(f"Generated edit streams for {len(graph_files)} graphs in {out_dir}")


def _edits_from_json(raw_edits: list) -> list[Edit]:
    return [
        Edit(
            index=e["index"],
            type=EditType(e["type"]),
            vertex=e["vertex"],
            removed_colors=e.get("removed_colors"),
        )
        for e in raw_edits
    ]


def _edit_to_json(e: Edit, origin: str) -> dict:
    return {
        "index": e.index,
        "type": e.type.value,
        "vertex": e.vertex,
        "removed_colors": e.removed_colors,
        "from": origin,
    }


def _replay_stream(G, L, c, edits):
    """Apply a sequential stream, reverting infeasible edits (same as labels)."""
    for edit in edits:
        G_prev, L_prev, c_prev = G, L, c
        G, L = apply_edit(G, L, edit)
        result = list_dsatur(G, L)
        if not result.success:
            G, L, c = G_prev, L_prev, c_prev
            continue
        c = result.coloring
    return G, L, c


def _proposal_edits(G, L, c, rng):
    """Yield shrink_list edits that remove the current color of v (forces a recolor).

    Pass 1: drop only the current color. Pass 2: current color plus one other.
    Random edits rarely hit r*=1 (~2%); forcing a recolor of v skips the
    common no-op shrink that leaves r*=0.
    """
    verts = [
        v for v in G.nodes()
        if len(L.get(v, set())) >= 2 and c.get(v) in L.get(v, set())
    ]
    rng.shuffle(verts)
    for v in verts:
        yield Edit(index=0, type=EditType.SHRINK_LIST, vertex=v, removed_colors=[c[v]])
    rng.shuffle(verts)
    for v in verts:
        others = [x for x in L[v] if x != c[v]]
        if not others:
            continue
        extra = rng.choice(others)
        yield Edit(
            index=0,
            type=EditType.SHRINK_LIST,
            vertex=v,
            removed_colors=[c[v], extra],
        )


def collect_target_r_star_edits(
    G,
    L,
    c,
    rng,
    n_target: int,
    max_attempts: int,
    start_index: int,
    origin: str,
    target_r: int = TARGET_R_STAR,
) -> tuple[list[tuple[Edit, str]], int]:
    """Rejection-sample shrink_list edits until n_target have r* == target_r.

    One-offs: a success does not mutate (G, L, c). Returns (pairs, attempts).
    """
    found: list[tuple[Edit, str]] = []
    used: set[tuple] = set()
    attempts = 0
    for cand in _proposal_edits(G, L, c, rng):
        if len(found) >= n_target or attempts >= max_attempts:
            break
        key = (cand.vertex, tuple(sorted(cand.removed_colors or [])))
        if key in used:
            continue
        used.add(key)
        attempts += 1
        G_t, L_t = apply_edit(G, L, cand)
        result = list_dsatur(G_t, L_t)
        if not result.success:
            continue
        r_star = repair_radius(G, c, result.coloring, cand)
        if r_star != target_r:
            continue
        cand.index = start_index + len(found)
        found.append((cand, origin))
    return found, attempts


def augment_edits_for_target_r1(
    graphs_dir: str,
    edits_dir: str,
    n_per_snapshot: int = 4,
    max_attempts: int = 100,
    seed: int = 1,
) -> None:
    """Append genuine r*=1 one-off edits to each existing stream JSON.

    Snapshots: 'initial' (pre-stream) and 'final' (after the sequential
    stream). Does not rewrite `edits`; only writes/replaces `targeted_edits`.
    Graph-level holdout is unchanged because no new graphs are added.
    """
    graph_files = sorted(f for f in os.listdir(graphs_dir) if f.endswith(".json"))
    n_graphs = 0
    n_found = 0
    n_skip = 0
    n_zero = 0
    attempts_total = 0

    for i, graph_file in enumerate(graph_files):
        graph_path = os.path.join(graphs_dir, graph_file)
        edit_path = os.path.join(edits_dir, graph_file.replace(".json", "_edits.json"))
        if not os.path.exists(edit_path):
            logger.warning("Edit file missing for %s, skipping", graph_file)
            n_skip += 1
            continue

        graph = load_graph(graph_path)
        G = nx.Graph(graph.edges)
        L = {v: set(lst) for v, lst in graph.lists.items()}
        G.add_nodes_from(L.keys())

        init = list_dsatur(G, L)
        if not init.success:
            logger.warning("Infeasible initial coloring for %s, skipping", graph.graph_id)
            n_skip += 1
            continue

        with open(edit_path, "r") as f:
            edit_data = json.load(f)
        stream = _edits_from_json(edit_data["edits"])

        rng = random.Random(seed + i * 1000)
        collected: list[tuple[Edit, str]] = []

        found_i, att_i = collect_target_r_star_edits(
            G, L, init.coloring, rng, n_per_snapshot, max_attempts,
            TARGETED_INDEX_BASE, "initial",
        )
        collected.extend(found_i)
        attempts_total += att_i

        G_f, L_f, c_f = _replay_stream(G, L, init.coloring, stream)
        found_f, att_f = collect_target_r_star_edits(
            G_f, L_f, c_f, rng, n_per_snapshot, max_attempts,
            TARGETED_INDEX_BASE + n_per_snapshot, "final",
        )
        collected.extend(found_f)
        attempts_total += att_f

        edit_data["targeted_edits"] = [_edit_to_json(e, origin) for e, origin in collected]
        with open(edit_path, "w") as f:
            json.dump(edit_data, f)

        n_graphs += 1
        n_found += len(collected)
        if not collected:
            n_zero += 1
        if n_graphs % 25 == 0:
            logger.info(
                "augment r*=1: %d graphs, %d targeted kept, %d attempts",
                n_graphs, n_found, attempts_total,
            )

    logger.info(
        "augment r*=1 done: graphs=%d skipped=%d targeted=%d "
        "graphs_with_zero=%d attempts=%d n_per_snapshot=%d",
        n_graphs, n_skip, n_found, n_zero, attempts_total, n_per_snapshot,
    )