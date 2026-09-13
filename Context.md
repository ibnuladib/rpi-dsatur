# Research Methodology

## GNN-Based Incremental Heuristics for Online List Coloring in Dynamic Networks

### Algorithm: RPI-DSATUR (Radius-Predicting Incremental DSATUR)

Every step below ends with an explicit **Validation gate** — a concrete pass/fail check. Do not proceed to the next step until the current gate passes. This is what makes the methodology "validated" rather than just sequential: each stage produces evidence that it actually worked before you build on top of it.

---

## 0. Contribution statement

> We propose RPI-DSATUR, a GNN-based incremental heuristic for online list coloring: a GNN predicts the local radius over which an edit's effects must be repaired, and a classical list-coloring solver (DSATUR) performs the actual recoloring within that radius. Decisions are made as edits arrive, using only the current graph state and no knowledge of future edits. The GNN governs *where* incremental repair happens; DSATUR guarantees *validity* of the repair.

---

## 1. Formal problem setup

- Graph $G_t=(V_t,E_t)$; vertex $v$ has color list $L_t(v)\subseteq1,\dots,k$; coloring $c_t:V_t\to1,\dots,k$ with $c_t(v)\in L_t(v)$ for all $v$, and $c_t(u)\neq c_t(v)$ for every edge $(u,v)\in E_t$.
- Edit stream $\delta_1,\delta_2,\dots,\delta_T$, each $\delta_i \in $`shrink_list(v, L')`, `remove_vertex(v)`$$ for this project's scope, with $L' \subsetneq L(v)$, $L' \neq \emptyset$.
- Online constraint: $c_t = f(G_{t-1}, L_{t-1}, c_{t-1}, \delta_t)$ — the decision function may depend only on the past and present, never on $\delta_{t+1},\dots,\delta_T$.
- **Repair radius** of an edit: $r^*(\delta_t) = \max_{v: c_t(v)\neq c_{t-1}(v)} \text{dist}{G{t-1}}(v_{\delta_t}, v)$, where $v_{\delta_t}$ is the edited vertex. Convention: if no vertex changes color, $r^*=0$. If $v_{\delta_t}$ itself is removed (the `remove_vertex` case), compute distances in $G_{t-1}$ restricted to $V_{t-1}\setminusv_{\delta_t}$ from $v_{\delta_t}$'s former neighbors.
- $r^*$ is the ground-truth regression target computed by comparing two full DSATUR runs (Section 5); it is **never available at inference time** — the GNN predicts an estimate $\hat r$ from pre-edit information only.

**Validation gate:** write these definitions into a shared `definitions.md` in the repo and have every team member confirm, on one shared 6-node example graph, that they compute $r^*$ by hand identically. Disagreement here means later results are uninterpretable — resolve before writing any code.

---

## 2. Environment setup

1. Python 3.10+, PyTorch ≥2.0, PyTorch Geometric, NetworkX ≥3.0, NumPy, SciPy (for the significance tests in Section 10), Matplotlib.
2. Repo structure:

```
/data_gen/        graph + list + edit-stream generators
/baselines/       list-Greedy, list-DSATUR
/labels/          repair-radius label computation
/model/           GNN radius predictor
/inference/       RPI-DSATUR algorithm (Section 7)
/experiments/     evaluation scripts, plotting
/tests/           unit tests for every module above
/results/
```

1. Global seed policy: every generator, model initializer, and train/test split takes an explicit `seed: int` argument, and no module reads `numpy.random` / `random` global state implicitly. Store the seed used for every artifact (graph, label set, trained model) alongside the artifact itself (e.g., a `.json` sidecar file).
2. Pin exact package versions in a `requirements.txt` generated with `pip freeze` — required for reproducibility if results are questioned later.

**Validation gate:** run the same generator twice with the same seed, confirm byte-identical output (hash the serialized graph). Run it twice with different seeds, confirm output differs. Both checks automated as a `test_determinism.py`.

---

## 3. Classical baselines (build first — everything else depends on this)

**3.1 — `list_greedy(G, L, order=None)`.** Process vertices in `order` (default: arbitrary/insertion order). Assign the lowest-indexed color in $L(v)$ not used by an already-colored neighbor. Return an explicit failure object (not `None`, to avoid silent bugs) if no such color exists, including which vertex failed.

**3.2 — `list_dsatur(G, L, fixed={})`.** Standard DSATUR generalized to lists, with an optional `fixed: dict[vertex, color]` of vertices whose colors are locked and never revisited. This parameter is what lets Section 7 call DSATUR on a *subgraph* while treating its boundary as constants — implement and test it now, not when Section 7 needs it.

- Priority queue, recomputed dynamically: at each step pick the uncolored vertex with (a) highest saturation degree = number of *distinct* colors among colored neighbors (including `fixed` neighbors), tie-break by (b) largest $|L(v)| / (\deg(v)+1)$ (list-degree ratio), tie-break by (c) vertex id (for determinism).
- Assign the lowest-indexed color in $L(v)$ not already used by a colored neighbor. If none exists, the whole call fails and returns the partial coloring plus the failing vertex (needed for diagnosing Section 7 retries).
- Explicitly handle: isolated vertices (trivially colorable), vertices with $|L(v)|=1$ that conflict with a `fixed` neighbor (immediate, correctly reported failure), and empty graphs (return the empty coloring, not an error).

**3.3 — Unit tests, required before trusting either function as a label generator:**

- 5 hand-built graphs (≤10 nodes) with lists chosen so you can verify the correct coloring (or correct infeasibility) by hand.
- 1 test confirming `list_dsatur(G, L, fixed={...})` never changes a `fixed` vertex's color, even under conflict — it must report failure instead of overriding a fixed color.
- 1 test on $K_{n}$ (complete graph) with $|L(v)|=n-1$ for all $v$ — a known-infeasible instance — confirming both functions correctly report failure rather than crashing or returning an invalid coloring.
- 1 property-based test: run `list_dsatur` on 100 random small graphs, and for every output, independently re-verify validity (every $c(v)\in L(v)$, no monochromatic edge) with a separate, trivially-correct O(V+E) checker function. This decouples "DSATUR looks right" from "my validity checker has the same bug as my DSATUR implementation."

**Validation gate:** all unit tests pass; the independent validity checker from 3.3 reports zero violations across ≥500 random small instances (mix of feasible and infeasible). Do not proceed to Section 4 until this passes — every later label and metric depends on these two functions being correct.

---

## 4. Synthetic data generation

**4.1 — Base graphs.** Erdős–Rényi $G(n,p)$ with $n \in 100, 200, 500$. Choose $p$ per $n$ so **average degree lands in [4,10]** (i.e., $p \approx \bar d / (n-1)$); record the realized average degree and degree-distribution std per graph, don't just trust the target. Generate 50 graphs per size (150 total) — more than the original 30–50 estimate, to leave headroom for the 70/15/15 split in Section 8.2 without starving the test set at $n=500$.

**4.2 — List assignment.** Per vertex $v$: $|L(v)| \sim \text{Uniform}d(v), d(v)+1, d(v)+2, d(v)+3$ (discrete, inclusive); draw that many colors uniformly without replacement from palette $1,\dots,k$, $k = \Delta(G) + 5$ where $\Delta(G)$ is max degree in that graph (generic feasibility headroom, not a guarantee — see 4.3).

**4.3 — Initial coloring + feasibility filter.** Run `list_dsatur` once per graph for $c_0$. If it fails, **do not discard silently** — log the graph's $n$, $p$, realized $\bar d$, and $k$ used, so you can later report *why* instances were infeasible (e.g., "infeasibility rate was 3% at $n=500$, concentrated in graphs with $\bar d > 9$"). Target: **infeasibility rate <10%** — if higher, increase $k$ (the palette-slack constant) and regenerate; a high discard rate silently biases your dataset toward easier instances without you knowing it.

**4.4 — Edit streams.** Per surviving graph, generate one stream of **20 edits**:

- 60% `shrink_list(v,L')`: pick $v$ uniformly among vertices with $|L(v)|\geq 2$ (so removal keeps $L'$ non-empty); remove 1 color w.p. 0.7, 2 colors w.p. 0.3, chosen uniformly at random from $L(v)$.
- 40% `remove_vertex(v)`: pick $v$ uniformly among all current vertices (isolated vertices allowed — removing one is a valid, trivial edit and should appear in the label distribution as an $r^*=0$ example, not be excluded).
- Apply edits sequentially to the *same* evolving graph (so edit 5 sees the graph as modified by edits 1–4) — this matches the real online setting and must not be shuffled or treated as independent one-off edits later.
- **Do not let a `remove_vertex` stream shrink a graph below 20% of its original size** — if it would, replace that step with a `shrink_list` edit instead. Otherwise late-stream edits on a near-empty graph become degenerate ($r^*$ trivially small for uninteresting reasons) and dilute the harder examples.

**Validation gate:** plot the realized average-degree distribution per $n$ (should cluster near target), report and sanity-check the infeasibility rate (<10%, see 4.3), and plot a histogram of edit types actually applied per stream (should be close to 60/40 — large deviations indicate a bug in the sampling-with-constraints logic in 4.4).

---

## 5. Label generation (the training signal)

For every (graph, edit-index) pair — 150 graphs × 20 edits = 3,000 raw examples before any filtering:

1. Run `list_dsatur` on $(G_{t-1}, L_{t-1})$; assert the result equals the stored $c_{t-1}$ (sanity check — catches state-tracking bugs in the stream simulator). Fail loudly, don't skip, if this mismatches.
2. Apply $\delta_t$ → $(G_t, L_t)$.
3. Run `list_dsatur` on $(G_t, L_t)$ **from scratch** → $c_t$. If this fails (edit made the instance infeasible), **discard this single edit** from the label set but keep the stream going with the previous valid state for the next edit — log the discard.
4. Compute $r^*$ per Section 1's definition via BFS outward from $v_{\delta_t}$ in $G_{t-1}$ (or, for `remove_vertex`, from its former neighbor set — see Section 1).
5. Bucket into classes $0,1,2,4,8,\text{full}$ using **ceiling assignment**: e.g. $r^*=3 \to$ bucket "4", $r^*=6\to$ bucket "8", $r^*>8$ or BFS never reaches a boundary $\to$ bucket "full". Ceiling (not nearest) is deliberate: at inference time under-predicting radius is unsafe (Section 7), so labels should reflect the same "round up" discipline you'll enforce later.
6. Store: (3-hop feature window around $v_{\delta_t}$ in $G_{t-1}$ — see Section 6.1 — edit type, bucket label, and the raw $r^*$ for later regression-style diagnostics even though the model itself is a classifier).

**Validation gate:**

- Report the discard rate from step 3 (<5% expected; if much higher, your `k` slack constant in 4.2 is too tight).
- Report the **class balance** across the 6 buckets. If one bucket has <5% of examples, either accept it (and report expected low recall on that class) or adjust the edit-generation distribution in 4.4 to produce more of that case — decide and document this explicitly rather than discovering it after training.
- Re-derive $r^*$ for 20 randomly sampled examples using an independent, separately-written script (not reusing the same BFS function), and confirm agreement — this catches bugs in the label pipeline itself, the single most consequential piece of code in the project since every downstream number depends on it.

---

## 6. Radius predictor (the GNN)

**6.1 — Input construction.** For edit at $v$ in $G_{t-1}$: extract the induced subgraph on $v$'s 3-hop neighborhood (fixed extraction radius for *feature context only* — unrelated to the predicted repair radius $\hat r$). Node features per vertex $u$ in this window:

- current color $c_{t-1}(u)$, embedded (not raw integer — use a learned embedding table of size $k$, since color indices are categorical, not ordinal),
- $|L(u)|$ and tightness ratio $|L(u)|/(\deg(u)+1)$,
- degree $d(u)$ (normalize by dividing by the graph's max degree, so the feature scale is comparable across the three graph sizes in the sweep),
- binary: is-edited-vertex,
- binary: edit-type (shrink=0, remove=1).

Pad or mask if the 3-hop window has fewer than some minimum node count (e.g., near the periphery of the graph) — decide on a fixed padding scheme now so batch construction doesn't silently break on small windows.

**6.2 — Architecture.** 3 GraphSAGE (or GCN) layers, hidden dim 64, ReLU, dropout 0.2 between layers → global mean-pool over the extracted window → 2-layer MLP (64→32→6) → softmax over the 6 buckets.

**6.3 — Loss and optimizer.** Cross-entropy over the 6 classes, **class-weighted inversely by frequency** from the Section 5 validation gate (if imbalance exists) so the rare "full recompute" class isn't ignored — that class is arguably the most important to get right, since missing it is the case that produces an invalid intermediate attempt in Section 7. Adam, lr $1\times10^{-3}$, batch size 64, early stopping on validation cross-entropy (patience 10 epochs), max 200 epochs.

**6.4 — This is a classification model, not a coloring model** — no Potts-energy or list-masking loss here; those apply only inside the Section 7 DSATUR call.

**Validation gate:**

- Report train/val/test accuracy **and** a full confusion matrix over the 6 buckets — accuracy alone hides whether errors are "off by one bucket" (tolerable, since Section 7 rounds up and retries) vs. "predicted bucket 0 when truth was full" (dangerous — a large under-prediction costs an extra retry cycle, which is the exact failure mode the retry loop exists to catch, so it should be rare, not common).
- Explicitly report **the rate of under-prediction** (predicted bucket < true bucket) vs over-prediction, separately from raw accuracy — under-prediction rate is the number that determines your Section 8 fallback-rate results, so know it before running the full experiment.
- Gate: val accuracy should exceed the majority-class baseline (predicting the most common bucket always) by a clear margin — if it doesn't, the model isn't learning signal and Section 7 will behave like the fixed-radius baseline by construction, which would make your headline comparison vacuous.

---

## 7. RPI-DSATUR inference algorithm (the core contribution — implement exactly)

```
function RPI_DSATUR(G_prev, L_prev, c_prev, edit):
    G_t, L_t = apply(edit, G_prev, L_prev)
    bucket_pred = GNN_predict(edit, G_t, L_t, c_prev)
    r_hat = bucket_to_radius(bucket_pred)      # round UP, never down
    attempt = 0
    while attempt < MAX_RETRIES:
        S = subgraph_within_radius(G_t, edit.vertex, r_hat)
        boundary_fixed = { u: c_prev[u] for u in neighbors_of(S) if u not in S }
        c_S = list_dsatur(S, L_t restricted to S, fixed=boundary_fixed)
        if c_S is a failure:
            r_hat = next_bucket_up(r_hat); attempt += 1; continue
        c_t = merge(c_prev, c_S)               # vertices outside S keep c_prev
        if count_conflicts(G_t, L_t, c_t) == 0:
            return c_t, r_hat, attempt          # success — log radius used and retries
        r_hat = next_bucket_up(r_hat); attempt += 1
    c_t = list_dsatur(G_t, L_t)                 # fallback: full recompute
    return c_t, "full_recompute", attempt
```

Set `MAX_RETRIES = 3`. Radius ladder for `next_bucket_up`: $0\to1\to2\to4\to8\to$ full — always move to the *next* bucket, never re-attempt the same radius.

**Edge cases to handle explicitly (each needs its own unit test):**

- `edit.vertex` was removed by the edit itself (`remove_vertex` case) — `subgraph_within_radius` must be defined relative to the *former* neighbors, since the vertex no longer exists in $G_t$.
- $S$ disconnects into multiple components after removal — `list_dsatur(S, ...)` must handle disconnected input correctly (verify this in Section 3.3, not discovered here for the first time).
- The full-graph fallback itself fails (instance genuinely infeasible after the edit) — this must be reported as a distinct outcome ("infeasible after edit"), not silently conflated with a successful full recompute, since it changes how you interpret the fallback-rate metric.

**Validation gate:**

- On the held-out test set, verify **every single output** has zero conflicts and zero list violations (checked with the same independent validity checker from Section 3.3) — 100% required, since validity is supposed to be guaranteed by construction; any failure here is a bug in the merge/boundary logic, not an acceptable error rate.
- Log and report the distribution of `attempt` counts (0, 1, 2, 3-fallback) across the test set — this is required raw data for Section 9's fallback-rate metric, so confirm the logging is wired up correctly before the full experimental run in Section 8, not after.

---

## 8. Experimental design

**8.1 — Systems compared** (identical edit streams, identical starting colorings, for fairness):

1. **Full recompute**: `list_dsatur(G_t, L_t)` from scratch every edit.
2. **Fixed-radius baseline**: identical algorithm to Section 7, but replace `GNN_predict` with a constant $\hat r = 2$ for every edit (still uses the same retry ladder and fallback). Isolates whether *learning* the radius adds value over a naive fixed guess.
3. **RPI-DSATUR**: full Section 7 algorithm with the trained predictor from Section 6.

**8.2 — Splits.** Split **by graph** (not by edit), 70/15/15 train/val/test — no test-set graph's edits appear in training data. With 150 graphs total (50 per size), that's 35/8/7 per size; if this feels thin for $n=500$, generate additional graphs at that size specifically rather than reducing rigor elsewhere.

**8.3 — Repetitions.** Predictor weights are fixed after training (Section 6). Evaluate the test set with **5 random seeds** controlling: (a) which specific test graphs are sampled if you subsample, and (b) edit-stream generation if you regenerate streams for evaluation rather than reusing the exact Section 4 streams. State explicitly in the writeup whether evaluation uses the *same* streams used for label generation or freshly generated ones — reusing them is simpler and fine, since the predictor never saw test-graph streams during training regardless.

**8.4 — Sweep.** Repeat 8.1–8.3 separately per graph size ($n=100,200,500$) to characterize how the time/quality gap scales with graph size — this scaling trend, not just a single-size number, is the more convincing evidence for the paper's speed claim.

**Validation gate:** before running the full matrix, run one small pilot (one graph, 3 systems, 1 seed) end to end and manually inspect the per-edit log for all three systems — confirm timing, vertices-touched, and validity numbers look sane (e.g., full-recompute should always touch the most vertices on average; fixed-radius and RPI-DSATUR should be faster). Catching a bug here costs minutes; catching it after the full 3-size × 3-system × 5-seed run costs a full day.

---

## 9. Metrics (compute per edit, aggregate per system per graph size)


| Metric           | How computed                                    | Expected pattern if the method works                              |
| ---------------- | ----------------------------------------------- | ----------------------------------------------------------------- |
| Validity         | conflicts + list violations after final output  | 0 for all three systems, always                                   |
| Wall-clock time  | time from `apply(edit)` to returning $c_t$      | RPI-DSATUR < full recompute; RPI-DSATUR ≤ fixed-radius on average |
| Vertices touched | $\lvertv: c_t(v)\neq c_{t-1}(v)\rvert$          | RPI-DSATUR < full recompute                                       |
| Radius accuracy  | predicted bucket vs. true $r^*$ (test set only) | notably above majority-class baseline (Section 6 gate)            |
| Fallback rate    | fraction of edits hitting `MAX_RETRIES`         | low (target <15%); nonzero is fine and expected                   |


**Validation gate:** for every metric, confirm the "expected pattern" column actually holds directionally before running significance tests — if RPI-DSATUR is *slower* than full recompute on average, that's a result to report honestly, but check first whether it's a genuine finding (e.g., overhead from small subgraph extraction dominates on very sparse edits) or an implementation bug (e.g., `subgraph_within_radius` not actually restricting computation).

---

## 10. Statistical reporting

Per metric, per graph size: report mean ± standard deviation over the 5 seeds. For RPI-DSATUR vs. full recompute and RPI-DSATUR vs. fixed-radius, run a **paired** test (Wilcoxon signed-rank, since per-edit timing/vertex-touched distributions are unlikely to be normal) over per-edit values from shared edit streams — pairing is valid here specifically because all three systems are run on identical streams. Report the p-value **and** an effect size (e.g., median difference, or rank-biserial correlation) — a p-value alone doesn't tell a reader whether the improvement is practically meaningful.

**Validation gate:** confirm sample sizes going into each test (should be ≈ test-graphs × 20 edits per size, minus Section 5 discards) are large enough that the test is meaningful (a few hundred paired observations minimum per graph size) — report the exact $n$ used in each test in the results table, not just the p-value.

---

## 11. Timeline (working days)


| Day | Task                                                    | Exit criterion                              |
| --- | ------------------------------------------------------- | ------------------------------------------- |
| 1   | Section 2–3: environment + baselines + unit tests       | Section 3 validation gate passes            |
| 2   | Section 4: data generation                              | Section 4 validation gate passes            |
| 3   | Section 5: label generation + sanity checks             | Section 5 validation gate passes            |
| 4   | Section 6: build + train radius predictor               | Section 6 validation gate passes            |
| 5   | Section 7: implement + debug RPI-DSATUR inference loop  | Section 7 validation gate passes            |
| 6   | Section 8: pilot run, then full experimental matrix     | Section 8 pilot gate passes before full run |
| 7   | Section 9–10: compute metrics, statistical tests, plots | All expected-pattern checks reviewed        |
| 8   | Write results section                                   | —                                           |


---

## 12. Deliverables checklist

- [ ] Working `list_greedy` and `list_dsatur` (with `fixed` support), unit-tested, validated against an independent checker (Section 3)
- [ ] ≥150 synthetic graphs with lists and 20-edit streams each, infeasibility rate <10% and reported (Section 4)
- [ ] Labeled (edit → repair-radius bucket) dataset with discard rate and class balance reported, labels independently re-verified on a sample (Section 5)
- [ ] Trained GNN radius classifier with accuracy, confusion matrix, and under/over-prediction rates reported (Section 6)
- [ ] RPI-DSATUR inference implementation matching Section 7 pseudocode exactly, 100% validity on held-out test set, edge cases unit-tested
- [ ] Full-recompute and fixed-radius baselines run on identical edit streams (Section 8)
- [ ] Results table: mean ± std, all 5 metrics, all 3 systems, all 3 graph sizes, with sample sizes stated
- [ ] Paired significance tests + effect sizes, RPI-DSATUR vs. each baseline (Section 10)
- [ ] Fallback-rate and radius-accuracy analysis as the direct "why does this work" evidence

---

## Appendix A — Implementation Spec (for a coding agent)

Everything below is meant to be handed to a coding agent directly, one section at a time, in order. Each function signature, file path, and schema is fixed — do not let an agent invent its own naming or shape, since later steps (and the validation gates above) assume these exact contracts.

### A.1 Shared data types (`/common/types.py`)

```python
from dataclasses import dataclass, field
from enum import Enum

@dataclass
class ColoringResult:
    success: bool
    coloring: dict[int, int]          # vertex_id -> color, partial if success=False
    failed_vertex: int | None = None  # set when success=False

@dataclass
class ValidityReport:
    is_valid: bool
    num_conflicts: int
    num_list_violations: int
    conflict_edges: list[tuple[int, int]] = field(default_factory=list)
    violating_vertices: list[int] = field(default_factory=list)

class EditType(str, Enum):
    SHRINK_LIST = "shrink_list"
    REMOVE_VERTEX = "remove_vertex"

@dataclass
class Edit:
    index: int
    type: EditType
    vertex: int
    removed_colors: list[int] | None = None   # populated only for SHRINK_LIST

@dataclass
class GraphInstance:
    graph_id: str
    n: int
    seed: int
    edges: list[tuple[int, int]]
    lists: dict[int, list[int]]       # vertex_id -> color list
    k: int
    avg_degree_realized: float

@dataclass
class EditOutcome:
    system: str                       # "full_recompute" | "fixed_radius" | "rpi_dsatur"
    graph_id: str
    edit_index: int
    time_ms: float
    vertices_touched: int
    radius_used: int | str            # int, or "full_recompute"
    attempts: int
    fallback_triggered: bool
    valid: bool                       # from ValidityReport.is_valid
```

Use these dataclasses everywhere — every module below imports from `common/types.py` rather than redefining ad hoc dicts. This is what keeps Section 8's metrics computation (Section 9) able to consume output from all three systems uniformly.

### A.2 Baselines module (`/baselines/`)

`/baselines/greedy.py`

```python
def list_greedy(
    G: "networkx.Graph",
    L: dict[int, set[int]],
    order: list[int] | None = None,
) -> ColoringResult: ...
```

`/baselines/dsatur.py`

```python
def list_dsatur(
    G: "networkx.Graph",
    L: dict[int, set[int]],
    fixed: dict[int, int] | None = None,
) -> ColoringResult: ...
```

- `fixed` vertices must never appear as keys in the returned `coloring` with a value different from what was passed in; if a `fixed` neighbor makes a vertex uncolorable, return `ColoringResult(success=False, failed_vertex=<that vertex>)`.
- Tie-break order for saturation degree ties: (1) larger `len(L[v]) / (G.degree[v] + 1)`, (2) smaller vertex id — implement as a sort key tuple `(-saturation, -list_ratio, vertex_id)` so behavior is deterministic and testable.

`/baselines/validity.py`

```python
def check_validity(
    G: "networkx.Graph",
    L: dict[int, set[int]],
    coloring: dict[int, int],
) -> ValidityReport: ...
```

- Must be written independently from `list_dsatur`'s internal conflict-checking logic (do not import or call any DSATUR-internal helper) — this independence is required by the Section 3.3 validation gate.

### A.3 Data generation module (`/data_gen/`)

`/data_gen/generate_graphs.py`

```python
def generate_graph(n: int, target_avg_degree: float, seed: int) -> GraphInstance: ...

def assign_lists(G: "networkx.Graph", seed: int) -> dict[int, list[int]]:
    # |L(v)| ~ DiscreteUniform{deg(v), deg(v)+1, deg(v)+2, deg(v)+3}
    # k = max_degree(G) + 5
    ...

def generate_dataset(
    sizes: list[int],
    graphs_per_size: int,
    base_seed: int,
    out_dir: str,
) -> None:
    # writes one JSON file per graph: {out_dir}/{graph_id}.json
    # graph_id format: "n{n}_s{seed}", e.g. "n100_s0042"
    ...
```

`/data_gen/generate_edits.py`

```python
def generate_edit_stream(
    graph: GraphInstance,
    num_edits: int,
    seed: int,
    min_size_fraction: float = 0.2,
) -> list[Edit]:
    # 60% SHRINK_LIST / 40% REMOVE_VERTEX, per Section 4.4 rules exactly
    ...
```

JSON schema written by `generate_dataset` (one file per graph, `{graph_id}.json`):

```json
{
  "graph_id": "n100_s0042",
  "n": 100,
  "seed": 42,
  "edges": [[0,1],[0,2], "..."],
  "lists": {"0": [1,2,5], "1": [2,3], "...": "..."},
  "k": 14,
  "avg_degree_realized": 6.1
}
```

Edit stream file (`{graph_id}_edits.json`):

```json
{
  "graph_id": "n100_s0042",
  "edits": [
    {"index": 0, "type": "shrink_list", "vertex": 17, "removed_colors": [3]},
    {"index": 1, "type": "remove_vertex", "vertex": 5, "removed_colors": null}
  ]
}
```

CLI entry point (`/data_gen/__main__.py`), so the agent can run this as one command:

```
python -m data_gen generate --sizes 100 200 500 --graphs-per-size 50 --base-seed 0 --out data/raw/graphs
python -m data_gen edits --graphs-dir data/raw/graphs --num-edits 20 --seed 0 --out data/raw/edits
```

### A.4 Label generation module (`/labels/`)

`/labels/build_labels.py`

```python
BUCKET_EDGES = [0, 1, 2, 4, 8]   # bucket "full" catches anything above 8
BUCKET_NAMES = ["0", "1", "2", "4", "8", "full"]

def repair_radius(
    G_prev: "networkx.Graph",
    c_prev: dict[int, int],
    c_t: dict[int, int],
    edit: Edit,
) -> int: ...   # BFS-based, per Section 1 definition; returns raw r*, not bucketed

def bucketize(r_star: int) -> int:
    # ceiling assignment into BUCKET_NAMES index, per Section 5 step 5
    ...

def extract_features(
    G_prev: "networkx.Graph",
    L_prev: dict[int, set[int]],
    c_prev: dict[int, int],
    edit: Edit,
    hop_radius: int = 3,
) -> "torch_geometric.data.Data": ...  # node features exactly as listed in Section 6.1

def build_label_dataset(
    graphs_dir: str,
    edits_dir: str,
    out_path: str,   # single output: data/labels/labels.pt  (list of torch_geometric Data objects, or a parquet index + a separate .pt tensor cache)
) -> None: ...
```

Each stored label example must retain: `graph_id`, `edit_index`, the PyG `Data` object (features + edge_index for the 3-hop window), `edit_type`, `bucket` (int 0–5), and `r_star` (raw int, kept for diagnostics per Section 5's validation gate even though training only uses `bucket`).

CLI: `python -m labels build --graphs-dir data/raw/graphs --edits-dir data/raw/edits --out data/labels/labels.pt`

### A.5 Model module (`/model/`)

`/model/config.yaml`

```yaml
model:
  hidden_dim: 64
  num_layers: 3
  dropout: 0.2
  conv_type: sage      # or gcn
train:
  lr: 0.001
  batch_size: 64
  max_epochs: 200
  patience: 10
  class_weighting: inverse_frequency
data:
  labels_path: data/labels/labels.pt
  split: [0.70, 0.15, 0.15]
  split_by: graph_id     # never split_by: edit
seed: 0
```

`/model/radius_gnn.py`

```python
class RadiusGNN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int = 6, dropout: float = 0.2): ...
    def forward(self, data: "torch_geometric.data.Batch") -> "torch.Tensor":  # logits, shape [batch, 6]
        ...
```

`/model/train.py`

```python
def train(config_path: str) -> None:
    # loads config.yaml, builds train/val/test split BY graph_id, trains, early-stops,
    # saves checkpoint to model/checkpoints/radius_gnn.pt,
    # writes model/reports/train_report.json with: val_accuracy, test_accuracy,
    # confusion_matrix (6x6 nested list), under_prediction_rate, over_prediction_rate
    ...
```

CLI: `python -m model.train --config model/config.yaml`

`train_report.json` schema (required for the Section 6 validation gate):

```json
{
  "val_accuracy": 0.71,
  "test_accuracy": 0.69,
  "majority_class_baseline_accuracy": 0.31,
  "confusion_matrix": [["...6x6..."]],
  "under_prediction_rate": 0.09,
  "over_prediction_rate": 0.22,
  "class_counts": {"0": 412, "1": 388, "2": 301, "4": 205, "8": 88, "full": 41}
}
```

### A.6 Inference module (`/inference/`)

`/inference/rpi_dsatur.py`

```python
def bucket_to_radius(bucket: int | str) -> int | str: ...   # "full" maps to a sentinel meaning "whole graph"
def next_bucket_up(bucket: int | str) -> int | str: ...      # ladder: 0->1->2->4->8->"full"
def subgraph_within_radius(G: "networkx.Graph", center: int, radius: int | str) -> "networkx.Graph": ...

def rpi_dsatur_step(
    G_prev: "networkx.Graph", L_prev: dict[int, set[int]], c_prev: dict[int, int],
    edit: Edit, model: "RadiusGNN", max_retries: int = 3,
) -> tuple[ColoringResult, EditOutcome]:
    # implements the pseudocode in Section 7 exactly
    ...

def fixed_radius_step(
    G_prev, L_prev, c_prev, edit, radius: int = 2, max_retries: int = 3,
) -> tuple[ColoringResult, EditOutcome]:
    # identical control flow to rpi_dsatur_step but skips model call, r_hat fixed at `radius`
    ...

def full_recompute_step(
    G_prev, L_prev, c_prev, edit,
) -> tuple[ColoringResult, EditOutcome]:
    # calls list_dsatur(G_t, L_t) directly, wraps result as EditOutcome for uniform logging
    ...
```

All three `*_step` functions must return the same `EditOutcome` shape (Section A.1) so `/experiments/run_matrix.py` can treat them identically.

### A.7 Experiments module (`/experiments/`)

`/experiments/run_matrix.py`

```
python -m experiments.run_matrix \
    --systems full_recompute fixed_radius rpi_dsatur \
    --sizes 100 200 500 \
    --seeds 0 1 2 3 4 \
    --model-checkpoint model/checkpoints/radius_gnn.pt \
    --out results/raw_outcomes.jsonl
```

Output: one JSON-lines file, one `EditOutcome` per line (flattened to JSON), across every (system, size, seed, graph, edit) combination — this single file is the input to Section 9/10's analysis notebook.

`/experiments/analyze.ipynb` (notebook, per the earlier files-vs-notebooks answer): loads `results/raw_outcomes.jsonl` into a pandas DataFrame, computes the Section 9 metrics table and Section 10 paired Wilcoxon tests, produces plots.

### A.8 Testing (`/tests/`)

One file per module, mirroring the source tree exactly: `test_greedy.py`, `test_dsatur.py`, `test_validity.py`, `test_generate_graphs.py`, `test_generate_edits.py`, `test_build_labels.py`, `test_radius_gnn.py`, `test_rpi_dsatur.py`. Run with:

```
pytest tests/ -v --cov=. --cov-report=term-missing
```

Every function listed in A.2–A.6 needs at least one corresponding test; the Section 3.3 property-based test (100 random small graphs, independently re-validated) belongs in `test_dsatur.py`.

### A.9 Logging convention

Use Python's `logging` module everywhere, one logger per module (`logging.getLogger(__name__)`), level `INFO` for milestone messages ("generated 50 graphs at n=100", "discard rate: 3.2%"), level `WARNING` for validation-gate near-misses (e.g., infeasibility rate approaching the 10% threshold), level `ERROR` only for actual failures that stop execution. Do not use bare `print()` in any module under `/baselines/`, `/data_gen/`, `/labels/`, `/model/`, or `/inference/` — reserve `print` for the analysis notebook only.

---

## 13. Dataset (as run)

- **Graphs:** 300 generated (n ∈ {100,200,500}); 6 infeasible at initial DSATUR skipped; **294 labeled**. Graph-level split seed 0: 205 / 44 / **45** holdout (`model/reports/hop_shell/test_graph_ids.json`, identical across lineages).
- **Sequential edits:** ~45 per graph (not the original spec of 20). **13,103** labeled sequential examples after <1% infeasible-edit discards. Test sequential: **2,007** (45 graphs).
- **Class distribution (sequential, 6-class linear buckets {0,1,2,3,4,full}):** `0:6830, 1:292, 2:931, 3:2875, 4:1873, full:302`. Class 1 is 2.2% on natural streams (51 test).
- *r=1 augmentation (label-only one-offs; sequential streams unchanged):** +2,935 targeted edits → **16,038** examples. Class 1: 292 → 3,227 (test 51 → 501). Used only for the merged-{0,1} operating point; Section 8 matrix replays sequential streams only.
- **Features locked:** hop window **4**, per-node `x` dim 7, 10-d `global_features` (4 structural + 6 post-edit slack). Retry ladder: `0→1→2→3→4→full`, `MAX_RETRIES=3`.

---

## 14. Diagnostic — class 1 (verdict B)

Independent of training. Slack does **not** mark the 0/1 boundary.

- *r=0** = no color diffs at dist>0 between two independent full DSATUR runs (only the edit vertex may recolor). It is not `slack_v==0` and not “a neighbor must recolor.”
- `slack_v_is_zero` is identically 0 on all 13,103 sequential labels (dead feature). Shrink-list never has `slack_v==0` (min 1); remove-vertex zeros the slack vector.
- 5-fold tree/LR ceiling on the 10 global scalars for class 1: recall **~0.20–0.21**. GNN recall 0.000 on natural streams is worse than that ceiling, but scalars alone will not solve class 1.
- **Decision (kept):** stop adding slack variants; leave globals at 10 dims. Treat class 1 on natural streams as a **label-semantics limitation**, not a local-slack feature gap. On targeted r*=1 extras, class 1 is learnable (recall 0.39–0.69) but accuracy-optimal rules spend those wins on 0↔1 swaps.

---

## 15. Classifier results (through hop-shell / r1 joint)

72% 6-class test accuracy **was not met**. Freeze at this section: best trained 6-class **0.7005**; best merged-{0,1} **0.7745**. Negative results in §18; threshold / hop-5 in §19 (canonical trained 6-class still 0.7005).

**One row per lineage** (test, with val-fit class bias when that run used it):


| lineage                            | labels           | selection              | pool      | 6-class acc | merged-{0,1} | class-1 R | 4→0 cell |
| ---------------------------------- | ---------------- | ---------------------- | --------- | ----------- | ------------ | --------- | -------- |
| §13 CE + √inv-freq + sampler       | sequential 13103 | macro-F1               | mean      | 0.561       | —            | 0.020     | —        |
| §14 slack, class weights off       | sequential 13103 | macro-F1               | mean      | 0.6931      | —            | 0.000     | 138      |
| guarded + bias                     | sequential 13103 | val-acc, F1≥0.40       | mean      | 0.7000      | 0.7175       | 0.000     | 146      |
| **hop-shell (secondary 6-class)**  | sequential 13103 | val-acc, F1≥0.40       | hop-shell | **0.7005**  | 0.7180       | 0.000     | 147      |
| joint (fallback to macro-F1)       | sequential 13103 | joint; infeasible      | mean      | 0.7000      | 0.7170       | 0.000     | 144      |
| r1 CE + macro-F1                   | r1 16038         | macro-F1               | mean      | 0.5438      | 0.7253       | **0.685** | 99       |
| r1 guarded + bias                  | r1 16038         | val-acc, F1≥0.40       | mean      | 0.5735      | 0.7705       | 0.004     | 144      |
| **r1 hop-shell (headline merged)** | r1 16038         | val-acc, F1≥0.40       | hop-shell | 0.5991      | **0.7745**   | 0.277     | 138      |
| r1 joint + bias                    | r1 16038         | acc | R1≥0.30, F1≥0.40 | mean      | 0.6064      | 0.7713       | 0.387     | 134      |


Margin over majority is stable (~+17pp): 0.7005 vs 0.5291 (sequential), 0.5991 vs 0.4322 (r1). The 4→0 cell is **134–154** across every run including hop-5 (~6.7–7.7pp) — information-bound, not optimization-bound. Full lineage table: §19.0.

### 15.1 Confusion — hop-shell sequential (acc 0.7005, test n=2007)

Rows = true, cols = pred, order `['0','1','2','3','4','full']`. Class-1 column is all zeros. Checkpoint: `model/checkpoints/radius_gnn_hopshell.pt`. Report: `model/reports/hop_shell/train_report.json`.

```
[[928  0   4  96  34   0]
 [ 35  0   2   9   5   0]
 [ 12  0  53  48   3   0]
 [ 61  0  22 338  17   0]
 [147  0   0  58  87   0]
 [ 47  0   0   0   1   0]]
```


| class | n    | P     | R     | F1    |
| ----- | ---- | ----- | ----- | ----- |
| 0     | 1062 | 0.754 | 0.874 | 0.810 |
| 1     | 51   | 0.000 | 0.000 | 0.000 |
| 2     | 116  | 0.654 | 0.457 | 0.538 |
| 3     | 438  | 0.616 | 0.772 | 0.685 |
| 4     | 292  | 0.592 | 0.298 | 0.396 |
| full  | 48   | 0.000 | 0.000 | 0.000 |


Under-prediction 0.191, over-prediction 0.109. Majority baseline 0.5291.

### 15.2 Confusion — r1 hop-shell (merged-{0,1} 0.7745, test n=2457)

Same 45 graphs; +450 class-1 one-offs. Checkpoint: `model/checkpoints/radius_gnn_r1_hopshell.pt`. Report: `model/reports/r1_hopshell/train_report.json`. 6-class acc 0.5991 (not the headline).

```
[[850  85   3  80  44   0]
 [346 139   1  10   5   0]
 [ 10   2  44  56   4   0]
 [ 61   5  13 329  30   0]
 [138   8   0  36 110   0]
 [ 42   5   0   0   1   0]]
```

---

## 16. Section 8 baseline + end-to-end cost (45 holdout graphs, seed 0)

Compiled from existing logs — **not re-run**. Static systems: `results/matrix_static_systems.jsonl` / `logs/matrix_static.log`. RPI: `results/matrix_rpi_hopshell.jsonl` / `logs/matrix_rpi.log`, loaded `**radius_gnn_hopshell.pt`** (the 0.7005 sequential hop-shell checkpoint). Pilot `logs/pilot_matrix.log` used a stale pre-§14 `radius_gnn.pt` and is discarded.

Identical sequential edit streams (targeted r*=1 edits are label-only). **2,025** edits (45 graphs × 45 stream steps). **2,007** join to sequential labels (18 unlabeled = infeasible-after-edit discards). Fixed-radius uses **r=2** (Section 8.1). Majority = always start at r=0.

### 16.1 Comparison table


| system                     | n edits | valid  | fallback | mean ms/edit | mean touched frac | mean vertices touched | wasted attempts | classifier acc    |
| -------------------------- | ------- | ------ | -------- | ------------ | ----------------- | --------------------- | --------------- | ----------------- |
| full_recompute             | 2025    | 0.9486 | 0.0000   | 5.6          | 0.1168            | 29.70                 | 0.00            | —                 |
| fixed_radius (r=2)         | 2025    | 0.9486 | 0.5802   | 54.9         | 0.1834            | 44.35                 | 2.51            | —                 |
| majority (r=0)             | 2025    | 0.9491 | 0.6805   | 11.0         | 0.1097            | 27.85                 | 2.05            | 0.5291 (always-0) |
| **rpi_dsatur (hop-shell)** | 2025    | 0.9491 | 0.6760   | 99.2         | **0.1107**        | **27.96**             | **2.04**        | **0.7005**        |


Touched fraction = `vertices_touched / n` (original graph size). Wasted attempts = mean Section 7 retry count (`attempts`). Validity ~0.949 on **all** systems including full recompute → infeasible-after-edit, not an RPI merge bug. Conditional on a successful coloring, merge/boundary validity holds.

Per size, RPI fallback tracks majority and diverges from fixed-r=2 as n grows (n=500: RPI=majority fallback **0.7179**, fixed **0.8889**).

### 16.2 End-to-end cost vs oracle (Section 7 / 9)

Joined RPI/fixed/majority outcomes to sequential labels on the 45 holdout graphs (n=2007 matched). Oracle radius = labeled bucket / `r_star`. Oracle that starts at the true bucket needs 0 retries; **wasted retry steps vs oracle = mean `attempts`**.


| system               | matched | mean wasted retries vs oracle | fallback (matched) | mean overshoot (start−oracle, ≥0) |
| -------------------- | ------- | ----------------------------- | ------------------ | --------------------------------- |
| full_recompute       | 2007    | 0.000                         | 0.0000             | n/a (always full)                 |
| fixed_radius r=2     | 2007    | 2.502                         | 0.5770             | 1.084                             |
| majority r=0         | 2007    | 2.044                         | 0.6781             | 0.000                             |
| rpi_dsatur hop-shell | 2007    | **2.030**                     | 0.6736             | 0.013                             |


Section 9 expected-pattern check (honest):


| metric           | expected if method works    | observed                                                                                               |
| ---------------- | --------------------------- | ------------------------------------------------------------------------------------------------------ |
| Validity         | 0 conflicts for all systems | ~0.949 success; failures match full recompute (infeasible edits)                                       |
| Wall-clock       | RPI < full; RPI ≤ fixed     | **RPI slower** (99.2 ms vs full 5.6 vs fixed 54.9) — GNN + retry ladder dominate DSATUR on these sizes |
| Vertices touched | RPI < full                  | **RPI 27.96 < full 29.70** and **≪ fixed 44.35**                                                       |
| Radius accuracy  | ≫ majority                  | **0.7005 vs 0.5291** on sequential test                                                                |
| Fallback rate    | <15%                        | **not met**: RPI 67.6%, fixed 58.0%, majority 68.1%                                                    |


**Main finding:** GNN-predicted radius **beats fixed-radius r=2 on local work** (touched fraction 0.111 vs 0.183, −40%; vertices touched 27.96 vs 44.35, −37%) and on **wasted retries vs oracle** (2.03 vs 2.50). It does **not** beat fixed-radius on fallback (−9.6pp) or latency. End-to-end, the accuracy-selected hop-shell checkpoint behaves like majority (r=0): it wins the cheap r*=0 first-try cases that fixed r=2 misses, and misses the high-bucket recall that would keep the ladder under `MAX_RETRIES`. Guarded-accuracy selection improved 6-class acc and **regressed** fallback vs the stale pilot (0.597 → 0.676).

Caveat (label vs incremental feasibility): even oracle bucket 0 has ~53% RPI fallback. `r*` is a full-DSATUR color-diff, not “minimum radius at which incremental DSATUR succeeds,” so Section 9’s 15% fallback target is not a fair readout of classifier error alone.

---

## Presentation Summary (Final)

- **Problem.** Online list coloring: each edit (shrink-list / remove-vertex) must be repaired from the current coloring only; full DSATUR from scratch wastes work when the damage is local. Predict a repair radius `r̂`, recolor the `r̂`-ball with list-DSATUR, retry up the ladder, fall back to full recompute.
- **Method (RPI-DSATUR).** A 4-layer GATv2 GNN on the 4-hop window around the edit predicts a 6-class radius bucket `{0,1,2,3,4,full}`; inference **rounds up** and retries `0→1→2→3→4→full` (max 3) then full DSATUR. Hop-shell pooling (mean per hop d=0..4, concat) + 10-d globals (structure + post-edit slack).
- **Dataset.** 294 graphs (n=100/200/500), 13,103 sequential labeled edits, 45-graph holdout (2,007 sequential test edits / 2,025 streamed edits). Class 1 is 2.2% naturally; +2,935 targeted r*=1 one-offs (label-only) raise it to 20% for the merged operating point — sequential eval streams are unchanged.
- **Headline metric.** Merged-{0,1} accuracy **0.7745** (r1 hop-shell; a `{0,1}` decode is radius 1, safe under round-up). Not a 6-class number.
- **Secondary 6-class (canonical trained / matrix).** **0.7005** sequential hop-4 hop-shell (`radius_gnn_hopshell.pt`). Majority 0.5291. 72% not reached.
- **Decode overlays (not deployed).** Hop-4 val-tuned thresholds **0.7030** (4→0 147→154). Hop-5 raw **0.6956** (failed vs 0.7005). Hop-5 thresholds **0.7065** (best 6-class *decode*; 4→0 still 148). All < 0.72.
- **Failed pushes to 72%.** Asymmetric ordinal CE **0.6811** (class-4 R 0.445). Shell-density globals **0.6796**. Hop-5 retrain **0.6956**. 4→0 cell stays **146–154** (~7.3pp) at hop 4 and hop 5.
- **Baseline (45 holdout graphs, hop-shell ckpt).** Fallback / mean ms / touched-frac / wasted-attempts: full 0.00 / 5.6 / 0.117 / 0.00; fixed r=2 **0.580 / 54.9 / 0.183 / 2.51**; majority **0.681 / 11.0 / 0.110 / 2.05**; RPI **0.676 / 99.2 / 0.111 / 2.04**. Classifier acc 0.7005 vs majority 0.5291.
- **Main finding.** GNN radius **beats fixed-radius on local work** (touched fraction −40%, vertices touched −37%) and on wasted retries vs oracle (2.03 vs 2.50). It does not beat fixed-radius on fallback or wall-clock; end-to-end it tracks majority (r=0).
- **Limitations.** Class 1 is a semantics gap (verdict B), not missing slack: natural-stream recall 0.000, scalar ceiling ~0.21. Hop-4 *and* hop-5 windows cannot separate 4 vs 0. Fallback ≫ 15% because `r`* ≠ incremental feasibility. RPI wall-clock > full recompute on n≤500.
- **Best config (deployed).** Hop 4, hop-shell pooling, guarded val-accuracy (macro-F1 ≥ 0.40), ordinal CE+MAE λ=0.5, sampler, no class weights, 10-d globals, `labels_pre_aug_backup.pt`. Paper: lead with merged-{0,1} 0.7745; report 6-class 0.7005 honestly; do not claim class-1 competence, a 72% 6-class result, or a speed win vs full DSATUR at these sizes.

---

## 18. Asymmetric ordinal loss + shell density (6-class push)

Attempt to raise hop-shell 6-class test acc **0.7005 → 0.72**. Sequential §14 labels, test n=2007. Hop-shell pooling, guarded val-acc (macro-F1 ≥ 0.40), sampler, no class weights. **72% not met.** Best 6-class remains **0.7005**.

### 18.1 Experiment 1 — `asymmetric_ordinal` (CE + expected cost)

`cost[t,p] = 2*max(0,t−p) + |t−p|`. Config `model/config_asymmetric.yaml`. Labels `labels_pre_aug_backup.pt`. Early stop epoch 58.


|                | no bias | with val-fit bias |
| -------------- | ------- | ----------------- |
| test acc       | 0.5561  | **0.6811**        |
| macro-F1       | —       | 0.4460            |
| class-4 recall | —       | **0.445**         |


Confusion (biased; rows=true, cols=pred):

```
[[875  0  8  76  62  41]
 [ 34  0  2   9   6   0]
 [ 12  0 55  45   4   0]
 [ 62  0 33 295  48   0]
 [136  0  0  21 130   5]
 [ 34  0  0   0   2  12]]
```

Top cells: 4→0 **136** (was 147), 0→3 **76**, 0→4 **62**. Bias fitted `[-2,-1.9,-2,-2]` on classes 2–full — recovers acc by undoing the over-prediction the loss spent the run creating. Class-4 recall 0.445 vs hop-shell 0.298; acc **−1.94pp**.

### 18.2 Fallback — +5 shell tightness means (globals 10→15)

Mean tightness per hop shell d=0..4, appended onto existing hop-4 labels (`data/labels/labels_shell_density.pt`, n=13103, test 2007). Same loss. Early stop epoch 41.


|                | no bias | with val-fit bias |
| -------------- | ------- | ----------------- |
| test acc       | 0.5541  | **0.6796**        |
| macro-F1       | —       | 0.4504            |
| class-4 recall | —       | **0.438**         |


Top cells: 4→0 **139**, 0→3 **75**, 3→0 **64**. Same failure mode as 18.1.

Reports: `model/reports/asymmetric/train_report.json`, `model/reports/asymmetric_shell/train_report.json`.

---

## 19. Experiment log — threshold decode + hop-5 (2026-09-03)

Attempt to raise hop-shell 6-class test acc **0.7005 → 0.72**. Sequential §14 labels, test n=2007, same 45-graph holdout. **72% not met.** Official best *trained* 6-class remains **0.7005**; best merged-{0,1} remains **0.7745**. Peak decode (hop-5 + val thresholds **0.7065**) still misses 72% and drops macro-F1 to **0.399** (below the 0.40 selection guard), so it is not a replacement checkpoint.

### 19.0 One row per lineage (authoritative)


| lineage                            | labels    | selection              | pool                | 6-class    | merged-{0,1} | class-1 R | 4→0     | verdict                                       |
| ---------------------------------- | --------- | ---------------------- | ------------------- | ---------- | ------------ | --------- | ------- | --------------------------------------------- |
| §13 CE + √inv-freq + sampler       | seq 13103 | macro-F1               | mean                | 0.5610     | —            | 0.020     | —       | double-rebalance hurt                         |
| §14 slack, class weights **off**   | seq       | macro-F1               | mean                | 0.6931     | —            | 0.000     | 138     | baseline                                      |
| guarded acc + logit bias           | seq       | val-acc, F1≥0.40       | mean                | 0.7000     | 0.7175       | 0.000     | 146     | no-bias 0.7010; bias slightly hurt            |
| **hop-shell (canonical 6-class)**  | seq       | val-acc, F1≥0.40       | hop-shell           | **0.7005** | 0.7180       | 0.000     | 147     | best *trained* 6-class; matrix ckpt           |
| joint; fallback macro-F1           | seq       | joint infeasible       | mean                | 0.7000     | 0.7170       | 0.000     | 144     | class-1 R=0 ⇒ fallback                        |
| r1 CE + macro-F1                   | r1 16038  | macro-F1               | mean                | 0.5438     | 0.7253       | **0.685** | 99      | class-1 wins, acc dies                        |
| r1 guarded + bias                  | r1        | val-acc, F1≥0.40       | mean                | 0.5735     | 0.7705       | 0.004     | 144     | acc-optimal kills class 1                     |
| **r1 hop-shell (headline merged)** | r1        | val-acc, F1≥0.40       | hop-shell           | 0.5991     | **0.7745**   | 0.277     | 138     | headline operating point                      |
| r1 joint + bias                    | r1        | acc | R1≥0.30, F1≥0.40 | mean                | 0.6064     | 0.7713       | 0.387     | 134     | best r1 6-class                               |
| r1 CE + post-hoc bias only         | r1        | val-acc bias           | mean                | 0.5832     | 0.7599       | 0.289     | 143     | no retrain; below r1 hop-shell                |
| §18 **asymmetric ordinal CE**      | seq       | val-acc, F1≥0.40       | hop-shell           | 0.6811     | 0.6981       | 0.000     | 136     | **FAILED** (−1.94pp); class-4 R 0.445         |
| §18 shell tightness 10→15-d        | seq       | val-acc, F1≥0.40       | hop-shell           | 0.6796     | 0.6966       | 0.000     | 139     | **FAILED**                                    |
| §19 hop-4 threshold (no retrain)   | seq       | val-acc `argmax(z−t)`  | hop-shell ckpt      | 0.7030     | 0.7205       | 0.000     | **154** | +0.25pp; 4→0 worse; 72% no                    |
| §19 hop-5 ordinal (epoch 15 ckpt)  | seq hop5  | val-acc; **no bias**   | hop-shell, 5 layers | 0.6956     | 0.7130       | 0.000     | 146     | **FAILED** vs 0.7005; class-4 R 0.428         |
| §19 hop-5 threshold                | seq hop5  | val-acc `argmax(z−t)`  | hop-5 ckpt          | **0.7065** | 0.7240       | 0.000     | 148     | best 6-class *decode*; F1 0.399; not deployed |


### 19.1 Experiment A — per-class thresholds on frozen hop-shell (no retrain)

Checkpoint `model/checkpoints/radius_gnn_hopshell.pt`, labels `labels_pre_aug_backup.pt`. `pred = argmax(logits − t)` fit on **val accuracy** via a joint 2D grid on `(t[0], t[4])` then coordinate descent (class 0 not pinned). Script: `model/eval_thresholds.py`. Report: `model/reports/hop_shell/threshold_eval.json`.

`t = [−1.6, 0, 0.1, 0, −0.7, 0]`. Val acc 0.7086 → 0.7157.


|                          | zero t                        | tuned t                           |
| ------------------------ | ----------------------------- | --------------------------------- |
| test acc                 | 0.6986                        | **0.7030**                        |
| macro-F1                 | 0.4019                        | 0.4125                            |
| class-0 / 1 / 4 / full R | 0.874 / 0.000 / 0.298 / 0.000 | 0.894 / 0.000 / **0.397** / 0.000 |
| 4→0                      | **147**                       | **154**                           |
| merged-{0,1}             | 0.7160                        | 0.7205                            |


Confusion (tuned; rows=true, cols=pred):

```
[[949  0   3  73  37   0]
 [ 35  0   2   9   5   0]
 [ 13  0  53  47   3   0]
 [ 66  0  22 293  57   0]
 [154  0   0  22 116   0]
 [ 47  0   0   0   1   0]]
```

Gains from 0→3 (96→73) and 4→3 (58→22), not 4→0 (147→154, worse). **+0.25pp vs 0.7005. 72% not met.**

### 19.2 Experiment B — hop-5 window + one retrain

`--hop-radius 5 --skip-targeted` → `data/labels/labels_hop5.pt`. Same class balance as hop-4 sequential (`0:6830, 1:292, 2:931, 3:2875, 4:1873, full:302`), n=13103, test **2007**. Feature window only changed; buckets still `{0,1,2,3,4,full}` with `r*≥5 → full`. Hop-shell pooling with 6 shells (0..5), `num_layers=5`. Config `model/config_hop5.yaml`. `tune_class_bias: false`.

Training: best guarded val-acc at **epoch 15** (val 0.7071, macro-F1 0.4431). Last logged epoch 33, no later val-acc best. Process died at ~25 min (Windows exit `4294967295`), no `train_report.json`. Patience 20 would have stopped at epoch 35 on the same checkpoint. Test numbers below are post-hoc `eval_thresholds` on `model/checkpoints/hop5/radius_gnn_hop5.pt`. Report: `model/reports/hop5/threshold_eval.json`.


|                          | no bias (selection)               | val-fit thresholds            |
| ------------------------ | --------------------------------- | ----------------------------- |
| test acc                 | **0.6956**                        | 0.7065                        |
| macro-F1                 | 0.4120                            | **0.3991** (below 0.40 guard) |
| class-0 / 1 / 4 / full R | 0.878 / 0.000 / **0.428** / 0.000 | 0.887 / 0.000 / 0.346 / 0.000 |
| 4→0                      | **146**                           | 148                           |
| merged-{0,1}             | 0.7130                            | 0.7240                        |


Confusion (no bias):

```
[[932  0   7  73  48   2]
 [ 35  0   2   9   5   0]
 [ 12  0  61  39   4   0]
 [ 65  0  39 278  56   0]
 [146  0   0  21 125   0]
 [ 47  0   0   0   1   0]]
```

Class-4 recall rose 0.298→0.428 via 4→3 (58→21), **not** 4→0 (147→146). Acc **−0.49pp vs 0.7005**. Hop 5 does not separate 4 vs 0. **72% not met.**

### 19.3 Comparison vs hop-shell 0.7005


| system                    | 6-class acc | macro-F1 | 0 / 1 / 4 / full R            | 4→0 | vs 0.7005              |
| ------------------------- | ----------- | -------- | ----------------------------- | --- | ---------------------- |
| hop-shell (official best) | **0.7005**  | 0.4049   | 0.874 / 0.000 / 0.298 / 0.000 | 147 | —                      |
| A: hop-4 + thresholds     | 0.7030      | 0.4125   | 0.894 / 0.000 / 0.397 / 0.000 | 154 | +0.25pp                |
| B: hop-5 (epoch-15 ckpt)  | 0.6956      | 0.4120   | 0.878 / 0.000 / 0.428 / 0.000 | 146 | −0.49pp                |
| B + thresholds            | 0.7065      | 0.3991   | 0.887 / 0.000 / 0.346 / 0.000 | 148 | +0.60pp, F1 guard fail |


4→0 stays **146–154** (~7.3pp) under threshold decode, hop-5 window, and §18 asymmetric/shell-density. Information-bound, not optimization-bound.

**Ceiling (current 6-class hop-shell approach):** ~70.0–70.7% on this test set. Reaching 72% needs ~+39 net correct of 2007 without new 0-false-positives. That mass sits in 4→0, which hop 5 can see and still cannot move.

### 19.4 Methodology decisions (locked)


| knob                     | value                                                                                                                                                                                          | why                                                                                                                                   |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Bucket scheme            | 6-class linear `{0,1,2,3,4,full}`                                                                                                                                                              | geometric `{0,1,2,4,8,full}` wasted classes on ER graphs with r* ≲ 5                                                                  |
| Label `r*`               | verdict B: BFS color-diff radius between two full DSATUR runs                                                                                                                                  | **not** slack, **not** “min radius where incremental DSATUR succeeds”                                                                 |
| Hop radius               | **4 primary**; 5 tested                                                                                                                                                                        | hop 5: class-4 R 0.298→0.428, acc 0.7005→0.6956, 4→0 unchanged                                                                        |
| Encoder                  | GATv2, 4 heads, hidden 64, dropout 0.2, `num_layers` = hop radius                                                                                                                              |                                                                                                                                       |
| Pooling                  | **hop-shell** (mean per hop, concat)                                                                                                                                                           | best trained 6-class; mean-pool is the 0.6931 / guarded 0.7000 line                                                                   |
| Globals                  | **10-d** = 4 structural (`degree_centrality, is_articulation, avg_degree_norm, density`) + 6 slack (`slack_v, slack_v_is_zero, list_size_v, n_zero_slack_nbrs, min_nbr_slack, mean_nbr_slack`) | `slack_v_is_zero` identically 0 on all 13,103 sequential labels — **dead**. No more slack variants. Shell-density 15-d failed (§18.2) |
| Per-node `x`             | dim 7                                                                                                                                                                                          |                                                                                                                                       |
| Train sampler            | `WeightedRandomSampler` **train-only**                                                                                                                                                         |                                                                                                                                       |
| `class_weighting`        | **none**                                                                                                                                                                                       | sampler + √inv-freq → §13 0.5610                                                                                                      |
| Loss (best 6-class)      | ordinal CE+MAE λ=0.5                                                                                                                                                                           | `asymmetric_ordinal` failed 0.6811                                                                                                    |
| Selection (6-class)      | guarded `val_accuracy`, macro-F1 floor **0.40**                                                                                                                                                | raw acc collapses to all-0 (~0.53)                                                                                                    |
| Selection (r1 / class 1) | joint: acc subject to class-1 R ≥ 0.30 and F1 ≥ 0.40                                                                                                                                           | infeasible on sequential (class-1 R stays 0)                                                                                          |
| Logit bias               | hop-shell `[0,0,−0.1,0,0,0]` (0.6986→0.7005); destructive on asymmetric (`≈−2` on classes 2–full)                                                                                              | do not pair with asymmetric loss                                                                                                      |
| Split                    | graph_id 70/15/15, seed 0                                                                                                                                                                      | never split by edit                                                                                                                   |
| `batch_size` / lr        | 32 / 1e-3                                                                                                                                                                                      | 64 hung; 1e-5 collapsed                                                                                                               |


Retry ladder: `0→1→2→3→4→full`, `MAX_RETRIES=3`.

### 19.5 Current best model card

**Deployed 6-class (Section 8 matrix).** `model/checkpoints/radius_gnn_hopshell.pt`. Hop 4, hop-shell, ordinal λ=0.5, guarded val-acc, sampler, no class weights, 10-d globals. Labels `data/labels/labels_pre_aug_backup.pt`. Report `model/reports/hop_shell/train_report.json`.

- Test acc **0.7005** (no-bias 0.6986), macro-F1 **0.4049**, majority 0.5291 (+17.1pp)
- Under-pred 0.1908 / over-pred 0.1086
- Canonical CM + per-class table: **§15.1** (one CM; do not duplicate)

**Headline merged-{0,1} (historical; r1 artifacts removed in §20 cleanup).** Was **0.7745** on r1-augmented labels (16,038 edits). CM: **§15.2**. Not deployed; final model is sequential hop-shell only.

**Section 8** — 45 holdout graphs, seed 0, RPI loaded `radius_gnn_hopshell.pt` (full table §16.1):


| system            | n    | valid  | fallback | ms/edit | touched frac | verts touched | wasted att. | clf acc    |
| ----------------- | ---- | ------ | -------- | ------- | ------------ | ------------- | ----------- | ---------- |
| full_recompute    | 2025 | 0.9486 | 0.0000   | 5.6     | 0.1168       | 29.70         | 0.00        | —          |
| fixed_radius r=2  | 2025 | 0.9486 | 0.5802   | 54.9    | 0.1834       | 44.35         | 2.51        | —          |
| majority r=0      | 2025 | 0.9491 | 0.6805   | 11.0    | 0.1097       | 27.85         | 2.05        | 0.5291     |
| **rpi hop-shell** | 2025 | 0.9491 | 0.6760   | 99.2    | **0.1107**   | **27.96**     | **2.04**    | **0.7005** |


### 19.6 Process / reproduction

**Active labels (post-§20 cleanup):** `data/labels/labels.pt` = sequential hop-4, n=13,103. `model/config.yaml` `data.labels_path` points here. No override needed.


| file        | contents              | n     | use for                          |
| ----------- | --------------------- | ----- | -------------------------------- |
| `labels.pt` | sequential hop 4 only | 13103 | **final model, matrix join, train** |


Rebuild labels (sequential hop-4):

```
.\.venv\Scripts\python.exe -m labels.build_labels --graphs-dir data/raw/graphs --edits-dir data/raw/edits --out data/labels/labels.pt --hop-radius 4 --skip-targeted
```

Train final model:

```
.\.venv\Scripts\python.exe -m model.train model/config.yaml
```

Threshold eval (no retrain):

```
.\.venv\Scripts\python.exe -m model.eval_thresholds --config model/config.yaml --checkpoint model/checkpoints/radius_gnn_hopshell.pt --labels data/labels/labels.pt --out model/reports/hop_shell/threshold_eval.json
```

Section 8 RPI matrix:

```
.\.venv\Scripts\python.exe -m experiments.run_matrix --systems rpi_dsatur --seeds 0 --model-checkpoint model/checkpoints/radius_gnn_hopshell.pt --model-config model/config.yaml --test-graph-ids-path model/reports/hop_shell/test_graph_ids.json --out results/matrix_rpi_hopshell.jsonl
```

Which labels for which metric: **6-class 0.7005 / matrix** → `data/labels/labels.pt`.

### 19.7 Known limitations and open questions

- **Class 1 / verdict B.** `r*=0` is “no color diff at dist>0 between two independent full DSATUR runs,” not `slack_v==0`. Natural-stream recall 0.000 (51 test). Scalar ceiling ~0.20–0.21. Targeted r*=1 extras are learnable (R 0.39–0.69) but accuracy-optimal rules spend those wins on 0↔1 swaps.
- **4→0 information bound.** 134–154 test errors (~6.7–7.7pp) at hop 4 *and* hop 5. Loss, pooling, asymmetric CE, shell density, and thresholds all leave this cell intact.
- **72% not met.** Best trained 6-class **0.7005**; best decode **0.7065**. Honest ceiling of the current approach ~70.0–70.7%.
- `**r*` ≠ incremental feasibility.** Oracle bucket 0 still has ~53% RPI fallback. Section 9’s 15% fallback target is not a fair classifier-error readout.
- **RPI slower than full recompute** on n≤500 (99.2 vs 5.6 ms/edit). Local-work win is real; wall-clock win is not.
- **Thresholds / hop-5 decode not in the inference path.** Matrix uses hop-shell weights; checkpoint does not store bias/thresholds.
- **Gaps.** No 5-seed Wilcoxon (seed 0 only). Historical experiment artifacts removed in §20.

---

## §20 Repository cleanup (model finalized)

**Date:** 2026-09-03. User finalized training at **6-class test acc 0.7005**. Removed failed/intermediate experiment artifacts (~2GB+). Raw graphs and edits under `data/raw/` unchanged.

### Kept (canonical)

| path | purpose |
|------|---------|
| `model/checkpoints/radius_gnn_hopshell.pt` | **final model** (0.7005) |
| `model/checkpoints/radius_gnn.pt` | copy of hopshell for default inference paths |
| `data/labels/labels.pt` | sequential hop-4 labels, n=13,103 |
| `model/config.yaml` | sole training config |
| `model/reports/hop_shell/` | train_report, test_graph_ids, threshold_eval |
| `results/matrix_*.jsonl` | Section 8 baseline outputs |
| `logs/train_hopshell.log`, `matrix_rpi.log`, `matrix_static.log` | reference logs |
| `Context.md`, `README.md`, source code, `data/raw/` | research + reproduction |

### Removed

- **Labels:** `labels_r1_aug.pt`, `labels_hop5.pt`, `labels_shell_density.pt`, `labels_pre_aug_backup.pt` (merged into `labels.pt`)
- **Checkpoints:** all non-hopshell weights (r1, joint, asymmetric, hop5, stale ordinal, decision_rule)
- **Configs:** `config_asymmetric*.yaml`, `config_hop5.yaml`, `config_r1*.yaml`, `config_s14_joint.yaml`
- **Reports:** all experiment report dirs except `hop_shell/`
- **Logs:** all except hopshell + matrix logs
- **Misc:** `IMPLEMENTATION.md`, `IMPLEMENTATION_NEXT.md`, `test.py`, `__pycache__/`, `.pytest_cache/`

Historical metrics from removed runs remain documented in §13–19 above.

