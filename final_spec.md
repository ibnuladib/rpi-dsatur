# Prompt for coding agent: RPI-DSATUR additional analysis + results notebook

You are extending an existing, working RPI-DSATUR codebase. Do not re-derive experimental results that already exist — the raw data described below is real, already generated, and must be treated as ground truth. Your job is (1) four specific new computations, and (2) one Jupyter notebook that produces all figures/tables listed in Part 2, reading from existing + newly-produced data. Follow file paths and function signatures exactly; this codebase has repeatedly broken from small path/naming mismatches between files, so treat every constant and path below as load-bearing, not a suggestion.

---

## Available inputs — read these, do not regenerate them

- `results/matrix_static_systems.jsonl` — full_recompute, fixed_radius(r=2), majority_r=0 outcomes, seed 0, n=100/200/500 holdout.
- `results/matrix_rpi_hopshell.jsonl` — rpi_dsatur (deployed hop-shell GATv2) outcomes, same holdout, same edit streams.
- `results/matrix_n2000_rpi_vs_full.jsonl` — RPI vs full_recompute only, 8 new n=2000 graphs, OOD, seed 0.
- `model/reports/hop_shell/train_report.json` — classifier confusion matrix, per-class metrics, class counts.
- `model/reports/test_graph_ids.json` — the exact held-out graph_ids used; **any new analysis touching "the holdout" must filter to this set, never re-derive it**.
- `data/raw/graphs/*.json`, `data/raw/edits/*_edits.json` — graph/edit-stream source files, needed for Task 1.4's fresh BFS computation.
- Do **not** use `results/raw_outcomes.jsonl` — this is a stale, different RPI run (documented mismatch: mean 68.2ms/fallback 59.7% vs the hop-shell numbers above). If you find it, ignore it.

Each JSONL row is one `EditOutcome`: `{system, graph_id, edit_index, time_ms, vertices_touched, radius_used, attempts, fallback_triggered, valid, seed, size}`. Join on `(graph_id, edit_index, seed)` for paired comparisons — all systems ran on identical edit streams, so this join must always produce complete, aligned rows with no missing pairs. Assert this and fail loudly if it doesn't hold.

---

## Part 1: New computations

### 1.1 — Holm-Bonferroni correction on the existing 6 Wilcoxon tests

File: `experiments/stats_corrections.py`

```python
def holm_bonferroni_correct(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """
    Given raw p-values from multiple hypothesis tests, return Holm-Bonferroni
    adjusted decisions. Sort ascending, adjust as
    p_adj_i = max(p_adj_1..i, p_i * (n - rank + 1)), enforce monotonicity.
    Return one dict per input p-value (ORIGINAL order preserved), each with:
      {"p_raw": float, "p_holm": float, "reject_at_0.05": bool, "rank": int}
    """
```

Apply this to the 6 existing tests (RPI-full/fixed/majority × time_ms/vertices). Do not recompute the Wilcoxon statistics themselves — those already exist (median Δ, mean Δ, p, r_rb reported previously); only add the Holm-adjusted p-value and rejection decision as new columns onto that existing table.

### 1.2 — Graph-level (not edit-level) paired significance test

File: `experiments/graph_level_stats.py`

```python
def aggregate_per_graph(df: "pd.DataFrame", metric: str, system: str) -> "pd.Series":
    """
    df has columns: system, graph_id, edit_index, seed, <metric>.
    Return one value per graph_id: the MEAN of <metric> across that graph's
    edits, for the given system. Index = graph_id.
    """

def paired_graph_level_wilcoxon(
    df: "pd.DataFrame", metric: str, system_a: str, system_b: str
) -> dict:
    """
    Aggregate both systems to one value per graph (via aggregate_per_graph),
    align on graph_id (inner join -- both systems must have every graph_id,
    assert this), then run scipy.stats.wilcoxon on the paired per-graph
    values. This directly addresses pseudoreplication: 2,025 edit-level
    pairs come from only 45 graphs, so edit-level significance overstates
    effective sample size. This test's n is the number of unique graph_ids
    (should be 45 for the n<=500 holdout).
    Return: {"n_graphs": int, "median_delta": float, "mean_delta": float,
             "p": float, "statistic": float}
    """
```

Run this for the same 3 system-pairs × 2 metrics as the edit-level tests (6 tests total), separately. Report both edit-level and graph-level results side by side in the final table — do not replace one with the other.

### 1.3 — Fixed-radius sweep at r=1, 3, 4

File: `experiments/run_fixed_radius_sweep.py`

Reuse `inference.rpi_dsatur.fixed_radius_step` exactly as-is (it already accepts a `radius` parameter) — do not reimplement this logic. Reuse `experiments/run_matrix.py`'s `run_system`/graph-loading machinery; do not duplicate it. Concretely:

```python
def run_fixed_radius_sweep(
    radii: list[int],
    graphs_dir: str = "data/raw/graphs",
    edits_dir: str = "data/raw/edits",
    test_graph_ids_path: str = "model/reports/test_graph_ids.json",
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
```

Run with `radii=[1, 3, 4]`. (r=2 already exists in `results/matrix_static_systems.jsonl` — do not rerun it, just include it when building the sweep summary table by reading the existing file and setting `radius=2` for those rows.)

### 1.4 — Hop-window coverage fraction vs. graph size n

File: `experiments/hop_coverage_analysis.py`

This is the one genuinely new measurement — it substantiates the claim "a 4-hop window nearly covers the whole graph at every tested n" with actual data instead of an assertion.

```python
def compute_hop_coverage_fractions(
    graphs_dir: str = "data/raw/graphs",
    test_graph_ids_path: str = "model/reports/test_graph_ids.json",
    n2000_graphs_dir: str = "data/raw/graphs_n2000",  # adjust to actual path used for the OOD graphs
    hop_radius: int = 4,
    sample_vertices_per_graph: int = 20,
) -> "pd.DataFrame":
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
```

Save the raw per-vertex results to `results/hop_coverage.csv`. This feeds Figure 2 in Part 2 directly (mean ± IQR of `coverage_fraction` grouped by `n`).

**If the n=2000 OOD graphs are not stored under a discoverable path, do not guess a location — search the repo for how `results/matrix_n2000_rpi_vs_full.jsonl` was originally produced (there must be a script or the graph files themselves referenced by `graph_id` matching pattern `n2000_s2000*`) and use the same source. If genuinely unavailable, skip the n=2000 rows in this analysis and note explicitly in the notebook output that OOD coverage could not be computed, rather than fabricating placeholder values.**

---

## Part 2: Notebook — `notebooks/results_analysis.ipynb`

Structure: one markdown header cell per figure/table below, followed by the code cell producing it. Save every figure as both a displayed inline plot AND a PNG to `results/figures/<name>.png` (create the directory if missing). Save every summary table as both a displayed DataFrame AND a CSV to `results/tables/<name>.csv`.

Import and reuse existing code — do not recompute what's already in the JSONL files: load them with `pandas.read_json(path, lines=True)`.

| # | Name | Contents | Source |
|---|---|---|---|
| 1 | `fig1_combined_scaling` | Median `time_ms` (log y) vs `n` (log x) ∈ {100,200,500,2000}, two lines: full_recompute, rpi_dsatur. Single combined figure — do not produce this as two separate in-distribution/OOD charts. | matrix_static_systems + matrix_rpi_hopshell + matrix_n2000 |
| 2 | `fig2_hop_coverage_vs_n` | Mean coverage_fraction with IQR error bars, vs `n` (log x axis) | `results/hop_coverage.csv` from Task 1.4 |
| 3 | `fig3_fallback_vs_n` | Fallback rate (%) vs `n`, all 4 systems for n≤500, RPI+full only at n=2000 | same JSONL sources as fig1 |
| 4 | `fig4_vertices_vs_n` | Mean vertices_touched vs `n`, all systems available at each n | same |
| 5 | `fig5_edit_type_breakdown` | Grouped bar: fallback rate AND mean vertices_touched, x-axis = edit_type (shrink_list/remove_vertex), grouped by system. Two subplots side by side. | matrix_static_systems + matrix_rpi_hopshell joined against `data/raw/edits/*` for edit type |
| 6 | `fig6_confusion_heatmap` | Heatmap of the classifier confusion matrix (rows=true, cols=predicted), annotated with raw counts | `model/reports/hop_shell/train_report.json` |
| 7 | `fig7_per_class_recall` | Bar chart, per-class recall, each bar annotated with its support count (n) above it | same train_report.json |
| 8 | `fig8_misprediction_cost` | Two-panel grouped bar/box: time_ms and vertices_touched, grouped by first-try (attempts==0) vs retried (attempts>0), RPI only | matrix_rpi_hopshell |
| 9 | `fig9_fixed_radius_sweep` | Two-panel line plot: median time_ms and fallback rate, x-axis = radius ∈ {1,2,3,4} | `results/fixed_radius_sweep.jsonl` (Task 1.3) + r=2 rows from matrix_static_systems |
| 10 | `table_wilcoxon_full` | The 6 existing Wilcoxon rows, extended with Holm-adjusted p (Task 1.1) AND the parallel graph-level test columns (Task 1.2) side by side | Tasks 1.1 + 1.2 outputs |
| 11 | `table_headline_summary` | The existing 4-system headline table (valid%, fallback%, first-try%, median/mean ms, mean vertices, mean retries), unchanged, just re-rendered from the JSONL for reproducibility | all matrix JSONLs |

### Notebook engineering requirements
- First cell: imports + a `DATA_DIR = "results"` / `FIGURES_DIR = "results/figures"` / `TABLES_DIR = "results/tables"` constants block. Create directories with `os.makedirs(..., exist_ok=True)`.
- Every join between JSONL sources must assert row-count sanity (e.g., after joining full_recompute and rpi_dsatur on `(graph_id, edit_index, seed)`, assert the joined row count equals `min(len(full_df), len(rpi_df))` and log a warning listing any `graph_id`s that failed to join, rather than silently dropping them).
- Use `matplotlib`/`seaborn` only (already implied by the rest of this codebase's stack) — no new plotting library dependencies.
- Every figure needs a title stating what's plotted AND the n (sample size) it's based on, e.g. `"Median time per edit by graph size (n_pairs=2025 for n<=500, n_pairs=160 for n=2000)"` — this directly supports the "state your sample sizes" requirement from the evaluation spec.
- Do not fabricate any data point. If a source file is missing or a join produces zero rows for a planned figure, render a placeholder cell with a clear markdown note ("data unavailable: <reason>") instead of skipping silently or interpolating.

---

## Deliverables checklist for the coding agent

- [ ] `experiments/stats_corrections.py` with `holm_bonferroni_correct`
- [ ] `experiments/graph_level_stats.py` with `aggregate_per_graph`, `paired_graph_level_wilcoxon`
- [ ] `experiments/run_fixed_radius_sweep.py`, run for r∈{1,3,4}, output at `results/fixed_radius_sweep.jsonl`
- [ ] `experiments/hop_coverage_analysis.py`, output at `results/hop_coverage.csv`
- [ ] `notebooks/results_analysis.ipynb` producing all 11 items in Part 2, figures saved to `results/figures/`, tables saved to `results/tables/`
- [ ] No modification to any existing baseline/inference/label-generation logic — this is additive analysis only
