# RPI-DSATUR

GNN-based incremental list coloring for dynamic graphs.

When a graph’s color lists change, a full DSATUR recompute is always valid but often wasteful: most edits only disturb a small neighborhood. **RPI-DSATUR** trains a graph neural net to predict that repair radius, then runs classical list-DSATUR only inside the predicted ball (with a retry ladder and full-graph fallback).

## What the experiments test

Does a learned radius plus local DSATUR beat the obvious baselines on **speed and locality**, without losing a valid list coloring?

| System | Idea |
| --- | --- |
| **Full recompute** | Re-run list-DSATUR on the whole graph after every edit. |
| **Fixed radius** | Same incremental solver as RPI, but always start at radius 2. |
| **RPI-DSATUR** | A 4-layer hop-shell GATv2 predicts a radius bucket `{0,1,2,3,4,full}` and retries `0→1→2→3→4→full`. |

Graphs are Erdős–Rényi with `n ∈ {100, 200, 500}`, plus an out-of-distribution `n=2000` check. Ground-truth radii come from comparing two full DSATUR colorings.

**Holdout headline:** hop-shell GNN test accuracy **0.7005** vs majority **0.5291**. RPI is more local than fixed `r=2`, but it is **not** faster than full DSATUR on these sizes.

## Folder structure

```
baselines/       Classical solvers used everywhere else: list-Greedy, list-DSATUR, validity checks.
common/          Shared types (graphs, edits, color lists).
data/raw/        Generated ER graphs and sequential edit streams.
                 graphs/ + edits/          n=100, 200, 500 (main matrix)
                 graphs_n2000/ + edits_n2000/   OOD probe
data/labels/     Radius-label tensors (gitignored; rebuild with labels.build_labels).
data_gen/        Graph and edit-stream generators.
labels/          Repair-radius label builder (compares two full DSATUR runs).
model/           Radius GNN, training, config, hop-shell checkpoint and reports.
inference/       RPI-DSATUR step: predicted radius, local DSATUR, retry, fallback.
experiments/     Evaluation matrix, hop-coverage helper, stats, figure/table generator.
results/         Saved experiment outputs (JSONL matrices, tables, the 9 analysis figures).
tests/           Unit tests for solvers, generators, labels, GNN, and RPI.
```

Helper scripts under `experiments/`:

- `run_matrix.py` — full / fixed / RPI comparison
- `run_fixed_radius_sweep.py` — fixed start radius in `{1,2,3,4}`
- `hop_coverage_analysis.py` — 4-hop window coverage vs `n`
- `results_analysis.py` — rebuilds `results/figures` and `results/tables`
- `graph_level_stats.py`, `stats_corrections.py` — Wilcoxon / Holm helpers

## Setup

Python 3.10+ recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce

```powershell
# Graphs and edit streams
python -m data_gen generate --sizes 100 200 500 --graphs-per-size 100 --base-seed 0 --out data/raw/graphs
python -m data_gen edits --graphs-dir data/raw/graphs --num-edits 20 --seed 0 --out data/raw/edits

# Radius labels (large .pt file; gitignored)
python -m labels.build_labels --graphs-dir data/raw/graphs --edits-dir data/raw/edits --out data/labels/labels.pt --hop-radius 4 --skip-targeted

# Train
python -m model.train model/config.yaml

# End-to-end matrix
python -m experiments.run_matrix --systems full_recompute fixed_radius rpi_dsatur --seeds 0 --out results/matrix_out.jsonl

# Rebuild analysis figures/tables from saved JSONLs
python -m experiments.results_analysis

# Tests
pytest
```

Deployed checkpoint: `model/checkpoints/radius_gnn_hopshell.pt`.  
Classifier report: `model/reports/hop_shell/train_report.json`.
