# RPI-DSATUR

GNN-based incremental list coloring for dynamic graphs.

When a graph’s color lists change, a full DSATUR recompute is always valid but often wasteful: most edits only disturb a small neighborhood. **RPI-DSATUR** trains a graph neural net to predict that repair radius, then runs classical list-DSATUR only inside the predicted ball (with a retry ladder and full-graph fallback).

## What the experiments test

Does a learned radius plus local DSATUR beat the obvious baselines on **speed and locality**, without losing a valid list coloring?

The matrix compares three systems on sequential edits of Erdős–Rényi graphs (`n ∈ {100, 200, 500}`, plus an out-of-distribution `n=2000` check):

| System | Idea |
| --- | --- |
| **Full recompute** | Re-run list-DSATUR on the whole graph after every edit (correctness ceiling, speed floor). |
| **Fixed radius** | Same incremental solver as RPI, but always start at radius 2. |
| **RPI-DSATUR** | A 4-layer hop-shell GATv2 predicts a radius bucket `{0,1,2,3,4,full}`; inference rounds up and retries `0→1→2→3→4→full`. |

The GNN itself is a 6-class classifier. Ground-truth radii come from comparing two full DSATUR colorings, not from incremental feasibility, so classifier error and end-to-end fallback are reported separately.

**Headline on the holdout stream:** the hop-shell GNN beats always-predict-0 accuracy (test acc **0.7005** vs majority **0.5291**). RPI is more local than fixed `r=2`, but it is **not** faster than full DSATUR on these sizes — GNN inference plus retries dominate the cheap full recompute.

## Setup

Python 3.10+ recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the main pieces

```powershell
# Graphs and edit streams
python -m data_gen generate --sizes 100 200 500 --graphs-per-size 100 --base-seed 0 --out data/raw/graphs
python -m data_gen edits --graphs-dir data/raw/graphs --num-edits 20 --seed 0 --out data/raw/edits

# Radius labels (writes a large .pt file; gitignored)
python -m labels.build_labels --graphs-dir data/raw/graphs --edits-dir data/raw/edits --out data/labels/labels.pt --hop-radius 4 --skip-targeted

# Train the radius GNN
python -m model.train model/config.yaml

# End-to-end system matrix
python -m experiments.run_matrix --systems full_recompute fixed_radius rpi_dsatur --seeds 0 --model-checkpoint model/checkpoints/radius_gnn_hopshell.pt --model-config model/config.yaml --out results/matrix_rpi_hopshell.jsonl

# Tests
pytest
```

Deployed checkpoint: `model/checkpoints/radius_gnn_hopshell.pt`. Training report: `model/reports/hop_shell/train_report.json`. Paper figures live in `paper_figures/` and `notebooks/`.

## Repo layout

```
baselines/     list-Greedy and list-DSATUR
data_gen/      ER graphs + list + edit streams
labels/        repair-radius label builder
model/         radius GNN, configs, reports, checkpoints
inference/     RPI-DSATUR retry / fallback
experiments/   evaluation matrix
results/       system comparison tables and jsonl
tests/         unit tests
```
