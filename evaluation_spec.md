# RPI-DSATUR Evaluation Specification
## Section 9/10 of the methodology — for direct use by a coding agent

This spec has two parts: (1) exactly what to compute and how, for the evaluation section of the paper; (2) exactly which graph sizes to test on, and why, including compute-budget guidance. Follow both precisely — the statistical validity of the results depends on getting the mechanics right, not just "close enough."

---

## Part 1: What the evaluation must prove

The evaluation is not a classifier-accuracy report — that's Section 6. This is a **system-level** comparison answering one question: *does predicting a repair radius and running DSATUR only within it beat full recompute and a naive fixed radius, on speed and locality, without sacrificing validity?*

Every table and every plot must place the three systems side by side, always in the same order:

1. `full_recompute` — ceiling on correctness/simplicity, floor on speed.
2. `fixed_radius` — isolates whether the *learned* radius adds value over a constant guess.
3. `rpi_dsatur` — the actual contribution.

If `rpi_dsatur` doesn't beat #1 on speed, there is no contribution. If it doesn't beat #2, the GNN isn't earning its keep — a fixed guess would have done as well. Structure the whole results section around where `rpi_dsatur` lands relative to both.

### 1.1 Metrics — what to compute and why each one matters

| Metric | Computation | What it proves |
|---|---|---|
| **Validity rate** | `mean(valid)` per system | Precondition, not a result — must be 100% for all three. If not 100% for `rpi_dsatur`, that's a bug in the merge/boundary logic, report it as such, don't average it away. |
| **Wall-clock time per edit** | `time_ms` — report **median and IQR**, not just mean | Headline speed claim. Report median because edit costs are right-skewed (a few expensive "large-radius" edits will otherwise dominate a mean). |
| **Vertices touched per edit** | `vertices_touched` — median and IQR | Locality claim, distinct from speed — proves the system is actually "incremental," not just coincidentally fast. |
| **Fallback rate** | fraction of `rpi_dsatur` (and `fixed_radius`) rows with `fallback_triggered=True` | The single most important diagnostic number. Feature it prominently. Directly answers "how often did the learned radius fail, requiring the safety net." |
| **Radius-prediction accuracy** | fraction of `rpi_dsatur` rows where `attempts == 0` (predicted radius worked on the first try) | Connects Section 6's classifier results to Section 7/8's system results in one place. |

### 1.2 Statistical testing — exact procedure

For each pair (`rpi_dsatur` vs `full_recompute`, `rpi_dsatur` vs `fixed_radius`), on each metric (`time_ms`, `vertices_touched`), separately per graph size:

1. **Pair by (graph_id, edit_index, seed)** — all three systems ran on identical edit streams, so this pairing is valid and is what makes the test powerful (it controls for edit-to-edit variance, e.g. some edits are intrinsically expensive for every system).
2. **Use Wilcoxon signed-rank test** (`scipy.stats.wilcoxon`), not a paired t-test — timing/vertices-touched data is right-skewed, not normal; a t-test here is a real methodological weakness a reviewer will flag.
3. **Report effect size alongside p-value**: median of the paired differences, and/or rank-biserial correlation. With hundreds to thousands of paired observations, even a trivial difference will be "significant" — the effect size is what tells a reader whether it's practically meaningful.
4. **State the exact n** (number of paired observations) in every reported result — a bare p-value with no n is close to meaningless.

### 1.3 Required breakdowns — do not report only one pooled number

- **By graph size** (100/200/500, see Part 2): expect `rpi_dsatur`'s advantage over `full_recompute` to *grow* with size, since full recompute cost scales with graph size while local repair should scale much more slowly. Showing this trend is a stronger, more generalizable claim than a single aggregate.
- **By prediction correctness**: split `rpi_dsatur` rows into `attempts == 0` (first-try success) vs `attempts > 0` (needed escalation/fallback). Report time/vertices-touched separately for each group — this quantifies the concrete cost of misprediction rather than leaving it as an abstract caveat.
- **By edit type** (`shrink_list` vs `remove_vertex`): report separately, don't pool — they have different structural effects on the graph and may behave differently.

### 1.4 Narrative order for the results section

1. Validity confirmation (one sentence).
2. `rpi_dsatur` vs `full_recompute` — time + vertices touched + significance test. ("Does the core idea work.")
3. `rpi_dsatur` vs `fixed_radius` — same metrics + test. ("Is the learned part contributing.")
4. Scaling behavior across graph sizes.
5. Fallback rate and radius accuracy, explicitly connected back to the Section 6 classifier limitations (the "1"/"full" bucket-merging decision should be cited here as the direct explanation for whatever fallback rate is observed).
6. Threats to validity (Part 3 below) — state these yourself, don't wait for a reviewer to find them.

---

## Part 2: Graph sizes to test — detailed plan

### 2.1 Core sizes: keep 100 / 200 / 500 (already established, do not change)

This matches the methodology's original design (Section 4.1) and, critically, matches what the radius-predictor was **trained** on. Testing on sizes the classifier never saw during training would conflate "does the system design work" with "does the classifier generalize to unseen graph sizes" — two different questions. Stay in-distribution for the main results.

### 2.2 Add one intermediate size for a smoother scaling curve: n=300

**Add this if time allows, as a fourth point, not a replacement for 100/200/500.** A scaling claim ("the advantage grows with graph size") is far more convincing with 4 points on the curve than 3 — it's the difference between showing a trend and showing three dots a reader has to trust form a trend. This does **not** require retraining the classifier — the existing radius predictor can be evaluated on n=300 test graphs directly, since it's a general-purpose predictor, not size-specific (though report this as an in-distribution-adjacent test explicitly, since n=300 wasn't part of training data generation).

If you add n=300: generate ~10-15 new graphs (fewer than the 50/size used for training data, since these are purely for evaluation, not classifier training) using the exact same generator as Section 4.1 (`p` chosen so avg degree lands in [4,10]), and run them through the same edit-stream generator.

### 2.3 Sample size per (size × seed) cell — use ALL held-out test graphs, not a subset

Your `run_matrix.py` already correctly restricts to `test_graph_ids.json` (the true held-out set from training). At current dataset settings (150 graphs total, 70/15/15 split), that's roughly **7-8 test graphs per size**. With 5 seeds and 20 edits/graph, that gives:

```
per size: ~7-8 graphs × 5 seeds × 20 edits ≈ 700-800 paired observations
```

This is comfortably large enough for the Wilcoxon tests in Part 1.2 to have real statistical power. Do not reduce `--seeds` below 5 or the graphs-per-size below what's in `test_graph_ids.json` — that would weaken exactly the significance tests the paper leans on.

### 2.4 Compute budget — what to actually expect, and how to avoid wasted time

**Run a timing pilot before committing to the full matrix** (you may have already done this):
```
python -m experiments.run_matrix --systems full_recompute fixed_radius rpi_dsatur --sizes 100 --seeds 0 --model-checkpoint model/checkpoints/radius_gnn.pt --model-config model/config.yaml --graphs-dir data/raw/graphs --edits-dir data/raw/edits --out results/pilot_outcomes.jsonl
```
Time this run. `full_recompute` is the dominant cost (DSATUR from scratch every edit) and scales worse than the other two systems as graph size grows — its cost is roughly proportional to graph size × edit count, run independently at every single edit. Use the n=100 pilot's wall-clock time to extrapolate:

- **n=200 full_recompute**: expect roughly 2-4x the n=100 per-edit cost (DSATUR is not strictly linear in n, but this is a reasonable planning estimate).
- **n=500 full_recompute**: expect roughly 5-15x the n=100 per-edit cost.
- Total matrix time ≈ (pilot time) × (3 or 4 sizes) × (5 seeds), weighted up for the larger sizes' extra per-edit cost.

If the extrapolated total is impractically long given remaining project time, **reduce seeds to 3 before reducing graph sizes or test-graph count** — seeds primarily buy you variance-reduction on the significance tests, while sizes and graph count are what the actual scaling and sample-size claims depend on. Losing statistical headroom is a smaller sacrifice than losing the scaling curve or shrinking your paired-observation count below a defensible threshold.

### 2.5 What NOT to do

- **Do not test on graph sizes larger than what training data covered (n>500) without explicitly flagging it as an out-of-distribution generalization test**, separate from the main results — the classifier has no reason to have learned anything reliable about repair-radius patterns at a scale it never saw.
- **Do not cherry-pick which held-out test graphs to include** — use the full `test_graph_ids.json` set for each size, every time. Silently dropping "inconvenient" graphs (e.g., ones where `rpi_dsatur` looks bad) would invalidate the whole statistical argument.
- **Do not change seeds between the pilot and the full run** — the pilot is for timing only; results from it should not be mixed into the final reported numbers unless the pilot's seed (0) is also included in the final `--seeds` list (which it is, per the standard command).

---

## Part 3: Threats to validity — include this section in the paper, stated plainly

- **Synthetic graphs only**: Erdős–Rényi graphs lack the spatial/geometric structure of real wireless-interference topologies. State this explicitly; frame it as future work, not a hidden gap.
- **Uniform-random edit locations**: real network churn is plausibly spatially/temporally clustered (the deferred Hawkes-process churn model from early project discussion). The current evaluation covers the simpler, non-adversarial churn case — say so.
- **In-distribution test graphs**: the classifier is evaluated on graphs from the same generative distribution it was trained on (same n range, same p range). This is expected and appropriate for validating the system design, but do not claim out-of-distribution generalization was tested unless Section 2.2's n=300 (or similar) extension was actually run and is reported as such.
