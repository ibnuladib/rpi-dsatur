"""Graph-level (not edit-level) paired significance tests.

Addresses pseudoreplication: many edit-level pairs come from few graphs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def aggregate_per_graph(df: pd.DataFrame, metric: str, system: str) -> pd.Series:
    """
    df has columns: system, graph_id, edit_index, seed, <metric>.
    Return one value per graph_id: the MEAN of <metric> across that graph's
    edits, for the given system. Index = graph_id.
    """
    sub = df.loc[df["system"] == system]
    if sub.empty:
        raise ValueError(f"No rows for system={system!r}")
    return sub.groupby("graph_id", sort=True)[metric].mean()


def paired_graph_level_wilcoxon(
    df: pd.DataFrame, metric: str, system_a: str, system_b: str
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
    a = aggregate_per_graph(df, metric, system_a)
    b = aggregate_per_graph(df, metric, system_b)

    ids_a = set(a.index)
    ids_b = set(b.index)
    if ids_a != ids_b:
        only_a = sorted(ids_a - ids_b)
        only_b = sorted(ids_b - ids_a)
        raise AssertionError(
            f"graph_id mismatch for {system_a} vs {system_b}: "
            f"only_a={only_a[:5]}{'...' if len(only_a) > 5 else ''} "
            f"only_b={only_b[:5]}{'...' if len(only_b) > 5 else ''}"
        )

    aligned = pd.DataFrame({"a": a, "b": b}).dropna()
    assert len(aligned) == len(a) == len(b), (
        f"Inner join dropped graphs: len(a)={len(a)} len(b)={len(b)} "
        f"aligned={len(aligned)}"
    )

    diffs = (aligned["a"] - aligned["b"]).to_numpy(dtype=float)
    n_graphs = int(len(diffs))
    if n_graphs == 0:
        raise ValueError("No paired graphs to test")

    nz = diffs[diffs != 0]
    if len(nz) == 0:
        p = 1.0
        statistic = float("nan")
    else:
        res = stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
        p = float(res.pvalue)
        statistic = float(res.statistic)

    return {
        "n_graphs": n_graphs,
        "median_delta": float(np.median(diffs)),
        "mean_delta": float(np.mean(diffs)),
        "p": p,
        "statistic": statistic,
    }
