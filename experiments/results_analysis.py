"""Rebuild results/figures and results/tables from the saved matrix JSONLs."""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.graph_level_stats import paired_graph_level_wilcoxon
from experiments.stats_corrections import holm_bonferroni_correct

DATA_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR = ROOT / "results" / "tables"

SYSTEM_ORDER = ["full_recompute", "fixed_radius", "majority", "rpi_dsatur"]
SYSTEM_LABEL = {
    "full_recompute": "Full recompute",
    "fixed_radius": "Fixed r=2",
    "majority": "Majority r=0",
    "rpi_dsatur": "RPI hop-shell",
}
SYSTEM_COLOR = {
    "full_recompute": "#4a4a4a",
    "fixed_radius": "#8a6d1b",
    "majority": "#3d5a80",
    "rpi_dsatur": "#8b1e1e",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_json(path, lines=True)


def assert_pair_join(a: pd.DataFrame, b: pd.DataFrame, keys, label: str) -> None:
    left = a.set_index(keys)
    right = b.set_index(keys)
    joined = left.join(right, how="inner", lsuffix="_a", rsuffix="_b")
    expected = min(len(a), len(b))
    if len(joined) != expected:
        only_a = set(a["graph_id"]) - set(b["graph_id"])
        only_b = set(b["graph_id"]) - set(a["graph_id"])
        warnings.warn(
            f"{label}: joined {len(joined)} != min({len(a)},{len(b)})={expected}; "
            f"only_a={sorted(only_a)[:10]} only_b={sorted(only_b)[:10]}"
        )
    assert len(joined) == expected, (
        f"{label}: join incomplete ({len(joined)} vs expected {expected})"
    )


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print("wrote", path)
    return path


def load_edit_types(graph_ids, edits_dir: Path) -> dict:
    mapping = {}
    missing = []
    for gid in sorted(graph_ids):
        fp = edits_dir / f"{gid}_edits.json"
        if not fp.exists():
            missing.append(gid)
            continue
        payload = json.loads(fp.read_text(encoding="utf-8"))
        for ed in payload.get("edits", []):
            mapping[(gid, int(ed["index"]))] = ed["type"]
    if missing:
        warnings.warn(f"missing edit files: {missing[:10]}")
    return mapping


def paired_edit_wilcoxon(rpi_df: pd.DataFrame, other_df: pd.DataFrame, metric: str) -> dict:
    a = rpi_df.set_index(["graph_id", "edit_index", "seed"])[metric]
    b = other_df.set_index(["graph_id", "edit_index", "seed"])[metric]
    common = a.index.intersection(b.index)
    assert len(common) == min(len(a), len(b)), (
        f"edit-level join incomplete: {len(common)} vs min({len(a)},{len(b)})"
    )
    diffs = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        p, W, r_rb = 1.0, float("nan"), float("nan")
    else:
        res = stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
        p, W = float(res.pvalue), float(res.statistic)
        n = len(nz)
        r_rb = 2.0 * W / (n * (n + 1) / 2.0) - 1.0
    return {
        "n_pairs": int(len(common)),
        "n_nonzero": int(len(nz)),
        "median_delta": float(np.median(diffs)),
        "mean_delta": float(np.mean(diffs)),
        "p": p,
        "r_rb": r_rb,
    }


def generate_all() -> dict:
    _setup_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    static = load_jsonl(DATA_DIR / "matrix_static_systems.jsonl")
    rpi = load_jsonl(DATA_DIR / "matrix_rpi_hopshell.jsonl")
    ood = load_jsonl(DATA_DIR / "matrix_n2000_rpi_vs_full.jsonl")

    static = static[static["seed"] == 0].copy()
    rpi = rpi[rpi["seed"] == 0].copy()
    ood = ood[ood["seed"] == 0].copy()

    holdout = pd.concat([static, rpi], ignore_index=True)
    all_rows = pd.concat([holdout, ood], ignore_index=True)

    keys = ["graph_id", "edit_index", "seed"]
    for sys in ["full_recompute", "fixed_radius", "majority"]:
        assert_pair_join(
            rpi[keys + ["time_ms"]],
            static.loc[static["system"] == sys, keys + ["time_ms"]],
            keys,
            f"rpi vs {sys}",
        )

    n_pairs_holdout = len(rpi)
    n_pairs_ood = int((ood["system"] == "rpi_dsatur").sum())

    # --- fig1 ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for system, marker in [("full_recompute", "o"), ("rpi_dsatur", "s")]:
        sub = all_rows[all_rows["system"] == system]
        g = sub.groupby("size")["time_ms"].median().sort_index()
        ax.plot(
            g.index.astype(float),
            g.values,
            marker=marker,
            color=SYSTEM_COLOR[system],
            label=SYSTEM_LABEL[system],
            linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Graph size n")
    ax.set_ylabel("Median time_ms per edit")
    ax.set_title(
        f"Median time per edit by graph size "
        f"(n_pairs={n_pairs_holdout} for n<=500, n_pairs={n_pairs_ood} for n=2000)"
    )
    ax.legend()
    ax.set_xticks([100, 200, 500, 2000])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    save_fig(fig, "fig1_combined_scaling")

    # --- fig2 ---
    cov_path = DATA_DIR / "hop_coverage.csv"
    if not cov_path.exists():
        print("data unavailable: results/hop_coverage.csv missing")
    else:
        cov = pd.read_csv(cov_path)
        stats_by_n = (
            cov.groupby("n")["coverage_fraction"]
            .agg(
                mean="mean",
                q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75),
                n_samples="count",
            )
            .reset_index()
            .sort_values("n")
        )
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        # Mean can fall outside [q25, q75] under skew; clip distances for matplotlib.
        yerr = np.vstack(
            [
                np.maximum(stats_by_n["mean"] - stats_by_n["q25"], 0.0),
                np.maximum(stats_by_n["q75"] - stats_by_n["mean"], 0.0),
            ]
        )
        ax.fill_between(
            stats_by_n["n"].astype(float),
            stats_by_n["q25"],
            stats_by_n["q75"],
            color="#1b4f72",
            alpha=0.15,
            label="IQR",
        )
        ax.errorbar(
            stats_by_n["n"].astype(float),
            stats_by_n["mean"],
            yerr=yerr,
            fmt="-o",
            color="#1b4f72",
            capsize=4,
            linewidth=2,
            label="mean",
        )
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Graph size n")
        ax.set_ylabel("Coverage fraction (4-hop)")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Mean 4-hop coverage vs n (IQR bars; "
            f"n_vertex_samples={int(stats_by_n['n_samples'].sum())}, "
            f"n_graphs={cov['graph_id'].nunique()})"
        )
        ax.set_xticks(stats_by_n["n"].tolist())
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        save_fig(fig, "fig2_hop_coverage_vs_n")

    # --- fig3 ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for system in SYSTEM_ORDER:
        sub = all_rows[all_rows["system"] == system]
        if sub.empty:
            continue
        g = sub.groupby("size")["fallback_triggered"].mean().sort_index() * 100.0
        if system in ("fixed_radius", "majority"):
            g = g[g.index <= 500]
        ax.plot(
            g.index.astype(float),
            g.values,
            marker="o",
            color=SYSTEM_COLOR[system],
            label=SYSTEM_LABEL[system],
            linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Graph size n")
    ax.set_ylabel("Fallback rate (%)")
    ax.set_title(
        f"Fallback rate vs n "
        f"(n_edits={n_pairs_holdout}/system for n<=500; "
        f"n_edits={n_pairs_ood}/system for n=2000, RPI+full only)"
    )
    ax.legend()
    ax.set_xticks([100, 200, 500, 2000])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    save_fig(fig, "fig3_fallback_vs_n")

    # --- fig4 ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for system in SYSTEM_ORDER:
        sub = all_rows[all_rows["system"] == system]
        if sub.empty:
            continue
        g = sub.groupby("size")["vertices_touched"].mean().sort_index()
        ax.plot(
            g.index.astype(float),
            g.values,
            marker="o",
            color=SYSTEM_COLOR[system],
            label=SYSTEM_LABEL[system],
            linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Graph size n")
    ax.set_ylabel("Mean vertices_touched")
    ax.set_title(
        f"Mean vertices touched vs n "
        f"(n_edits={n_pairs_holdout}/system holdout; n_edits={n_pairs_ood} OOD)"
    )
    ax.legend()
    ax.set_xticks([100, 200, 500, 2000])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    save_fig(fig, "fig4_vertices_vs_n")

    # --- fig5 ---
    edit_map = load_edit_types(set(holdout["graph_id"]), ROOT / "data" / "raw" / "edits")
    holdout_et = holdout.copy()
    holdout_et["edit_type"] = [
        edit_map.get((gid, int(ei)), "UNKNOWN")
        for gid, ei in zip(holdout_et["graph_id"], holdout_et["edit_index"])
    ]
    assert (holdout_et["edit_type"] != "UNKNOWN").all(), "edit_type join incomplete"
    edit_types = ["shrink_list", "remove_vertex"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fb = (
        holdout_et.groupby(["edit_type", "system"])["fallback_triggered"]
        .mean()
        .mul(100.0)
        .unstack("system")
        .reindex(edit_types)[SYSTEM_ORDER]
    )
    fb.plot(kind="bar", ax=axes[0], color=[SYSTEM_COLOR[s] for s in SYSTEM_ORDER], rot=0)
    axes[0].set_ylabel("Fallback rate (%)")
    axes[0].set_title("Fallback rate by edit type")
    axes[0].legend([SYSTEM_LABEL[s] for s in SYSTEM_ORDER], fontsize=8)
    vt = (
        holdout_et.groupby(["edit_type", "system"])["vertices_touched"]
        .mean()
        .unstack("system")
        .reindex(edit_types)[SYSTEM_ORDER]
    )
    vt.plot(
        kind="bar",
        ax=axes[1],
        color=[SYSTEM_COLOR[s] for s in SYSTEM_ORDER],
        rot=0,
        legend=False,
    )
    axes[1].set_ylabel("Mean vertices_touched")
    axes[1].set_title("Mean vertices touched by edit type")
    fig.suptitle(
        f"Holdout edit-type breakdown (n_edits={len(holdout_et)}, "
        f"n_graphs={holdout_et['graph_id'].nunique()})",
        y=1.02,
    )
    save_fig(fig, "fig5_edit_type_breakdown")

    # --- fig6 ---
    report = json.loads(
        (ROOT / "model" / "reports" / "hop_shell" / "train_report.json").read_text(
            encoding="utf-8"
        )
    )
    labels = report["confusion_matrix_labels"]
    cm = np.array(report["confusion_matrix"], dtype=int)
    n_test = int(cm.sum())
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Hop-shell confusion matrix (rows=true; n_test_edits={n_test})")
    save_fig(fig, "fig6_confusion_heatmap")

    # --- fig7 ---
    recalls = [report["test_per_class_recall"][lab] for lab in labels]
    supports = [report["class_counts"][lab] for lab in labels]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(labels, recalls, color="#1b4f72", edgecolor="none")
    for bar, n in zip(bars, supports):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall")
    ax.set_xlabel("Class (radius bucket)")
    ax.set_title(
        f"Per-class recall on hop-shell test split "
        f"(n_test_edits={sum(supports)}, accuracy={report['test_accuracy']:.3f})"
    )
    save_fig(fig, "fig7_per_class_recall")

    # --- fig8 ---
    rpi_cost = rpi.copy()
    rpi_cost["retry_group"] = np.where(
        rpi_cost["attempts"] == 0,
        "first-try (attempts=0)",
        "retried (attempts>0)",
    )
    n0 = int((rpi_cost["attempts"] == 0).sum())
    n1 = int((rpi_cost["attempts"] > 0).sum())
    order = ["first-try (attempts=0)", "retried (attempts>0)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.boxplot(
        data=rpi_cost,
        x="retry_group",
        y="time_ms",
        order=order,
        ax=axes[0],
        color="#8b1e1e",
        showfliers=False,
    )
    axes[0].set_ylabel("time_ms")
    axes[0].set_xlabel("")
    axes[0].set_title("time_ms")
    sns.boxplot(
        data=rpi_cost,
        x="retry_group",
        y="vertices_touched",
        order=order,
        ax=axes[1],
        color="#8b1e1e",
        showfliers=False,
    )
    axes[1].set_ylabel("vertices_touched")
    axes[1].set_xlabel("")
    axes[1].set_title("vertices_touched")
    fig.suptitle(
        f"RPI misprediction / retry cost "
        f"(n_first_try={n0}, n_retried={n1}; holdout n_edits={len(rpi_cost)})",
        y=1.02,
    )
    save_fig(fig, "fig8_misprediction_cost")

    # --- fig9 ---
    sweep_path = DATA_DIR / "fixed_radius_sweep.jsonl"
    if not sweep_path.exists():
        print("data unavailable: results/fixed_radius_sweep.jsonl missing")
    else:
        sweep = load_jsonl(sweep_path)
        r2 = static[static["system"] == "fixed_radius"].copy()
        r2["radius"] = 2
        sweep_all = pd.concat([sweep, r2], ignore_index=True)
        by_r = (
            sweep_all.groupby("radius")
            .agg(
                median_ms=("time_ms", "median"),
                fallback_pct=("fallback_triggered", lambda s: 100.0 * s.mean()),
                n=("time_ms", "count"),
            )
            .sort_index()
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(by_r.index, by_r["median_ms"], "-o", color="#8a6d1b", linewidth=2)
        axes[0].set_xlabel("Fixed radius")
        axes[0].set_ylabel("Median time_ms")
        axes[0].set_xticks([1, 2, 3, 4])
        axes[0].set_title("Median time")
        axes[1].plot(by_r.index, by_r["fallback_pct"], "-o", color="#8a6d1b", linewidth=2)
        axes[1].set_xlabel("Fixed radius")
        axes[1].set_ylabel("Fallback rate (%)")
        axes[1].set_xticks([1, 2, 3, 4])
        axes[1].set_title("Fallback rate")
        n_per = int(by_r["n"].iloc[0])
        fig.suptitle(
            f"Fixed-radius sweep on holdout "
            f"(n_edits={n_per}/radius; n_graphs={r2['graph_id'].nunique()})",
            y=1.02,
        )
        save_fig(fig, "fig9_fixed_radius_sweep")

    # --- table_wilcoxon_full ---
    comparisons = [
        ("RPI − full", "full_recompute"),
        ("RPI − fixed r=2", "fixed_radius"),
        ("RPI − majority", "majority"),
    ]
    metrics = [("time_ms", "time_ms"), ("vertices", "vertices_touched")]
    edit_rows = []
    raw_ps = []
    for cname, sys in comparisons:
        other = static[static["system"] == sys]
        for mlabel, mcol in metrics:
            w = paired_edit_wilcoxon(rpi, other, mcol)
            edit_rows.append(
                {
                    "comparison": cname,
                    "metric": mlabel,
                    **w,
                    "baseline_system": sys,
                    "metric_col": mcol,
                }
            )
            raw_ps.append(w["p"])
    holm = holm_bonferroni_correct(raw_ps, alpha=0.05)
    rows = []
    for er, h in zip(edit_rows, holm):
        g = paired_graph_level_wilcoxon(
            holdout, er["metric_col"], "rpi_dsatur", er["baseline_system"]
        )
        rows.append(
            {
                "comparison": er["comparison"],
                "metric": er["metric"],
                "n_pairs_edit": er["n_pairs"],
                "n_nonzero_edit": er["n_nonzero"],
                "median_delta_edit": er["median_delta"],
                "mean_delta_edit": er["mean_delta"],
                "p_edit": er["p"],
                "r_rb_edit": er["r_rb"],
                "p_holm": h["p_holm"],
                "reject_holm_0.05": h["reject_at_0.05"],
                "holm_rank": h["rank"],
                "n_graphs": g["n_graphs"],
                "median_delta_graph": g["median_delta"],
                "mean_delta_graph": g["mean_delta"],
                "p_graph": g["p"],
                "statistic_graph": g["statistic"],
            }
        )
    wilcoxon_full = pd.DataFrame(rows)
    save_table(wilcoxon_full, "table_wilcoxon_full")

    # --- table_headline_summary ---
    headline_rows = []
    for sys in SYSTEM_ORDER:
        sub = holdout[holdout["system"] == sys]
        att = sub["attempts"].to_numpy(dtype=float)
        headline_rows.append(
            {
                "system": SYSTEM_LABEL[sys],
                "n_edits": len(sub),
                "valid_pct": 100.0 * sub["valid"].mean(),
                "fallback_pct": 100.0 * sub["fallback_triggered"].mean(),
                "first_try_pct": 100.0 * (att == 0).mean(),
                "median_ms": float(sub["time_ms"].median()),
                "mean_ms": float(sub["time_ms"].mean()),
                "mean_vertices": float(sub["vertices_touched"].mean()),
                "mean_retries": float(att.mean()),
            }
        )
    save_table(pd.DataFrame(headline_rows), "table_headline_summary")

    return {
        "figures_dir": str(FIGURES_DIR),
        "tables_dir": str(TABLES_DIR),
        "n_pairs_holdout": n_pairs_holdout,
        "n_pairs_ood": n_pairs_ood,
    }


if __name__ == "__main__":
    info = generate_all()
    print(info)
