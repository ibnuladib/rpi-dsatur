"""Generate publication figures/tables for the RPI-DSATUR holdout + OOD study.

Reads:
  results/matrix_static_systems.jsonl   (full, fixed_radius, majority)
  results/matrix_rpi_hopshell.jsonl     (rpi_dsatur hop-shell)
  results/matrix_n2000_rpi_vs_full.jsonl
  model/reports/hop_shell/train_report.json
  data/raw/edits and data/raw/edits_n2000 (edit-type joins)

Does not read results/raw_outcomes.jsonl (stale / different RPI).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from scipy import stats

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

# Historical 6-class test accuracies cited from Context.md / remaining reports
# (artifacts for several of these runs were deleted in §20). Not recomputed.
LINEAGE = [
    ("§13 CE+sqrt inv-freq", 0.561),
    ("§14 slack, no class wts", 0.6931),
    ("Guarded + bias (mean pool)", 0.7000),
    ("Hop-shell (canonical)", 0.7005),
    ("Asymmetric ordinal", 0.6811),
    ("Hop-5 retrain", 0.6956),
    ("Hop-5 + thresholds", 0.7065),
]


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "results" / "matrix_static_systems.jsonl").exists():
            return p
    return here.parents[1]


ROOT = repo_root()
OUT = ROOT / "paper_figures"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
        }
    )


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_edit_types(graph_ids: set[str], edits_dir: Path) -> dict[tuple[str, int], str]:
    mapping: dict[tuple[str, int], str] = {}
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
        print(f"WARNING: missing edit files in {edits_dir}: {missing}")
    return mapping


def pct(x: np.ndarray) -> float:
    return 100.0 * float(np.mean(x)) if len(x) else float("nan")


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty row set")
    t = np.array([r["time_ms"] for r in rows], dtype=float)
    v = np.array([r["vertices_touched"] for r in rows], dtype=float)
    sizes = np.array([r["size"] for r in rows], dtype=float)
    valid = np.array([bool(r["valid"]) for r in rows])
    fb = np.array([bool(r["fallback_triggered"]) for r in rows])
    att = np.array([int(r["attempts"]) for r in rows], dtype=float)
    first = att == 0
    return {
        "n": len(rows),
        "valid_pct": pct(valid),
        "invalid_n": int((~valid).sum()),
        "fallback_pct": pct(fb),
        "first_try_pct": pct(first),
        "median_ms": float(np.median(t)),
        "mean_ms": float(np.mean(t)),
        "p25_ms": float(np.percentile(t, 25)),
        "p75_ms": float(np.percentile(t, 75)),
        "mean_verts": float(np.mean(v)),
        "median_verts": float(np.median(v)),
        "mean_frac_pct": float(np.mean(100.0 * v / sizes)),
        "mean_attempts": float(np.mean(att)),
        "times": t,
        "verts": v,
        "valid": valid,
        "fallback": fb,
        "attempts": att,
    }


def paired_keys(rows: list[dict]) -> dict[tuple, dict]:
    return {(r["graph_id"], int(r["edit_index"]), int(r["seed"])): r for r in rows}


def wilcoxon_rpi_minus(rpi_rows: list[dict], other_rows: list[dict], field: str) -> dict:
    a = paired_keys(rpi_rows)
    b = paired_keys(other_rows)
    keys = sorted(set(a) & set(b))
    diffs = np.array([a[k][field] - b[k][field] for k in keys], dtype=float)
    nz = diffs[diffs != 0]
    n_pos = int((diffs > 0).sum())
    n_neg = int((diffs < 0).sum())
    n_tie = int((diffs == 0).sum())
    if len(nz) == 0:
        p = 1.0
        w = float("nan")
        r_rb = float("nan")
    else:
        res = stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
        p = float(res.pvalue)
        w = float(res.statistic)
        n = len(nz)
        r_rb = 2.0 * w / (n * (n + 1) / 2.0) - 1.0
    return {
        "n_pairs": len(keys),
        "n_nonzero": int(len(nz)),
        "n_ties": n_tie,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "median_diff": float(np.median(diffs)),
        "mean_diff": float(np.mean(diffs)),
        "p": p,
        "W": w,
        "r_rb": r_rb,
        "diffs": diffs,
    }


def savefig(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def table_fig(headers: list[str], rows: list[list], stem: str, title: str) -> None:
    n_c = len(headers)
    n_r = len(rows)
    fig_w = max(8.0, 0.95 * n_c + 1.5)
    fig_h = max(1.6, 0.38 * (n_r + 2) + 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, pad=8)
    tbl = ax.table(
        cellText=[[str(c) for c in r] for r in rows],
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if r == 0:
            cell.set_facecolor("#ececec")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7f7f7")
        else:
            cell.set_facecolor("white")
        if c == 0:
            cell.set_text_props(ha="left")
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def grouped_bars(ax, categories, series: dict[str, list[float]], ylabel: str, title: str) -> None:
    x = np.arange(len(categories))
    keys = list(series)
    width = 0.8 / max(len(keys), 1)
    for i, k in enumerate(keys):
        color = SYSTEM_COLOR.get(k, "#333333")
        offs = x + (i - (len(keys) - 1) / 2) * width
        ax.bar(offs, series[k], width, label=SYSTEM_LABEL.get(k, k), color=color, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=min(4, len(keys)))


def fmt_p(p: float) -> str:
    if p != p:
        return "NA"
    if p < 1e-300:
        return "<1e-300"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def load_all():
    static = load_jsonl(ROOT / "results" / "matrix_static_systems.jsonl")
    rpi = load_jsonl(ROOT / "results" / "matrix_rpi_hopshell.jsonl")
    ood = load_jsonl(ROOT / "results" / "matrix_n2000_rpi_vs_full.jsonl")
    report = json.loads((ROOT / "model" / "reports" / "hop_shell" / "train_report.json").read_text(encoding="utf-8"))
    test_ids = json.loads((ROOT / "model" / "reports" / "hop_shell" / "test_graph_ids.json").read_text(encoding="utf-8"))
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in static:
        by_sys[r["system"]].append(r)
    for r in rpi:
        by_sys[r["system"]].append(r)
    ood_sys: dict[str, list[dict]] = defaultdict(list)
    for r in ood:
        ood_sys[r["system"]].append(r)
    holdout_ids = {r["graph_id"] for r in static} | {r["graph_id"] for r in rpi}
    edit_map = load_edit_types(holdout_ids, ROOT / "data" / "raw" / "edits")
    ood_ids = {r["graph_id"] for r in ood}
    ood_edit_map = load_edit_types(ood_ids, ROOT / "data" / "raw" / "edits_n2000")
    for rows in list(by_sys.values()) + list(ood_sys.values()):
        for r in rows:
            src = ood_edit_map if r["size"] == 2000 else edit_map
            r["edit_type"] = src.get((r["graph_id"], int(r["edit_index"])))
    return {
        "by_sys": dict(by_sys),
        "ood_sys": dict(ood_sys),
        "report": report,
        "test_ids": test_ids,
        "edit_map": edit_map,
        "ood_edit_map": ood_edit_map,
    }


def attach_edit_filter(rows: list[dict], etype: str | None) -> list[dict]:
    if etype is None:
        return rows
    return [r for r in rows if r.get("edit_type") == etype]


def system_table_rows(by_sys: dict[str, list[dict]], size: int | None = None) -> list[list]:
    rows_out = []
    for sys in SYSTEM_ORDER:
        if sys not in by_sys:
            continue
        subset = by_sys[sys] if size is None else [r for r in by_sys[sys] if r["size"] == size]
        if not subset:
            continue
        s = summarize(subset)
        rows_out.append(
            [
                SYSTEM_LABEL[sys],
                s["n"],
                f"{s['valid_pct']:.2f}",
                f"{s['fallback_pct']:.2f}",
                f"{s['first_try_pct']:.2f}",
                f"{s['median_ms']:.2f}",
                f"{s['mean_ms']:.2f}",
                f"{s['mean_verts']:.2f}",
                f"{s['mean_frac_pct']:.2f}",
                f"{s['mean_attempts']:.2f}",
            ]
        )
    return rows_out


def generate_all() -> dict:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    by_sys = data["by_sys"]
    ood_sys = data["ood_sys"]
    report = data["report"]

    headers = [
        "System",
        "Edits",
        "Valid %",
        "Fallback %",
        "First-try %",
        "Median ms",
        "Mean ms",
        "Mean verts",
        "Touched frac %",
        "Mean retries",
    ]

    # --- Tables ---
    tab01 = system_table_rows(by_sys)
    write_csv(OUT / "tab01_system_summary.csv", headers, tab01)
    table_fig(headers, tab01, "tab01_system_summary", "Holdout system summary (seed 0; n=100/200/500 pooled)")

    tab02_headers = ["n"] + headers
    tab02 = []
    for n in (100, 200, 500):
        for row in system_table_rows(by_sys, size=n):
            tab02.append([n] + row)
    write_csv(OUT / "tab02_system_summary_by_n.csv", tab02_headers, tab02)
    table_fig(tab02_headers, tab02, "tab02_system_summary_by_n", "Holdout system summary by graph size")

    ood_headers = [
        "System",
        "Edits",
        "Valid %",
        "Fallback %",
        "First-try %",
        "Median ms",
        "Mean ms",
        "IQR ms",
        "Mean verts",
    ]
    ood_rows = []
    for sys in ("full_recompute", "rpi_dsatur"):
        s = summarize(ood_sys[sys])
        ood_rows.append(
            [
                SYSTEM_LABEL[sys],
                s["n"],
                f"{s['valid_pct']:.2f}",
                f"{s['fallback_pct']:.2f}",
                f"{s['first_try_pct']:.2f}",
                f"{s['median_ms']:.2f}",
                f"{s['mean_ms']:.2f}",
                f"{s['p25_ms']:.2f}–{s['p75_ms']:.2f}",
                f"{s['mean_verts']:.2f}",
            ]
        )
    write_csv(OUT / "tab03_ood_n2000.csv", ood_headers, ood_rows)
    table_fig(ood_headers, ood_rows, "tab03_ood_n2000", "OOD n=2000: RPI hop-shell vs full only (8 graphs × 20 edits)")

    labels = report["confusion_matrix_labels"]
    cm = np.array(report["confusion_matrix"], dtype=int)
    cm_headers = ["true\\pred"] + labels + ["n", "P", "R", "F1"]
    cm_rows = []
    for i, lab in enumerate(labels):
        n_i = int(cm[i].sum())
        cm_rows.append(
            [
                lab,
                *[int(x) for x in cm[i]],
                n_i,
                f"{report['test_per_class_precision'][lab]:.3f}",
                f"{report['test_per_class_recall'][lab]:.3f}",
                f"{report['test_per_class_f1'][lab]:.3f}",
            ]
        )
    write_csv(OUT / "tab04_confusion_matrix.csv", cm_headers, cm_rows)
    table_fig(cm_headers, cm_rows, "tab04_confusion_matrix", "Hop-shell test confusion (rows=true; sequential test n=2,007)")

    wx_headers = ["Comparison", "Metric", "n pairs", "n ≠0", "Median Δ", "Mean Δ", "p", "r_rb"]
    wx_specs = [
        ("RPI − full", "time_ms", by_sys["rpi_dsatur"], by_sys["full_recompute"]),
        ("RPI − full", "vertices_touched", by_sys["rpi_dsatur"], by_sys["full_recompute"]),
        ("RPI − fixed r=2", "time_ms", by_sys["rpi_dsatur"], by_sys["fixed_radius"]),
        ("RPI − fixed r=2", "vertices_touched", by_sys["rpi_dsatur"], by_sys["fixed_radius"]),
        ("RPI − majority", "time_ms", by_sys["rpi_dsatur"], by_sys["majority"]),
        ("RPI − majority", "vertices_touched", by_sys["rpi_dsatur"], by_sys["majority"]),
        ("OOD RPI − full", "time_ms", ood_sys["rpi_dsatur"], ood_sys["full_recompute"]),
        ("OOD RPI − full", "vertices_touched", ood_sys["rpi_dsatur"], ood_sys["full_recompute"]),
    ]
    wx_rows = []
    wx_results = {}
    for name, field, a, b in wx_specs:
        w = wilcoxon_rpi_minus(a, b, field)
        wx_results[(name, field)] = w
        unit = " ms" if field == "time_ms" else ""
        wx_rows.append(
            [
                name,
                "time_ms" if field == "time_ms" else "vertices",
                w["n_pairs"],
                w["n_nonzero"],
                f"{w['median_diff']:+.2f}{unit}",
                f"{w['mean_diff']:+.2f}{unit}",
                fmt_p(w["p"]),
                f"{w['r_rb']:.2f}" if w["r_rb"] == w["r_rb"] else "NA",
            ]
        )
    write_csv(OUT / "tab05_wilcoxon.csv", wx_headers, wx_rows)
    table_fig(wx_headers, wx_rows, "tab05_wilcoxon", "Paired Wilcoxon (RPI minus baseline; seed 0; nonzero diffs)")

    et_headers = [
        "Edit type",
        "System",
        "n",
        "Fallback %",
        "First-try %",
        "Median ms",
        "Mean verts",
        "Valid %",
    ]
    et_rows = []
    for etype in ("shrink_list", "remove_vertex"):
        for sys in SYSTEM_ORDER:
            subset = attach_edit_filter(by_sys[sys], etype)
            s = summarize(subset)
            et_rows.append(
                [
                    etype,
                    SYSTEM_LABEL[sys],
                    s["n"],
                    f"{s['fallback_pct']:.2f}",
                    f"{s['first_try_pct']:.2f}",
                    f"{s['median_ms']:.2f}",
                    f"{s['mean_verts']:.2f}",
                    f"{s['valid_pct']:.2f}",
                ]
            )
    write_csv(OUT / "tab06_edit_type.csv", et_headers, et_rows)
    table_fig(et_headers, et_rows, "tab06_edit_type", "Holdout metrics by edit type (joined from data/raw/edits)")

    s9_headers = ["Metric", "If the method works", "Observed (seed 0, holdout)"]
    s_full = summarize(by_sys["full_recompute"])
    s_fix = summarize(by_sys["fixed_radius"])
    s_maj = summarize(by_sys["majority"])
    s_rpi = summarize(by_sys["rpi_dsatur"])
    s9 = [
        [
            "Validity",
            "0 conflicts for all systems",
            f"~{s_rpi['valid_pct']:.1f}% success; failures match full "
            f"({s_full['invalid_n']} vs {s_rpi['invalid_n']} infeasible)",
        ],
        [
            "Wall-clock",
            "RPI < full; RPI ≤ fixed",
            f"Fail: RPI median {s_rpi['median_ms']:.1f} ms vs full {s_full['median_ms']:.1f} "
            f"vs fixed {s_fix['median_ms']:.1f}",
        ],
        [
            "Vertices touched",
            "RPI < full",
            f"Pass vs full ({s_rpi['mean_verts']:.2f} < {s_full['mean_verts']:.2f}) and vs fixed "
            f"({s_rpi['mean_verts']:.2f} ≪ {s_fix['mean_verts']:.2f}); ties majority "
            f"({s_maj['mean_verts']:.2f})",
        ],
        [
            "Radius accuracy",
            "≫ majority-class baseline",
            f"Pass: {100*report['test_accuracy']:.2f}% vs "
            f"{100*report['majority_class_baseline_accuracy']:.2f}%",
        ],
        [
            "Fallback",
            "<15%",
            f"Fail: RPI {s_rpi['fallback_pct']:.1f}%, fixed {s_fix['fallback_pct']:.1f}%, "
            f"majority {s_maj['fallback_pct']:.1f}%",
        ],
    ]
    write_csv(OUT / "tab07_section9_checklist.csv", s9_headers, s9)
    table_fig(s9_headers, s9, "tab07_section9_checklist", "Section 9 expected-pattern check")

    clf_headers = ["Metric", "Value"]
    clf_rows = [
        ["Test n", sum(report["class_counts"].values())],
        ["6-class accuracy", f"{report['test_accuracy']:.4f}"],
        ["Majority-class baseline", f"{report['majority_class_baseline_accuracy']:.4f}"],
        ["Δ vs majority (pp)", f"{100*(report['test_accuracy']-report['majority_class_baseline_accuracy']):.2f}"],
        ["Macro recall", f"{report['test_macro_recall']:.4f}"],
        ["Macro F1", f"{report['test_macro_f1']:.4f}"],
        ["Under-prediction rate", f"{report['under_prediction_rate']:.4f}"],
        ["Over-prediction rate", f"{report['over_prediction_rate']:.4f}"],
        ["Merged-{0,1} (secondary)", f"{report['test_accuracy_merged_01_secondary']:.4f}"],
        ["Selection", report["selection_metric"]],
        ["Loss", f"{report['loss']} λ={report['ordinal_lambda']}"],
    ]
    write_csv(OUT / "tab08_classifier_metrics.csv", clf_headers, clf_rows)
    table_fig(clf_headers, clf_rows, "tab08_classifier_metrics", "Deployed hop-shell classifier (test sequential edits)")

    rpi_ft = [r for r in by_sys["rpi_dsatur"] if int(r["attempts"]) == 0]
    rpi_rt = [r for r in by_sys["rpi_dsatur"] if int(r["attempts"]) > 0]
    ft = summarize(rpi_ft)
    rt = summarize(rpi_rt)
    cost_headers = ["Subset", "Edits", "Median ms", "Mean ms", "Mean verts", "Fallback %", "Valid %"]
    cost_rows = [
        ["First-try (attempts=0)", ft["n"], f"{ft['median_ms']:.2f}", f"{ft['mean_ms']:.2f}", f"{ft['mean_verts']:.2f}", f"{ft['fallback_pct']:.2f}", f"{ft['valid_pct']:.2f}"],
        ["Retried (attempts>0)", rt["n"], f"{rt['median_ms']:.2f}", f"{rt['mean_ms']:.2f}", f"{rt['mean_verts']:.2f}", f"{rt['fallback_pct']:.2f}", f"{rt['valid_pct']:.2f}"],
    ]
    write_csv(OUT / "tab09_firsttry_vs_retry.csv", cost_headers, cost_rows)
    table_fig(cost_headers, cost_rows, "tab09_firsttry_vs_retry", "RPI cost of a missed first try (holdout)")

    # --- KPI bar: accuracy vs majority ---
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    accs = [100 * report["majority_class_baseline_accuracy"], 100 * report["test_accuracy"]]
    ax.bar(["Always-0\n(majority)", "Hop-shell\nGATv2"], accs, color=["#6b6b6b", "#3d5a80"], width=0.55)
    ax.axhline(72, color="#8b1e1e", ls="--", lw=1, label="72% spec target")
    ax.set_ylabel("6-class test accuracy (%)")
    ax.set_ylim(0, 85)
    ax.set_title("Radius classifier vs majority baseline")
    for i, v in enumerate(accs):
        ax.text(i, v + 1.2, f"{v:.2f}%", ha="center", fontsize=9)
    ax.legend(frameon=False)
    savefig(fig, "fig01_accuracy_vs_majority")

    # --- System comparison bars ---
    cats = [SYSTEM_LABEL[s] for s in SYSTEM_ORDER]
    overall = {s: summarize(by_sys[s]) for s in SYSTEM_ORDER}

    def sys_bar(values, ylabel, stem, title, hline=None, hlabel=None):
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        colors = [SYSTEM_COLOR[s] for s in SYSTEM_ORDER]
        ax.bar(cats, values, color=colors, width=0.62)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if hline is not None:
            ax.axhline(hline, color="#2f6f4f", ls="--", lw=1, label=hlabel)
            ax.legend(frameon=False)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        savefig(fig, stem)

    sys_bar([overall[s]["valid_pct"] for s in SYSTEM_ORDER], "Valid colorings (%)",
            "fig02_validity_by_system",
            "Validity is shared with full recompute (holdout, 2,025 edits)")
    sys_bar([overall[s]["median_ms"] for s in SYSTEM_ORDER], "Median time (ms)",
            "fig03_median_time_by_system",
            "RPI is slower than full DSATUR on holdout wall-clock")
    sys_bar([overall[s]["fallback_pct"] for s in SYSTEM_ORDER], "Fallback rate (%)",
            "fig04_fallback_by_system",
            "RPI fallback tracks majority r=0 (15% target unmet)",
            hline=15, hlabel="15% target")
    sys_bar([overall[s]["mean_verts"] for s in SYSTEM_ORDER], "Mean vertices touched",
            "fig05_mean_verts_by_system",
            "RPI touches fewer vertices than fixed r=2 (−37% mean)")
    sys_bar([overall[s]["first_try_pct"] for s in SYSTEM_ORDER], "First-try rate (%)",
            "fig06_first_try_by_system",
            "First-try success is almost entirely RPI/majority on shrink_list")

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    x = np.arange(4)
    w = 0.25
    ax.bar(x - w, [overall[s]["p25_ms"] for s in SYSTEM_ORDER], w, label="p25", color="#b0b0b0")
    ax.bar(x, [overall[s]["median_ms"] for s in SYSTEM_ORDER], w, label="Median", color="#3d5a80")
    ax.bar(x + w, [overall[s]["p75_ms"] for s in SYSTEM_ORDER], w, label="p75", color="#8b1e1e")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Median wall-clock with IQR (holdout; right-skewed)")
    ax.legend(frameon=False)
    savefig(fig, "fig07_time_iqr_by_system")

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    w = 0.35
    ax.bar(x - w / 2, [overall[s]["mean_verts"] for s in SYSTEM_ORDER], w, label="Mean vertices", color="#2f6f4f")
    ax.bar(x + w / 2, [overall[s]["mean_ms"] for s in SYSTEM_ORDER], w, label="Mean time (ms)", color="#8b1e1e")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Mean vertices / mean ms")
    ax.set_title("Locality vs speed: fewer vertices do not yield a faster system")
    ax.legend(frameon=False)
    savefig(fig, "fig08_locality_vs_speed")

    # Confusion heatmap
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Greys")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted radius bucket")
    ax.set_ylabel("True radius bucket")
    ax.set_title("Hop-shell 6-class confusion (test n=2,007)")
    vmax = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > 0.55 * vmax else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    savefig(fig, "fig09_confusion_matrix")

    # Per-class P/R/F1
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    xx = np.arange(len(labels))
    w = 0.25
    prec = [report["test_per_class_precision"][k] * 100 for k in labels]
    rec = [report["test_per_class_recall"][k] * 100 for k in labels]
    f1 = [report["test_per_class_f1"][k] * 100 for k in labels]
    ax.bar(xx - w, prec, w, label="Precision", color="#4a4a4a")
    ax.bar(xx, rec, w, label="Recall", color="#3d5a80")
    ax.bar(xx + w, f1, w, label="F1", color="#8b1e1e")
    ax.set_xticks(xx)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percent")
    ax.set_title("Per-class precision / recall / F1 (class 1 and full are never predicted)")
    ax.legend(frameon=False)
    savefig(fig, "fig10_perclass_prf1")

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(["Under-predict", "Over-predict"],
           [100 * report["under_prediction_rate"], 100 * report["over_prediction_rate"]],
           color=["#8b1e1e", "#3d5a80"], width=0.5)
    ax.set_ylabel("Rate on test edits (%)")
    ax.set_title("Ordinal errors: under-prediction is more common")
    savefig(fig, "fig11_under_vs_over")

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    counts = [report["class_counts"][k] for k in labels]
    ax.bar(labels, counts, color="#3d5a80")
    ax.set_ylabel("Test edits")
    ax.set_xlabel("True radius bucket")
    ax.set_title("Test class distribution is majority-0 (n=2,007 sequential edits)")
    for i, c in enumerate(counts):
        ax.text(i, c + 8, str(c), ha="center", fontsize=8)
    savefig(fig, "fig12_class_distribution")

    # Time boxplots
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    series = [overall[s]["times"] for s in SYSTEM_ORDER]
    bp = ax.boxplot(series, showfliers=False, patch_artist=True, medianprops={"color": "black", "lw": 1.4})
    ax.set_xticklabels(cats)
    for patch, sys in zip(bp["boxes"], SYSTEM_ORDER):
        patch.set_facecolor(SYSTEM_COLOR[sys])
        patch.set_alpha(0.55)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-edit time (holdout; outliers hidden — distributions are right-skewed)")
    savefig(fig, "fig13_time_box_by_system")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    positions = []
    data_box = []
    colors = []
    tick_pos = []
    tick_lab = []
    pos = 1
    for n in (100, 200, 500):
        start = pos
        for sys in SYSTEM_ORDER:
            subset = [r["time_ms"] for r in by_sys[sys] if r["size"] == n]
            data_box.append(subset)
            positions.append(pos)
            colors.append(SYSTEM_COLOR[sys])
            pos += 1
        tick_pos.append((start + pos - 1) / 2)
        tick_lab.append(f"n={n}")
        pos += 1
    bp = ax.boxplot(data_box, positions=positions, showfliers=False, patch_artist=True,
                    widths=0.7, medianprops={"color": "black", "lw": 1.2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Time by system and n (medians; outliers hidden)")
    ax.legend(handles=[Patch(facecolor=SYSTEM_COLOR[s], alpha=0.55, label=SYSTEM_LABEL[s]) for s in SYSTEM_ORDER],
              frameon=False, ncol=2)
    savefig(fig, "fig14_time_box_by_n")

    # Scaling holdout
    sizes = [100, 200, 500]
    med_t = {s: [summarize([r for r in by_sys[s] if r["size"] == n])["median_ms"] for n in sizes] for s in SYSTEM_ORDER}
    fb_n = {s: [summarize([r for r in by_sys[s] if r["size"] == n])["fallback_pct"] for n in sizes] for s in SYSTEM_ORDER}
    vt_n = {s: [summarize([r for r in by_sys[s] if r["size"] == n])["mean_verts"] for n in sizes] for s in SYSTEM_ORDER}

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for s in SYSTEM_ORDER:
        ax.plot(sizes, med_t[s], marker="o", color=SYSTEM_COLOR[s], label=SYSTEM_LABEL[s])
    ax.set_xlabel("Graph size n")
    ax.set_ylabel("Median time (ms)")
    ax.set_title("Holdout scaling: median time grows for all systems")
    ax.set_xticks(sizes)
    ax.legend(frameon=False)
    savefig(fig, "fig15_scaling_median_time")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for s in SYSTEM_ORDER:
        ax.plot(sizes, fb_n[s], marker="o", color=SYSTEM_COLOR[s], label=SYSTEM_LABEL[s])
    ax.axhline(15, color="#2f6f4f", ls="--", lw=1, label="15% target")
    ax.set_xlabel("Graph size n")
    ax.set_ylabel("Fallback rate (%)")
    ax.set_title("Holdout fallback vs n (fixed r=2 collapses at n=500)")
    ax.set_xticks(sizes)
    ax.legend(frameon=False)
    savefig(fig, "fig16_scaling_fallback")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for s in SYSTEM_ORDER:
        ax.plot(sizes, vt_n[s], marker="o", color=SYSTEM_COLOR[s], label=SYSTEM_LABEL[s])
    ax.set_xlabel("Graph size n")
    ax.set_ylabel("Mean vertices touched")
    ax.set_title("Holdout locality vs n (RPI ≈ majority at n=200 and 500)")
    ax.set_xticks(sizes)
    ax.legend(frameon=False)
    savefig(fig, "fig17_scaling_verts")

    # OOD scaling including 2000
    s_ood_f = summarize(ood_sys["full_recompute"])
    s_ood_r = summarize(ood_sys["rpi_dsatur"])
    xs = [100, 200, 500, 2000]
    full_med = med_t["full_recompute"] + [s_ood_f["median_ms"]]
    rpi_med = med_t["rpi_dsatur"] + [s_ood_r["median_ms"]]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(xs[:3], full_med[:3], marker="o", color=SYSTEM_COLOR["full_recompute"], label="Full recompute (holdout)")
    ax.plot(xs[:3], rpi_med[:3], marker="o", color=SYSTEM_COLOR["rpi_dsatur"], label="RPI hop-shell (holdout)")
    ax.plot(xs[2:], full_med[2:], marker="s", ls="--", color=SYSTEM_COLOR["full_recompute"], label="Full (OOD n=2000)")
    ax.plot(xs[2:], rpi_med[2:], marker="s", ls="--", color=SYSTEM_COLOR["rpi_dsatur"], label="RPI (OOD n=2000)")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v))))
    ax.set_xlabel("Graph size n (log scale)")
    ax.set_ylabel("Median time (ms)")
    ax.set_title("Median time vs n: OOD n=2000 is still slower for RPI")
    ax.axvline(2000, color="#888888", ls=":", lw=0.8)
    ax.text(2000, ax.get_ylim()[1] if False else rpi_med[-1], "  OOD", color="#666666", va="bottom", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    savefig(fig, "fig18_ood_scaling_median_time")

    rpi_fb = fb_n["rpi_dsatur"] + [s_ood_r["fallback_pct"]]
    rpi_ft_n = [summarize([r for r in by_sys["rpi_dsatur"] if r["size"] == n])["first_try_pct"] for n in (100, 200, 500)]
    rpi_ft_n = rpi_ft_n + [s_ood_r["first_try_pct"]]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(xs, rpi_fb, marker="o", color=SYSTEM_COLOR["rpi_dsatur"], label="RPI fallback")
    ax.plot(xs, rpi_ft_n, marker="o", color="#2f6f4f", label="RPI first-try")
    ax.axhline(15, color="#2f6f4f", ls="--", lw=1, label="15% fallback target")
    ax.axvline(2000, color="#888888", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v))))
    ax.set_xlabel("Graph size n (log scale)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("RPI fallback stays ~65–72% through OOD n=2000")
    ax.legend(frameon=False)
    savefig(fig, "fig19_ood_fallback_firsttry")

    ratio = [r / f for r, f in zip(rpi_med, full_med)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(xs[:3], ratio[:3], marker="o", color="#3d5a80", label="Holdout")
    ax.plot(xs[2:], ratio[2:], marker="s", ls="--", color="#8b1e1e", label="Includes OOD n=2000")
    ax.axhline(1.0, color="#4a4a4a", ls=":", lw=1, label="Parity")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v))))
    ax.set_xlabel("Graph size n (log scale)")
    ax.set_ylabel("Median time ratio (RPI / full)")
    ax.set_title("RPI/full median-time ratio shrinks with n but stays >1")
    ax.legend(frameon=False)
    for x, y in zip(xs, ratio):
        ax.text(x, y, f"  {y:.1f}×", fontsize=8, va="bottom")
    savefig(fig, "fig20_ratio_time_vs_n")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(xs[:3], fb_n["rpi_dsatur"], marker="o", color=SYSTEM_COLOR["rpi_dsatur"], label="RPI holdout")
    ax.plot(xs[:3], fb_n["fixed_radius"], marker="o", color=SYSTEM_COLOR["fixed_radius"], label="Fixed r=2")
    ax.plot(xs[:3], fb_n["majority"], marker="o", color=SYSTEM_COLOR["majority"], label="Majority r=0")
    ax.plot([500, 2000], [fb_n["rpi_dsatur"][-1], s_ood_r["fallback_pct"]], marker="s", ls="--",
            color=SYSTEM_COLOR["rpi_dsatur"], label="RPI OOD")
    ax.axhline(15, color="#2f6f4f", ls="--", lw=1, label="15% target")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v))))
    ax.set_xlabel("Graph size n (log scale)")
    ax.set_ylabel("Fallback rate (%)")
    ax.set_title("Fallback vs n (fixed/majority not run at n=2000)")
    ax.legend(frameon=False, fontsize=7)
    savefig(fig, "fig21_fallback_vs_n")

    vt_ood_f = s_ood_f["mean_verts"]
    vt_ood_r = s_ood_r["mean_verts"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(sizes, vt_n["full_recompute"], marker="o", color=SYSTEM_COLOR["full_recompute"], label="Full holdout")
    ax.plot(sizes, vt_n["rpi_dsatur"], marker="o", color=SYSTEM_COLOR["rpi_dsatur"], label="RPI holdout")
    ax.plot([500, 2000], [vt_n["full_recompute"][-1], vt_ood_f], marker="s", ls="--",
            color=SYSTEM_COLOR["full_recompute"], label="Full OOD")
    ax.plot([500, 2000], [vt_n["rpi_dsatur"][-1], vt_ood_r], marker="s", ls="--",
            color=SYSTEM_COLOR["rpi_dsatur"], label="RPI OOD")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v))))
    ax.set_xlabel("Graph size n (log scale)")
    ax.set_ylabel("Mean vertices touched")
    ax.set_title("Vertices touched vs n, including OOD n=2000")
    ax.legend(frameon=False)
    savefig(fig, "fig22_verts_vs_n")

    # Paired scatter + delta hist
    rpi_map = paired_keys(by_sys["rpi_dsatur"])
    full_map = paired_keys(by_sys["full_recompute"])
    keys = sorted(set(rpi_map) & set(full_map))
    t_r = np.array([rpi_map[k]["time_ms"] for k in keys])
    t_f = np.array([full_map[k]["time_ms"] for k in keys])
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(t_f, t_r, s=8, alpha=0.25, c=SYSTEM_COLOR["rpi_dsatur"], linewidths=0)
    lim = max(t_f.max(), t_r.max()) * 1.02
    ax.plot([0, lim], [0, lim], color="#4a4a4a", ls="--", lw=1, label="y = x")
    ax.set_xlabel("Full recompute time (ms)")
    ax.set_ylabel("RPI hop-shell time (ms)")
    ax.set_title("Paired per-edit times (holdout n=2,025): RPI above the diagonal")
    ax.legend(frameon=False)
    savefig(fig, "fig23_paired_time_scatter")

    d_t = wx_results[("RPI − full", "time_ms")]["diffs"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(d_t, bins=40, color="#3d5a80", edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(d_t), color="#8b1e1e", lw=1.4, label=f"Median Δ = {np.median(d_t):+.2f} ms")
    ax.axvline(0, color="#4a4a4a", ls="--", lw=1)
    ax.set_xlabel("Δ time (RPI − full) (ms)")
    ax.set_ylabel("Edits")
    ax.set_title("Holdout paired time gap: RPI is slower on essentially every edit")
    ax.legend(frameon=False)
    savefig(fig, "fig24_delta_time_hist")

    d_ood = wx_results[("OOD RPI − full", "time_ms")]["diffs"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(d_ood, bins=24, color="#8b1e1e", edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(d_ood), color="#3d5a80", lw=1.4, label=f"Median Δ = {np.median(d_ood):+.2f} ms")
    ax.axvline(0, color="#4a4a4a", ls="--", lw=1)
    ax.set_xlabel("Δ time (RPI − full) (ms)")
    ax.set_ylabel("Edits")
    ax.set_title("OOD n=2000 paired time gap (160 edits; RPI slower on most pairs)")
    ax.legend(frameon=False)
    savefig(fig, "fig25_ood_delta_time_hist")

    # Edit-type bars
    etypes = ["shrink_list", "remove_vertex"]
    fb_et = {s: [summarize(attach_edit_filter(by_sys[s], e))["fallback_pct"] for e in etypes] for s in SYSTEM_ORDER}
    vt_et = {s: [summarize(attach_edit_filter(by_sys[s], e))["mean_verts"] for e in etypes] for s in SYSTEM_ORDER}
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    grouped_bars(ax, etypes, fb_et, "Fallback rate (%)", "Fallback by edit type (holdout)")
    savefig(fig, "fig26_edit_type_fallback")

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    grouped_bars(ax, etypes, vt_et, "Mean vertices touched", "Vertices touched by edit type (holdout)")
    savefig(fig, "fig27_edit_type_verts")

    # Stacked first-try / recovered / fallback
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    xx = np.arange(len(etypes))
    w = 0.22
    for i, sys in enumerate(SYSTEM_ORDER):
        ft_s, rec_s, fb_s = [], [], []
        for e in etypes:
            subset = attach_edit_filter(by_sys[sys], e)
            att = np.array([int(r["attempts"]) for r in subset])
            fb = np.array([bool(r["fallback_triggered"]) for r in subset])
            n = len(subset)
            ft_s.append(100 * (att == 0).mean() if n else 0)
            rec_s.append(100 * ((att > 0) & (~fb)).mean() if n else 0)
            fb_s.append(100 * fb.mean() if n else 0)
        offs = xx + (i - 1.5) * w
        ax.bar(offs, ft_s, w, color="#2f6f4f", label="First-try" if i == 0 else None)
        ax.bar(offs, rec_s, w, bottom=ft_s, color="#c4a35a", label="Recovered (retry, no FB)" if i == 0 else None)
        ax.bar(offs, fb_s, w, bottom=np.array(ft_s) + np.array(rec_s), color="#8b1e1e",
               label="Fallback" if i == 0 else None)
    ax.set_xticks(xx)
    ax.set_xticklabels(etypes)
    ax.set_ylabel("Share of edits (%)")
    ax.set_title("First-try vs recovered vs fallback by edit type (grouped by system L→R)")
    ax.legend(frameon=False)
    # system tick annotation
    ax.set_xlabel("Edit type  ·  bars in each group: Full, Fixed r=2, Majority, RPI")
    savefig(fig, "fig28_firsttry_fallback_stacked")

    # Validity overlap
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    keys_v = sorted(set(paired_keys(by_sys["full_recompute"])) & set(paired_keys(by_sys["rpi_dsatur"])))
    fmap = paired_keys(by_sys["full_recompute"])
    rmap = paired_keys(by_sys["rpi_dsatur"])
    both_ok = sum(1 for k in keys_v if fmap[k]["valid"] and rmap[k]["valid"])
    both_bad = sum(1 for k in keys_v if (not fmap[k]["valid"]) and (not rmap[k]["valid"]) )
    disagree = sum(1 for k in keys_v if bool(fmap[k]["valid"]) != bool(rmap[k]["valid"]))
    ax.bar(["Both valid", "Both infeasible", "Disagree"], [both_ok, both_bad, disagree],
           color=["#2f6f4f", "#8b1e1e", "#8a6d1b"])
    ax.set_ylabel("Paired holdout edits")
    ax.set_title("Validity disagreements with full recompute are essentially zero")
    for i, v in enumerate([both_ok, both_bad, disagree]):
        ax.text(i, v + 8, str(v), ha="center", fontsize=9)
    savefig(fig, "fig29_validity_shared")

    # OOD validity
    keys_o = sorted(set(paired_keys(ood_sys["full_recompute"])) & set(paired_keys(ood_sys["rpi_dsatur"])))
    fo = paired_keys(ood_sys["full_recompute"])
    ro = paired_keys(ood_sys["rpi_dsatur"])
    o_ok = sum(1 for k in keys_o if fo[k]["valid"] and ro[k]["valid"])
    o_bad = sum(1 for k in keys_o if (not fo[k]["valid"]) and (not ro[k]["valid"]))
    o_dis = sum(1 for k in keys_o if bool(fo[k]["valid"]) != bool(ro[k]["valid"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(["Both valid", "Both infeasible", "Disagree"], [o_ok, o_bad, o_dis],
           color=["#2f6f4f", "#8b1e1e", "#8a6d1b"])
    ax.set_ylabel("Paired OOD edits")
    ax.set_title("OOD n=2000: 18 shared infeasible edits, 0 disagreements")
    for i, v in enumerate([o_ok, o_bad, o_dis]):
        ax.text(i, v + 1, str(v), ha="center", fontsize=9)
    savefig(fig, "fig30_ood_validity_shared")

    # Training lineage (historical, not recomputed)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = [n for n, _ in LINEAGE]
    vals = [100 * v for _, v in LINEAGE]
    colors = ["#6b6b6b"] * len(names)
    colors[3] = "#3d5a80"
    ax.barh(names, vals, color=colors)
    ax.axvline(72, color="#8b1e1e", ls="--", lw=1, label="72% target")
    ax.axvline(100 * report["majority_class_baseline_accuracy"], color="#4a4a4a", ls=":", lw=1, label="Majority baseline")
    ax.set_xlabel("6-class test accuracy (%)")
    ax.set_xlim(50, 80)
    ax.set_title("Training lineage (Context.md; only hop-shell is the deployed checkpoint)")
    ax.legend(frameon=False)
    ax.invert_yaxis()
    savefig(fig, "fig31_training_lineage")

    # Cost of misprediction
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(["First-try\n(attempts=0)", "Retried\n(attempts>0)"],
           [ft["median_ms"], rt["median_ms"]], color=["#2f6f4f", "#8b1e1e"], width=0.5)
    ax.set_ylabel("Median time (ms)")
    ax.set_title("RPI: a retry roughly doubles median time (holdout)")
    savefig(fig, "fig32_cost_of_misprediction")

    # Radius used on first-try
    from collections import Counter
    ru = Counter(str(r["radius_used"]) for r in rpi_ft)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    labs_r = sorted(ru, key=lambda k: (k != "0", k))
    ax.bar(labs_r, [ru[k] for k in labs_r], color="#3d5a80")
    ax.set_ylabel("First-try edits")
    ax.set_xlabel("radius_used")
    ax.set_title("First-try RPI almost always uses radius 0")
    savefig(fig, "fig33_firsttry_radius_used")

    # OOD edit-type (RPI vs full only)
    ood_et_headers = ["Edit type", "System", "n", "Fallback %", "First-try %", "Median ms", "Mean verts"]
    ood_et_rows = []
    for etype in etypes:
        for sys in ("full_recompute", "rpi_dsatur"):
            subset = attach_edit_filter(ood_sys[sys], etype)
            if not subset:
                continue
            s = summarize(subset)
            ood_et_rows.append([
                etype, SYSTEM_LABEL[sys], s["n"], f"{s['fallback_pct']:.2f}",
                f"{s['first_try_pct']:.2f}", f"{s['median_ms']:.2f}", f"{s['mean_verts']:.2f}",
            ])
    write_csv(OUT / "tab10_ood_edit_type.csv", ood_et_headers, ood_et_rows)
    table_fig(ood_et_headers, ood_et_rows, "tab10_ood_edit_type", "OOD n=2000 metrics by edit type")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ood_fb = {s: [summarize(attach_edit_filter(ood_sys[s], e))["fallback_pct"] for e in etypes]
              for s in ("full_recompute", "rpi_dsatur")}
    grouped_bars(ax, etypes, ood_fb, "Fallback rate (%)", "OOD n=2000 fallback by edit type (RPI vs full only)")
    savefig(fig, "fig34_ood_edit_type_fallback")

    # KPI strip as a simple figure
    fig, ax = plt.subplots(figsize=(8.4, 2.4))
    ax.axis("off")
    kpis = [
        (f"{100*report['test_accuracy']:.2f}%", "6-class test acc\n(vs majority "
         f"{100*report['majority_class_baseline_accuracy']:.2f}%)"),
        (f"{s_rpi['median_ms']:.1f} ms", f"RPI median /edit\n(full: {s_full['median_ms']:.1f} ms)"),
        (f"{s_rpi['fallback_pct']:.1f}%", "RPI fallback\n(target <15%)"),
        (f"{100*(s_rpi['mean_verts']/s_fix['mean_verts']-1):.0f}%", "RPI vs fixed r=2\nmean vertices"),
    ]
    for i, (val, lab) in enumerate(kpis):
        x0 = 0.02 + i * 0.25
        ax.add_patch(FancyBboxPatch((x0, 0.18), 0.22, 0.68, boxstyle="round,pad=0.02",
                                    facecolor="#f4f4f4", edgecolor="#d0d0d0", transform=ax.transAxes))
        ax.text(x0 + 0.11, 0.62, val, ha="center", va="center", fontsize=13, weight="bold",
                transform=ax.transAxes)
        ax.text(x0 + 0.11, 0.36, lab, ha="center", va="center", fontsize=7.5, color="#444444",
                transform=ax.transAxes)
    ax.set_title("Holdout headline numbers (seed 0; 45 graphs; 2,025 edits)", loc="left")
    savefig(fig, "fig00_kpi_strip")

    # Limitations table
    lim_headers = ["Gap", "Status"]
    lim_rows = [
        ["5-seed Wilcoxon", "Missing — seed 0 only"],
        ["n=300 extra size", "Not generated"],
        ["72% 6-class accuracy", f"Not met (deployed {100*report['test_accuracy']:.2f}%)"],
        ["Wall-clock win vs full", "Not observed at n≤500 or OOD n=2000"],
        ["n=2000 OOD RPI vs full", f"Ran: 8×20; RPI median {s_ood_r['median_ms']:.1f} vs {s_ood_f['median_ms']:.1f} ms"],
        ["Class-1 / full recall", "0.000 on sequential test"],
        ["Real / geometric topologies", "Synthetic ER only"],
        ["results/raw_outcomes.jsonl", "Do not mix — different RPI"],
    ]
    write_csv(OUT / "tab11_limitations.csv", lim_headers, lim_rows)
    table_fig(lim_headers, lim_rows, "tab11_limitations", "Limitations and evaluation gaps")

    manifest = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return {
        "overall": overall,
        "report": report,
        "ood_full": s_ood_f,
        "ood_rpi": s_ood_r,
        "wx": wx_results,
        "validity_holdout": (both_ok, both_bad, disagree),
        "validity_ood": (o_ok, o_bad, o_dis),
        "first_try": ft,
        "retried": rt,
        "edit_map_n": len(data["edit_map"]),
        "ood_edit_map_n": len(data["ood_edit_map"]),
        "out": str(OUT),
        "n_files": len(manifest),
    }


if __name__ == "__main__":
    info = generate_all()
    print("Wrote", info["n_files"], "files to", info["out"])
    print("Holdout RPI median ms", info["overall"]["rpi_dsatur"]["median_ms"])
    print("Holdout full median ms", info["overall"]["full_recompute"]["median_ms"])
    print("Test acc", info["report"]["test_accuracy"])
    print("Edit-type keys", info["edit_map_n"], "OOD keys", info["ood_edit_map_n"])
