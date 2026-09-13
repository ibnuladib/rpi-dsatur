"""Build notebooks/results_analysis.ipynb — one md header + producing code cell per item."""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "results_analysis.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip() + "\n"))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))


md(
    "# RPI-DSATUR results analysis\n\n"
    "Produces the 11 figures/tables required by final_spec.md. "
    "Reads existing matrix JSONLs plus newly generated "
    "results/fixed_radius_sweep.jsonl and results/hop_coverage.csv. "
    "Does not read results/raw_outcomes.jsonl.\n\n"
    "Re-run: `python notebooks/results_analysis.py` or execute all cells below."
)

md("## Setup")
code(
    """
import os
import sys
from pathlib import Path

from IPython.display import Image, display
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "results" / "matrix_static_systems.jsonl").exists():
    ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

DATA_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR = ROOT / "results" / "tables"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

import results_analysis as ra
# Force module paths to this ROOT (in case the kernel cwd differed at import).
ra.ROOT = ROOT
ra.DATA_DIR = DATA_DIR
ra.FIGURES_DIR = FIGURES_DIR
ra.TABLES_DIR = TABLES_DIR

print("ROOT =", ROOT)
print("FIGURES_DIR =", FIGURES_DIR)
print("TABLES_DIR =", TABLES_DIR)
"""
)

ITEMS = [
    (
        "fig1_combined_scaling",
        "Figure 1",
        "Median time_ms (log y) vs n (log x) for full_recompute and rpi_dsatur "
        "(single combined holdout+OOD chart).",
    ),
    (
        "fig2_hop_coverage_vs_n",
        "Figure 2",
        "Mean 4-hop coverage_fraction with IQR vs n (log x).",
    ),
    (
        "fig3_fallback_vs_n",
        "Figure 3",
        "Fallback rate (%) vs n; all 4 systems for n<=500, RPI+full at n=2000.",
    ),
    (
        "fig4_vertices_vs_n",
        "Figure 4",
        "Mean vertices_touched vs n for every system available at each size.",
    ),
    (
        "fig5_edit_type_breakdown",
        "Figure 5",
        "Grouped bars: fallback rate and mean vertices_touched by edit_type x system.",
    ),
    (
        "fig6_confusion_heatmap",
        "Figure 6",
        "Heatmap of the hop-shell classifier confusion matrix (rows=true, cols=predicted).",
    ),
    (
        "fig7_per_class_recall",
        "Figure 7",
        "Per-class recall bars annotated with support count n.",
    ),
    (
        "fig8_misprediction_cost",
        "Figure 8",
        "RPI time_ms and vertices_touched for first-try (attempts==0) vs retried (attempts>0).",
    ),
    (
        "fig9_fixed_radius_sweep",
        "Figure 9",
        "Median time_ms and fallback rate vs fixed radius in {1,2,3,4}.",
    ),
    (
        "table_wilcoxon_full",
        "Table 10",
        "Six edit-level Wilcoxon rows + Holm-Bonferroni + parallel graph-level tests.",
    ),
    (
        "table_headline_summary",
        "Table 11",
        "Four-system headline summary re-rendered from the holdout JSONLs.",
    ),
]

md(
    "## Produce all artifacts\n\n"
    "Runs the full Part-2 generator (writes every figure PNG and table CSV). "
    "Subsequent cells display each artifact."
)
code(
    """
info = ra.generate_all()
print(info)
"""
)

for stem, title, blurb in ITEMS:
    md(f"## {title} — `{stem}`\n\n{blurb}")
    if stem.startswith("fig"):
        code(
            f"""
p = FIGURES_DIR / "{stem}.png"
assert p.exists(), f"missing {{p}} — re-run the produce cell"
display(Image(filename=str(p)))
print(p)
"""
        )
    else:
        code(
            f"""
p = TABLES_DIR / "{stem}.csv"
assert p.exists(), f"missing {{p}} — re-run the produce cell"
df = pd.read_csv(p)
display(df)
print(p)
"""
        )

md("## Done\n\nArtifacts: `results/figures/*.png`, `results/tables/*.csv`.")

nb["cells"] = cells
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
