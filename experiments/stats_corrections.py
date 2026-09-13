"""Multiple-testing corrections for the RPI-DSATUR Wilcoxon battery."""

from __future__ import annotations


def holm_bonferroni_correct(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """
    Given raw p-values from multiple hypothesis tests, return Holm-Bonferroni
    adjusted decisions. Sort ascending, adjust as
    p_adj_i = max(p_adj_1..i, p_i * (n - rank + 1)), enforce monotonicity.
    Return one dict per input p-value (ORIGINAL order preserved), each with:
      {"p_raw": float, "p_holm": float, "reject_at_0.05": bool, "rank": int}
    """
    n = len(p_values)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: (p_values[i], i))
    p_holm_sorted: list[float] = []
    for rank0, idx in enumerate(order):
        rank = rank0 + 1  # 1-based among ascending p-values
        raw = float(p_values[idx])
        candidate = raw * (n - rank + 1)
        if p_holm_sorted:
            candidate = max(p_holm_sorted[-1], candidate)
        p_holm_sorted.append(min(candidate, 1.0))

    # Enforce monotonicity in the reverse direction as well (standard Holm).
    for i in range(n - 2, -1, -1):
        p_holm_sorted[i] = min(p_holm_sorted[i], p_holm_sorted[i + 1])

    by_orig = [None] * n
    for rank0, idx in enumerate(order):
        p_holm = p_holm_sorted[rank0]
        by_orig[idx] = {
            "p_raw": float(p_values[idx]),
            "p_holm": float(p_holm),
            "reject_at_0.05": bool(p_holm <= alpha),
            "rank": rank0 + 1,
        }
    return by_orig  # type: ignore[return-value]


if __name__ == "__main__":
    # Smoke test against a known ordering.
    demo = [0.01, 0.04, 0.03, 0.005]
    out = holm_bonferroni_correct(demo)
    for row in out:
        print(row)
