"""Per-class threshold tuning on a frozen checkpoint (no retrain).

pred = argmax(logits - t). t is fit on VAL accuracy only, then applied to test.

Unlike train.fit_class_bias this does not pin class 0, and it starts with a
joint 2D grid on (t[0], t[4]) so the 4↔0 tradeoff is searched together
instead of one coordinate at a time.

CLI:
  python -m model.eval_thresholds --config model/config.yaml \
      --checkpoint model/checkpoints/radius_gnn_hopshell.pt \
      --labels data/labels/labels_pre_aug_backup.pt \
      --out model/reports/hop_shell/threshold_eval.json
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader as PyGDataLoader

from labels.build_labels import BUCKET_NAMES
from model.eval_bias import metrics_from_preds
from model.radius_gnn import create_model_from_config
from model.train import evaluate, load_and_split_data_from_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _acc(logits, y, t):
    return float(((logits - t).argmax(axis=1) == y).mean())


def fit_thresholds(logits, targets, num_classes, grid=None, n_passes=3):
    """Maximize val accuracy over t in pred = argmax(logits - t).

    1. Joint coarse grid on (t[0], t[4]) — 4 vs 0 boundary.
    2. Coordinate descent over all classes (class 0 not pinned).
    """
    if grid is None:
        grid = np.linspace(-1.5, 1.5, 31)  # step 0.1
    y = np.asarray(targets)
    t = np.zeros(num_classes, dtype=np.float64)
    base = _acc(logits, y, t)
    best = base
    logger.info(f"[thr] zero t: val_acc={base:.4f}")

    # Joint 2D search on classes 0 and 4.
    t0_keep, t4_keep = 0.0, 0.0
    for g0 in grid:
        t[0] = g0
        for g4 in grid:
            t[4] = g4
            s = _acc(logits, y, t)
            if s > best:
                best, t0_keep, t4_keep = s, g0, g4
    t[0], t[4] = t0_keep, t4_keep
    logger.info(f"[thr] after 2D (t0, t4)=({t[0]:.2f}, {t[4]:.2f}): val_acc={best:.4f}")

    local = np.linspace(-1.0, 1.0, 21)
    for _ in range(n_passes):
        improved = False
        for c in range(num_classes):
            keep, cur = t[c], best
            for d in local:
                cand = np.clip(keep + d, -2.0, 2.0)
                t[c] = cand
                s = _acc(logits, y, t)
                if s > cur:
                    cur, keep = s, cand
            t[c] = keep
            if cur > best:
                best, improved = cur, True
        if not improved:
            break

    if best <= base:
        logger.info("[thr] no threshold vector beat zeros; using zeros")
        return np.zeros(num_classes, dtype=np.float32), base

    logger.info(f"[thr] val accuracy {base:.4f} -> {best:.4f}")
    logger.info(f"[thr] t={ [round(float(x), 3) for x in t] }")
    return t.astype(np.float32), best


def _cell_4_to_0(cm):
    # rows=true, cols=pred; class 4 -> class 0
    return int(cm[4][0]) if len(cm) > 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="model/config.yaml")
    ap.add_argument("--checkpoint", default="model/checkpoints/radius_gnn_hopshell.pt")
    ap.add_argument("--labels", default="data/labels/labels_pre_aug_backup.pt")
    ap.add_argument("--out", default="model/reports/hop_shell/threshold_eval.json")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    seed = config.get("seed", 0)
    data_config = config["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_list = torch.load(args.labels, weights_only=False)
    logger.info(f"Loaded {len(data_list)} labeled examples from {args.labels}")
    _, val_dataset, test_dataset = load_and_split_data_from_list(
        data_list, data_config["split"], data_config["split_by"], seed
    )
    logger.info(f"val n={len(val_dataset)}, test n={len(test_dataset)}")

    val_loader = PyGDataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = PyGDataLoader(test_dataset, batch_size=64, shuffle=False)

    num_classes = len(BUCKET_NAMES)
    in_dim = data_list[0].x.shape[1]
    gf_dim = int(data_list[0].global_features.shape[-1])
    model = create_model_from_config(
        config, in_dim, num_classes=num_classes, num_global_features=gf_dim
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    logger.info(f"Loaded checkpoint {args.checkpoint}")

    val_m = evaluate(model, val_loader, device, num_classes=num_classes)
    test_m = evaluate(model, test_loader, device, num_classes=num_classes)

    t, val_acc_t = fit_thresholds(val_m["logits"], val_m["targets"], num_classes)
    test_preds_t = (test_m["logits"] - t).argmax(axis=1).tolist()

    baseline = metrics_from_preds(test_m["targets"], test_m["preds"], num_classes)
    tuned = metrics_from_preds(test_m["targets"], test_preds_t, num_classes)
    val_tuned = metrics_from_preds(
        val_m["targets"], (val_m["logits"] - t).argmax(axis=1).tolist(), num_classes
    )

    out = {
        "checkpoint": args.checkpoint,
        "labels": args.labels,
        "config": args.config,
        "test_n": len(test_dataset),
        "thresholds": [float(x) for x in t],
        "val_acc_zero": float(val_m["accuracy"]),
        "val_acc_tuned": float(val_acc_t),
        "val_tuned": val_tuned,
        "test_zero": baseline,
        "test_tuned": tuned,
        "four_to_zero_before": _cell_4_to_0(baseline["confusion_matrix"]),
        "four_to_zero_after": _cell_4_to_0(tuned["confusion_matrix"]),
        "class4_recall_before": baseline["per_class_recall"][4],
        "class4_recall_after": tuned["per_class_recall"][4],
        "confusion_matrix_labels": list(BUCKET_NAMES),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(
        f"test zero: acc={baseline['accuracy']:.4f} class4_R={baseline['per_class_recall'][4]:.3f} "
        f"4→0={out['four_to_zero_before']}"
    )
    logger.info(
        f"test tuned: acc={tuned['accuracy']:.4f} class4_R={tuned['per_class_recall'][4]:.3f} "
        f"4→0={out['four_to_zero_after']}"
    )
    logger.info(f"CM tuned:\n{np.array(tuned['confusion_matrix'])}")
    logger.info(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
