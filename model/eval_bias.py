"""Post-hoc evaluation of an existing checkpoint WITHOUT retraining.

Fits a per-class additive logit bias on the VAL split only, then reports test
metrics with and without it. Two bias variants:
  - unconstrained: maximize val accuracy (train.fit_class_bias)
  - constrained:   maximize val accuracy subject to class-1 recall and macro-F1
                   floors (post-hoc analog of joint checkpoint selection)

CLI:
  python -m model.eval_bias --config model/config_r1.yaml \
      --checkpoint model/checkpoints/radius_gnn.pt \
      --labels data/labels/labels_r1_aug.pt \
      --out model/reports/r1_ce_bias_only/bias_eval_report.json
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from model.radius_gnn import create_model_from_config
from model.train import evaluate, fit_class_bias, load_and_split_data_from_list
from labels.build_labels import BUCKET_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit_class_bias_constrained(logits, targets, num_classes,
                               min_class1_recall=0.30, min_macro_f1=0.40,
                               grid=None, n_passes=3):
    """Coordinate ascent on accuracy, but a bias vector is admissible only if
    class-1 recall >= min_class1_recall AND macro-F1 >= min_macro_f1 on the
    same (validation) split. Returns zeros if the zero bias itself is
    inadmissible or no admissible bias beats it."""
    if grid is None:
        grid = np.linspace(-2.0, 2.0, 41)

    y = np.asarray(targets)
    labels = list(range(num_classes))
    bias = np.zeros(num_classes, dtype=np.float64)

    def stats(b):
        preds = (logits + b).argmax(axis=1)
        acc = float((preds == y).mean())
        rec1 = float(recall_score(y, preds, labels=labels, average=None, zero_division=0)[1])
        mf1 = float(f1_score(y, preds, labels=labels, average="macro", zero_division=0))
        return acc, rec1, mf1

    def admissible(b):
        _, rec1, mf1 = stats(b)
        return rec1 >= min_class1_recall and mf1 >= min_macro_f1

    base, base_rec1, base_mf1 = stats(bias)
    logger.info(f"[bias-c] zero bias: val_acc={base:.4f}, "
                f"class1_recall={base_rec1:.4f}, macro_f1={base_mf1:.4f}")
    if not admissible(bias):
        logger.warning("[bias-c] zero bias violates the floors; returning zeros")
        return np.zeros(num_classes, dtype=np.float32)

    best = base
    for _ in range(n_passes):
        improved = False
        for c in range(1, num_classes):
            keep, cur = bias[c], best
            for g in grid:
                bias[c] = g
                acc, _, _ = stats(bias)
                if acc > cur and admissible(bias):
                    cur, keep = acc, g
            bias[c] = keep
            if cur > best:
                best, improved = cur, True
        if not improved:
            break

    if best <= base:
        logger.info("[bias-c] no admissible bias improved val accuracy; using zeros")
        return np.zeros(num_classes, dtype=np.float32)

    logger.info(f"[bias-c] val accuracy {base:.4f} -> {best:.4f}")
    logger.info(f"[bias-c] fitted bias: {[round(float(b), 2) for b in bias]}")
    return bias.astype(np.float32)


def metrics_from_preds(targets, preds, num_classes):
    labels = list(range(num_classes))
    total = len(targets)
    cm = confusion_matrix(targets, preds, labels=labels)
    merged_correct = sum(
        1 for p, t in zip(preds, targets) if p == t or (p in (0, 1) and t in (0, 1))
    )
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, preds, labels=labels, average="macro", zero_division=0)),
        "per_class_precision": [float(v) for v in precision_score(targets, preds, labels=labels, average=None, zero_division=0)],
        "per_class_recall": [float(v) for v in recall_score(targets, preds, labels=labels, average=None, zero_division=0)],
        "per_class_f1": [float(v) for v in f1_score(targets, preds, labels=labels, average=None, zero_division=0)],
        "under_prediction_rate": sum(1 for p, t in zip(preds, targets) if p < t) / total,
        "over_prediction_rate": sum(1 for p, t in zip(preds, targets) if p > t) / total,
        "merged_01_accuracy": merged_correct / total,
        "confusion_matrix": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-class1-recall", type=float, default=0.30)
    ap.add_argument("--min-macro-f1", type=float, default=0.40)
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

    val_loader = PyGDataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = PyGDataLoader(test_dataset, batch_size=64, shuffle=False)

    num_classes = len(BUCKET_NAMES)
    in_dim = data_list[0].x.shape[1]
    model = create_model_from_config(config, in_dim, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    logger.info(f"Loaded checkpoint {args.checkpoint}")

    val_metrics = evaluate(model, val_loader, device, num_classes=num_classes)
    test_metrics = evaluate(model, test_loader, device, num_classes=num_classes)
    logger.info(f"Val: acc={val_metrics['accuracy']:.4f}, macro_f1={val_metrics['macro_f1']:.4f}, "
                f"class1_recall={val_metrics['per_class_recall'][1]:.4f}")
    logger.info(f"Test (no bias): acc={test_metrics['accuracy']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}, "
                f"class1_recall={test_metrics['per_class_recall'][1]:.4f}")

    bias_u = fit_class_bias(val_metrics["logits"], val_metrics["targets"], num_classes)
    bias_c = fit_class_bias_constrained(
        val_metrics["logits"], val_metrics["targets"], num_classes,
        min_class1_recall=args.min_class1_recall, min_macro_f1=args.min_macro_f1,
    )

    out = {
        "checkpoint": args.checkpoint,
        "labels": args.labels,
        "config": args.config,
        "val": metrics_from_preds(val_metrics["targets"], val_metrics["preds"], num_classes),
        "test_no_bias": metrics_from_preds(test_metrics["targets"], test_metrics["preds"], num_classes),
        "bias_unconstrained": [float(b) for b in bias_u],
        "test_bias_unconstrained": metrics_from_preds(
            test_metrics["targets"],
            (test_metrics["logits"] + bias_u).argmax(axis=1).tolist(),
            num_classes),
        "bias_constrained": [float(b) for b in bias_c],
        "test_bias_constrained": metrics_from_preds(
            test_metrics["targets"],
            (test_metrics["logits"] + bias_c).argmax(axis=1).tolist(),
            num_classes),
        "confusion_matrix_labels": list(BUCKET_NAMES),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    for key in ("test_no_bias", "test_bias_unconstrained", "test_bias_constrained"):
        m = out[key]
        logger.info(f"{key}: acc={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}, "
                    f"class1_recall={m['per_class_recall'][1]:.4f}, "
                    f"merged01={m['merged_01_accuracy']:.4f}")
    logger.info(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
