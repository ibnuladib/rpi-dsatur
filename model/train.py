"""
Training script for RadiusGNN (Section 6 / Appendix A.5).

Selection/early-stopping is done on validation MACRO-F1, not raw accuracy
or macro-recall alone. Macro-recall-only selection was tried and rejected:
it let the model spam a single class (e.g. always predicting bucket "2")
to inflate recall while precision collapsed -- macro-F1 punishes that.

Class weighting defaults to sqrt-inverse-frequency, not raw inverse-
frequency: raw inverse-frequency produced ~24x weight ratios on the
rarest class (9 examples) vs the majority (215 examples), which is a
likely source of the training instability seen between runs.

Train-set per-class recall/F1 is logged alongside validation every epoch,
specifically to distinguish two different failure modes for weak classes:
  - if TRAIN recall/F1 on a class is also near zero, the model cannot even
    memorize that class -> likely a feature/receptive-field insufficiency
    (e.g. the 3-hop window can't see far enough to tell "8" from "full").
  - if TRAIN is high but VAL/TEST is near zero, that's overfitting on very
    few rare-class examples, not a capacity problem.
"""

import os
import json
import logging
import shutil
from collections import Counter

import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
)

from model.radius_gnn import create_model_from_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bucket names are imported from labels.build_labels, NOT redefined here --
# a hardcoded local copy of this list is exactly what went stale last time
# the bucket scheme changed (this file kept showing the old geometric names
# after build_labels.py switched to the linear scheme). Single source of
# truth from now on.
from labels.build_labels import BUCKET_NAMES, NUM_BUCKETS as NUM_CLASSES


class OrdinalExpectedCostLoss(nn.Module):
    """CE plus expected |bucket_index - target| under softmax.

    Buckets are ordered 0 < 1 < 2 < 3 < 4 < full. Far-apart mistakes
    (true 4 predicted 0, true full predicted 0) cost more than adjacent
    ones. Does not apply class-frequency weights -- sampler-only rebalance
    is unchanged.
    """

    def __init__(self, num_classes: int, mae_lambda: float = 0.5, weight=None):
        super().__init__()
        self.mae_lambda = mae_lambda
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.register_buffer(
            "class_idx", torch.arange(num_classes, dtype=torch.float32)
        )

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=-1)
        expected = (probs * self.class_idx).sum(dim=-1)
        mae = (expected - targets.float()).abs().mean()
        return ce + self.mae_lambda * mae


class AsymmetricOrdinalLoss(nn.Module):
    """CE plus expected asymmetric ordinal cost under softmax.

    cost[true, pred] = 2 * max(0, true - pred) + |true - pred|
    Under-prediction (pred < true) costs 3*|diff|; over-prediction costs |diff|.
    Targets the 4→0 leak (cost 12 vs 4 for the reverse 0→4 cell).
    """

    def __init__(self, num_classes: int, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.register_buffer(
            "class_idx", torch.arange(num_classes, dtype=torch.float32)
        )

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=-1)
        diff = targets.float().unsqueeze(1) - self.class_idx.unsqueeze(0)
        cost = 2.0 * diff.clamp(min=0) + diff.abs()
        expected = (probs * cost).sum(dim=-1).mean()
        return ce + expected


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------

def load_and_split_data_from_list(data_list: list, split: list[float], split_by: str, seed: int):
    """Split an already-loaded (and possibly filtered/remapped) list of
    examples by graph_id (never by individual edit). Factored out from
    load_and_split_data so filter_and_remap_dataset() can run in between
    loading and splitting (see the excluded_buckets experiment in train())."""
    if split_by != "graph_id":
        raise ValueError(f"split_by must be 'graph_id' per methodology Section 8.2, got {split_by!r}")

    graph_to_indices: dict[str, list[int]] = {}
    for i, data in enumerate(data_list):
        gid = getattr(data, "graph_id", "")
        graph_to_indices.setdefault(gid, []).append(i)

    graph_ids = list(graph_to_indices.keys())
    logger.info(f"Unique graphs: {len(graph_ids)}")

    train_ids, temp_ids = train_test_split(
        graph_ids, train_size=split[0], random_state=seed
    )
    val_size = split[1] / (split[1] + split[2])
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=val_size, random_state=seed
    )

    train_indices = [i for gid in train_ids for i in graph_to_indices[gid]]
    val_indices = [i for gid in val_ids for i in graph_to_indices[gid]]
    test_indices = [i for gid in test_ids for i in graph_to_indices[gid]]

    logger.info(
        f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    return (
        Subset(data_list, train_indices),
        Subset(data_list, val_indices),
        Subset(data_list, test_indices),
    )


def load_and_split_data(labels_path: str, split: list[float], split_by: str, seed: int):
    """Backward-compatible wrapper: load from disk, then split (no filtering)."""
    data_list = torch.load(labels_path, weights_only=False)
    logger.info(f"Loaded {len(data_list)} labeled examples")
    return load_and_split_data_from_list(data_list, split, split_by, seed)


# ---------------------------------------------------------------------------
# Ad-hoc experiment: drop hard classes ("1", "full") to see achievable
# score on the remaining, better-supported buckets.
#
# This is a TOGGLE, not a permanent scheme change -- set
# train.excluded_buckets: [] in config.yaml to disable and go back to all
# 6 classes. When non-empty, excluded examples are dropped and the
# remaining bucket indices are REMAPPED to a contiguous range (e.g.
# {0,2,3,4} -> {0,1,2,3}) so the model's softmax size matches what's
# actually being classified, rather than training a 6-way head on 4
# populated classes.
# ---------------------------------------------------------------------------

def filter_and_remap_dataset(data_list, exclude_names: list[str]):
    """Drop examples whose bucket name is in exclude_names, and remap the
    remaining bucket indices to a contiguous range starting at 0.

    Returns (filtered_data_list, new_bucket_names). Mutates each kept
    example's .bucket in place to the new contiguous index -- data_list
    objects are reused across calls, so only ever call this ONCE per
    loaded data_list per process (train() below does this exactly once,
    right after torch.load).
    """
    if not exclude_names:
        return data_list, list(BUCKET_NAMES)

    exclude_indices = {BUCKET_NAMES.index(n) for n in exclude_names}
    new_bucket_names = [n for n in BUCKET_NAMES if n not in exclude_names]

    remap: dict[int, int] = {}
    new_idx = 0
    for old_idx, name in enumerate(BUCKET_NAMES):
        if old_idx in exclude_indices:
            continue
        remap[old_idx] = new_idx
        new_idx += 1

    filtered = []
    for d in data_list:
        if d.bucket in exclude_indices:
            continue
        d.bucket = remap[d.bucket]
        filtered.append(d)

    logger.info(
        f"[excluded_buckets experiment] dropped {exclude_names}: "
        f"{len(data_list)} -> {len(filtered)} examples"
    )
    logger.info(f"[excluded_buckets experiment] remapped bucket names: {new_bucket_names}")
    return filtered, new_bucket_names


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------

# Weight ratio (max/min among present classes) above which we warn that
# raw inverse-frequency weighting may destabilize training.
_EXTREME_RATIO_WARN_THRESHOLD = 10.0


def compute_class_weights(data_list, num_classes: int = NUM_CLASSES, mode: str = "sqrt_inv_freq"):
    """Class weights that handle empty/rare buckets safely.

    mode="inv_freq": raw inverse frequency. Can produce extreme ratios when
        one class is much rarer than another (e.g. 9 vs 215 examples ->
        ~24x weight ratio), which can cause noisy/unstable gradients on the
        rare-class batches and destabilize the rest of training.
    mode="sqrt_inv_freq" (default, recommended): sqrt of inverse frequency
        -- still up-weights rare classes, without the extreme swings.

    Empty buckets get weight 0 (no loss contribution). Non-empty buckets
    are rescaled so the average weight over present classes is 1.0.
    """
    if mode not in ("inv_freq", "sqrt_inv_freq"):
        raise ValueError(f"Unknown class-weighting mode: {mode!r}")

    counts = torch.zeros(num_classes)
    for data in data_list:
        bucket = getattr(data, "bucket", 0)
        counts[bucket] += 1

    weights = torch.zeros(num_classes)
    present = counts > 0
    n_present = int(present.sum().item())
    if n_present == 0:
        return weights

    if mode == "sqrt_inv_freq":
        weights[present] = 1.0 / counts[present].sqrt()
    else:
        weights[present] = 1.0 / counts[present]

    # Rescale so the average weight over present classes is 1.0
    weights[present] = weights[present] / weights[present].sum() * n_present

    logger.info(f"Class weighting mode: {mode}")
    logger.info(f"Class counts:  {dict(zip(BUCKET_NAMES, counts.int().tolist()))}")
    logger.info(f"Class weights: {dict(zip(BUCKET_NAMES, [round(float(w), 4) for w in weights.tolist()]))}")

    # Explicit warning if raw inverse-frequency produces an extreme ratio --
    # surfaces the exact risk we diagnosed, instead of leaving it silent.
    present_weights = weights[present]
    if present_weights.numel() > 1:
        ratio = float(present_weights.max() / present_weights.min())
        if mode == "inv_freq" and ratio > _EXTREME_RATIO_WARN_THRESHOLD:
            logger.warning(
                f"Class weight ratio is {ratio:.1f}x (mode='inv_freq'). "
                f"This is a likely source of training instability -- "
                f"consider mode='sqrt_inv_freq' instead."
            )

    return weights


# ---------------------------------------------------------------------------
# Minority-class oversampling
# ---------------------------------------------------------------------------

def make_weighted_sampler(dataset, num_classes: int = NUM_CLASSES) -> WeightedRandomSampler:
    """Per-example sampling weight = inverse sqrt frequency of its bucket.

    This is a DIFFERENT mechanism from compute_class_weights() above:
    class weighting scales the LOSS MAGNITUDE for a class, but each rare
    example is still only seen once per epoch. This sampler instead makes
    rare-class examples (e.g. bucket "1", "full") appear MORE OFTEN per
    epoch (with replacement), giving the optimizer more actual gradient
    steps informed by them. The two are complementary and both are applied
    together in train() below.
    """
    counts = torch.zeros(num_classes)
    buckets = [getattr(dataset[i], "bucket", 0) for i in range(len(dataset))]
    for b in buckets:
        counts[b] += 1

    weight_per_class = 1.0 / counts.sqrt().clamp(min=1)
    sample_weights = torch.tensor([weight_per_class[b] for b in buckets], dtype=torch.double)

    logger.info(f"Oversampling weights per class: "
                f"{dict(zip(BUCKET_NAMES, [round(float(w), 4) for w in weight_per_class.tolist()]))}")

    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, criterion=None, num_classes: int = NUM_CLASSES):
    """Evaluate model on a dataset.

    Returns a dict with: accuracy, macro_recall, macro_f1, per_class_recall,
    per_class_precision, per_class_f1, preds, targets, avg_loss. Used for
    train/val/test alike so the SAME metrics are directly comparable across
    splits (needed for the train-vs-val diagnostic below).
    """
    model.eval()
    all_preds, all_targets = [], []
    all_logits = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            targets = batch.bucket

            if criterion is not None:
                loss = criterion(logits, targets)
                total_loss += loss.item() * batch.num_graphs

            all_logits.append(logits.cpu())
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset) if criterion is not None else 0.0
    labels = list(range(num_classes))

    accuracy = accuracy_score(all_targets, all_preds)
    macro_recall = recall_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0)
    per_class_recall = recall_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)
    per_class_precision = precision_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "preds": all_preds,
        "targets": all_targets,
        "avg_loss": avg_loss,
        "logits": torch.cat(all_logits, dim=0).numpy() if all_logits else None,
    }


def _fmt_per_class(values, names=BUCKET_NAMES):
    return ", ".join(f"{n}={v:.3f}" for n, v in zip(names, values))


def _diagnose_weak_classes(train_metrics, val_metrics, weak_threshold: float = 0.15,
                            bucket_names: list[str] = None):
    """Log an explicit train-vs-val diagnosis for any class with weak val recall.

    weak_threshold: a class is flagged "weak" if its val per-class recall is
    below this. For each weak class we log whether train recall is ALSO weak
    (-> likely feature/receptive-field insufficiency: the model can't even
    fit this class on data it has seen) or high (-> likely overfitting /
    too few examples to generalize, not a capacity problem).
    """
    names = bucket_names if bucket_names is not None else BUCKET_NAMES
    for idx, name in enumerate(names):
        val_r = val_metrics["per_class_recall"][idx]
        if val_r < weak_threshold:
            train_r = train_metrics["per_class_recall"][idx]
            if train_r < weak_threshold:
                verdict = "WEAK ON TRAIN TOO -> likely feature/receptive-field insufficiency, not overfitting"
            else:
                verdict = "high on train, low on val -> likely overfitting on too few rare-class examples"
            logger.info(
                f"  [diagnosis] class '{name}': train_recall={train_r:.3f}, "
                f"val_recall={val_r:.3f} -- {verdict}"
            )


def fit_class_bias(logits, targets, num_classes, grid=None, n_passes=3):
    """Fit an additive per-class logit bias that maximizes ACCURACY on the
    given (validation) split, by coordinate ascent.

    The training loader uses WeightedRandomSampler, so the network's softmax
    approximates a posterior under a ~rebalanced class prior, while val/test
    keep the true prior. argmax on that mismatched posterior is not the
    accuracy-maximizing rule. This fits the correction directly rather than
    assuming a closed-form prior ratio.

    Class 0's bias is pinned to 0.0 for identifiability (only differences
    between biases affect argmax). Returns a float32 numpy array [num_classes];
    all zeros if no bias beats the unbiased baseline.
    """
    import numpy as _np
    if grid is None:
        grid = _np.linspace(-2.0, 2.0, 41)          # step 0.1

    y = _np.asarray(targets)
    bias = _np.zeros(num_classes, dtype=_np.float64)

    def acc(b):
        return float(((logits + b).argmax(axis=1) == y).mean())

    base = acc(bias)
    best = base
    for _ in range(n_passes):
        improved = False
        for c in range(1, num_classes):             # class 0 pinned at 0.0
            keep, cur = bias[c], best
            for g in grid:
                bias[c] = g
                s = acc(bias)
                if s > cur:
                    cur, keep = s, g
            bias[c] = keep
            if cur > best:
                best, improved = cur, True
        if not improved:
            break

    if best <= base:
        logger.info("[bias] no per-class bias improved val accuracy; using zeros")
        return _np.zeros(num_classes, dtype=_np.float32)

    logger.info(f"[bias] val accuracy {base:.4f} -> {best:.4f}")
    logger.info(f"[bias] fitted per-class bias: "
                f"{[round(float(b), 2) for b in bias]}")
    return bias.astype(_np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path: str):
    """Train the radius predictor end to end and write model/reports/train_report.json."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_config = config["train"]
    data_config = config["data"]
    seed = config.get("seed", 0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data ---
    # Ad-hoc experiment toggle: drop hard classes ("1", "full") and see
    # achievable score on the remaining, better-supported buckets. Set
    # train.excluded_buckets: [] (or omit the key) to disable and train on
    # all 6 classes as usual.
    excluded_buckets = train_config.get("excluded_buckets", [])
    if excluded_buckets:
        logger.warning(
            f"Running with excluded_buckets={excluded_buckets} -- this is an "
            f"ad-hoc scoped experiment, NOT the full 6-class task. Results "
            f"here are not directly comparable to full-task runs."
        )

    # Env override so a supervised run can pin a specific dataset file even if
    # another process rebuilds/replaces the canonical labels.pt mid-flight.
    labels_path = os.environ.get("RADIUS_GNN_LABELS_PATH", data_config["labels_path"])
    raw_data_list = torch.load(labels_path, weights_only=False)
    logger.info(f"Loaded {len(raw_data_list)} labeled examples from {labels_path}")
    if not raw_data_list:
        raise RuntimeError(f"No examples in {data_config['labels_path']}")
    sample0 = raw_data_list[0]
    if not hasattr(sample0, "global_features") or sample0.global_features is None:
        raise RuntimeError(
            f"{data_config['labels_path']} has no global_features on samples. "
            "Rebuild labels with the current labels/build_labels.py before training."
        )
    logger.info(
        f"global_features present: shape={tuple(sample0.global_features.shape)}"
    )
    filtered_data_list, active_bucket_names = filter_and_remap_dataset(raw_data_list, excluded_buckets)
    active_num_classes = len(active_bucket_names)

    train_dataset, val_dataset, test_dataset = load_and_split_data_from_list(
        filtered_data_list,
        data_config["split"],
        data_config["split_by"],
        seed,
    )

    batch_size = train_config["batch_size"]

    # Oversample minority buckets ("1", "full") within the TRAINING loader
    # only -- val/test must keep the true class distribution so recall/F1
    # numbers stay honest. shuffle= and sampler= are mutually exclusive on
    # DataLoader; the sampler already randomizes order every epoch.
    pin = device.type == "cuda"
    use_oversampling = train_config.get("oversample_minority_classes", True)
    if use_oversampling:
        train_sampler = make_weighted_sampler(train_dataset, num_classes=active_num_classes)
        train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=pin)
    else:
        train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin)

    # Separate, non-shuffled, NON-oversampled loader over the TRAIN set
    # purely for the train-vs-val diagnostic below -- this must reflect the
    # TRUE training-set class distribution (not the oversampled one) so
    # train_metrics is comparable to val_metrics/test_metrics on equal terms.
    train_eval_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)

    # --- Model ---
    sample = train_dataset[0]
    in_dim = sample.x.shape[1]
    gf_dim = int(sample.global_features.shape[-1])
    logger.info(f"Input feature dimension: {in_dim}")
    logger.info(f"global_features dimension: {gf_dim}")

    model = create_model_from_config(
        config, in_dim, num_classes=active_num_classes, num_global_features=gf_dim
    ).to(device)

    # --- Loss / class weighting ---
    # Default is sqrt_inv_freq (gentler). "inverse_frequency" in the config
    # maps to the raw mode and will log a warning if the resulting ratio is
    # extreme (see compute_class_weights).
    weighting_key = train_config.get("class_weighting", "sqrt_inv_freq")
    weighting_map = {
        "inverse_frequency": "inv_freq",
        "inv_freq": "inv_freq",
        "sqrt_inv_freq": "sqrt_inv_freq",
        "none": None,
    }
    if weighting_key not in weighting_map:
        raise ValueError(f"Unknown train.class_weighting value: {weighting_key!r}")
    mode = weighting_map[weighting_key]

    if mode is not None:
        class_weights = compute_class_weights(
            [train_dataset[i] for i in range(len(train_dataset))], num_classes=active_num_classes, mode=mode
        ).to(device)
    else:
        logger.warning("class_weighting='none' -- rare classes will likely be ignored by the loss.")
        class_weights = None

    loss_name = train_config.get("loss", "ce")
    ordinal_lambda = float(train_config.get("ordinal_lambda", 0.5))
    if loss_name == "ordinal":
        criterion = OrdinalExpectedCostLoss(
            active_num_classes, mae_lambda=ordinal_lambda, weight=class_weights
        ).to(device)
        logger.info(f"Loss: ordinal CE+MAE (lambda={ordinal_lambda})")
    elif loss_name == "asymmetric_ordinal":
        criterion = AsymmetricOrdinalLoss(
            active_num_classes, weight=class_weights
        ).to(device)
        logger.info("Loss: asymmetric_ordinal (under-pred 2x + |diff|, plus CE)")
    elif loss_name in ("ce", "cross_entropy", None):
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Loss: cross_entropy")
    else:
        raise ValueError(f"Unknown train.loss value: {loss_name!r}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

    # --- Training loop ---
    # Selection/early-stopping is on validation MACRO-F1 (not accuracy, not
    # macro-recall alone) so precision collapse (a class predicted constantly
    # to farm recall) is penalized, matching what happened with macro-recall
    # selection in the previous run.
    selection_metric = train_config.get("selection_metric", "val_macro_f1")
    min_macro_f1 = float(train_config.get("min_macro_f1_for_selection", 0.40))
    min_class1_recall = float(train_config.get("min_val_class1_recall", 0.30))
    best_val_score = -1.0
    best_val_metrics_snapshot = None  # metrics dict at the epoch the selection score peaked, for reporting
    # Joint-selection fallback: always track the best macro-F1 epoch in
    # parallel, so that if NO epoch satisfies the joint floors we fall back
    # to macro-F1 selection instead of raising (recorded in the report as
    # selection_fallback_used).
    best_fallback_macro_f1 = -1.0
    fallback_snapshot = None
    logger.info(f"Selection metric: {selection_metric} "
                f"(guard: macro_f1 >= {min_macro_f1}"
                + (f", class1_recall >= {min_class1_recall}" if selection_metric == "joint" else "")
                + ")")
    patience = train_config["patience"]
    patience_counter = 0
    max_epochs = train_config["max_epochs"]

    # Path overrides exist so a second supervised run can execute concurrently
    # without two writers racing on the canonical checkpoint/report paths
    # (a silently-swapped mid-run checkpoint is the exact failure mode the
    # checkpoint-hygiene rule exists to prevent). Defaults are unchanged.
    checkpoint_dir = os.environ.get("RADIUS_GNN_CHECKPOINT_DIR", "model/checkpoints")
    report_dir = os.environ.get("RADIUS_GNN_REPORT_DIR", "model/reports")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, os.environ.get("RADIUS_GNN_CHECKPOINT_NAME", "radius_gnn.pt"))
    fallback_checkpoint_path = checkpoint_path + ".macrof1_fallback"

    test_graph_ids = sorted({
        str(getattr(test_dataset[i], "graph_id", ""))
        for i in range(len(test_dataset))
    })
    test_ids_path = os.path.join(report_dir, "test_graph_ids.json")
    with open(test_ids_path, "w") as f:
        json.dump({"seed": seed, "test_graph_ids": test_graph_ids}, f, indent=2)
    logger.info(f"Saved {len(test_graph_ids)} held-out test graph_ids to {test_ids_path}")

    # How often to run the (more expensive) train-set diagnostic pass.
    # Every epoch is fine for small datasets; increase this if train_eval_loader
    # becomes a bottleneck on larger data.
    diagnostic_every_n_epochs = 5

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.bucket)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_loader.dataset)

        val_metrics = evaluate(model, val_loader, device, criterion, num_classes=active_num_classes)

        logger.info(
            f"Epoch {epoch + 1}/{max_epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_metrics['avg_loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_macro_recall={val_metrics['macro_recall']:.4f}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        logger.info(f"  val per-class recall: {_fmt_per_class(val_metrics['per_class_recall'], names=active_bucket_names)}")
        logger.info(f"  val per-class f1:     {_fmt_per_class(val_metrics['per_class_f1'], names=active_bucket_names)}")

        if (epoch + 1) % diagnostic_every_n_epochs == 0:
            train_metrics = evaluate(model, train_eval_loader, device, criterion=None, num_classes=active_num_classes)
            logger.info(
                f"  [diag] train_acc={train_metrics['accuracy']:.4f}, "
                f"train_macro_f1={train_metrics['macro_f1']:.4f}"
            )
            logger.info(f"  [diag] train per-class recall: {_fmt_per_class(train_metrics['per_class_recall'], names=active_bucket_names)}")
            _diagnose_weak_classes(train_metrics, val_metrics, bucket_names=active_bucket_names)

        if selection_metric == "val_accuracy_guarded":
            # Accuracy is the reported headline metric AND the Section 6 gate
            # metric, but raw accuracy alone is maximized by the degenerate
            # all-class-0 predictor (val majority ~0.53, see lab log 13.4).
            # The macro-F1 floor makes that solution inadmissible.
            score = (val_metrics["accuracy"]
                     if val_metrics["macro_f1"] >= min_macro_f1 else -1.0)
        elif selection_metric == "joint":
            # Maximize val accuracy among epochs satisfying BOTH floors:
            # class-1 recall >= min_class1_recall AND macro_f1 >= min_macro_f1.
            # Motivation (lab log 16.5): on r1-augmented data, macro-F1
            # selection over-fires class 1 (acc 0.544) while accuracy selection
            # abandons it (recall 0.004); the joint rule picks the most
            # accurate epoch that still keeps class 1 alive.
            class1_recall = (float(val_metrics["per_class_recall"][1])
                             if active_num_classes > 1 else 0.0)
            admissible = (val_metrics["macro_f1"] >= min_macro_f1
                          and class1_recall >= min_class1_recall)
            score = val_metrics["accuracy"] if admissible else -1.0
            if val_metrics["macro_f1"] > best_fallback_macro_f1:
                best_fallback_macro_f1 = val_metrics["macro_f1"]
                fallback_snapshot = val_metrics
                torch.save(model.state_dict(), fallback_checkpoint_path)
        else:
            score = val_metrics["macro_f1"]

        if score > best_val_score:
            best_val_score = score
            best_val_metrics_snapshot = val_metrics
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"  -> new best ({selection_metric}={best_val_score:.4f}, "
                        f"val_acc={val_metrics['accuracy']:.4f}, "
                        f"val_macro_f1={val_metrics['macro_f1']:.4f}); checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    selection_fallback_used = None
    if best_val_metrics_snapshot is None:
        if selection_metric == "joint" and fallback_snapshot is not None:
            selection_fallback_used = "val_macro_f1"
            shutil.copyfile(fallback_checkpoint_path, checkpoint_path)
            best_val_metrics_snapshot = fallback_snapshot
            logger.warning(
                f"Joint selection infeasible: no epoch reached class-1 recall "
                f">= {min_class1_recall} with macro_f1 >= {min_macro_f1}. "
                f"Falling back to best val_macro_f1 epoch "
                f"(macro_f1={best_fallback_macro_f1:.4f})."
            )
        else:
            raise RuntimeError(
                f"No checkpoint was saved: no epoch reached val_macro_f1 >= "
                f"{min_macro_f1} under selection_metric={selection_metric!r}. "
                f"Either the model collapsed to a single class, or the guard is "
                f"set above what this configuration can reach."
            )

    # --- Final test evaluation, using the best checkpoint ---
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, num_classes=active_num_classes)

    import numpy as _np

    test_metrics_raw_acc = float(test_metrics["accuracy"])
    class_bias = _np.zeros(active_num_classes, dtype=_np.float32)

    if train_config.get("tune_class_bias", False):
        val_metrics_final = evaluate(model, val_loader, device,
                                     num_classes=active_num_classes)
        class_bias = fit_class_bias(val_metrics_final["logits"],
                                    val_metrics_final["targets"],
                                    active_num_classes)
        if _np.any(class_bias != 0):
            biased_preds = (test_metrics["logits"] + class_bias).argmax(axis=1).tolist()
            test_metrics["preds"] = biased_preds
            # recompute every downstream metric from the biased predictions
            t = test_metrics["targets"]
            labels_r = list(range(active_num_classes))
            test_metrics["accuracy"] = accuracy_score(t, biased_preds)
            test_metrics["macro_recall"] = recall_score(t, biased_preds, labels=labels_r,
                                                        average="macro", zero_division=0)
            test_metrics["macro_f1"] = f1_score(t, biased_preds, labels=labels_r,
                                                average="macro", zero_division=0)
            test_metrics["per_class_recall"] = recall_score(t, biased_preds, labels=labels_r,
                                                            average=None, zero_division=0)
            test_metrics["per_class_precision"] = precision_score(t, biased_preds, labels=labels_r,
                                                                  average=None, zero_division=0)
            test_metrics["per_class_f1"] = f1_score(t, biased_preds, labels=labels_r,
                                                    average=None, zero_division=0)
            logger.info(f"Test accuracy: {test_metrics_raw_acc:.4f} (no bias) -> "
                        f"{test_metrics['accuracy']:.4f} (with val-fitted bias)")

    test_preds, test_targets = test_metrics["preds"], test_metrics["targets"]

    cm = confusion_matrix(test_targets, test_preds, labels=list(range(active_num_classes)))

    under = sum(1 for p, t in zip(test_preds, test_targets) if p < t)
    over = sum(1 for p, t in zip(test_preds, test_targets) if p > t)
    total = len(test_preds)
    under_rate = under / total
    over_rate = over / total

    raw_class_counts = Counter(test_targets)
    majority_count = max(raw_class_counts.values())
    majority_baseline = majority_count / total
    # Report class counts keyed by bucket NAME, not raw index, per Appendix A.5 schema.
    class_counts_by_name = {
        active_bucket_names[idx]: int(raw_class_counts.get(idx, 0)) for idx in range(active_num_classes)
    }

    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test macro recall: {test_metrics['macro_recall']:.4f}")
    logger.info(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"Majority class baseline: {majority_baseline:.4f}")
    logger.info(f"Under-prediction rate: {under_rate:.4f}")
    logger.info(f"Over-prediction rate: {over_rate:.4f}")
    logger.info(f"Test per-class precision: {_fmt_per_class(test_metrics['per_class_precision'], names=active_bucket_names)}")
    logger.info(f"Test per-class recall:    {_fmt_per_class(test_metrics['per_class_recall'], names=active_bucket_names)}")
    logger.info(f"Test per-class f1:        {_fmt_per_class(test_metrics['per_class_f1'], names=active_bucket_names)}")
    logger.info(f"Confusion matrix (rows=true, cols=pred, order={active_bucket_names}):\n{cm}")

    if test_metrics["accuracy"] < majority_baseline:
        logger.warning(
            f"Test accuracy ({test_metrics['accuracy']:.4f}) is BELOW the majority-class "
            f"baseline ({majority_baseline:.4f}). The model is not yet usable as-is -- "
            f"do not proceed to Section 7/8 with this checkpoint."
        )

    # Secondary metric only. Section 15 (verdict B) established that r*=0 vs
    # r*=1 is not separable from available information (scalar ceiling ~0.21
    # recall on class 1; slack_v_is_zero is a dead column). This reports what
    # accuracy would be if buckets "0" and "1" were one class. It is NOT the
    # headline 6-class number and must never be quoted as such.
    merged_correct = sum(
        1 for p, t in zip(test_preds, test_targets)
        if p == t or (p in (0, 1) and t in (0, 1))
    )
    test_accuracy_merged_01 = merged_correct / total
    logger.info(f"[secondary] test accuracy with buckets 0,1 merged: "
                f"{test_accuracy_merged_01:.4f}")

    report = {
        "val_accuracy": float(best_val_metrics_snapshot["accuracy"]),
        "val_macro_recall": float(best_val_metrics_snapshot["macro_recall"]),
        "val_macro_f1": float(best_val_metrics_snapshot["macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_recall": float(test_metrics["macro_recall"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "selection_metric": selection_metric,
        "min_macro_f1_for_selection": min_macro_f1,
        "min_val_class1_recall": min_class1_recall if selection_metric == "joint" else None,
        "selection_fallback_used": selection_fallback_used,
        "val_class1_recall": (float(best_val_metrics_snapshot["per_class_recall"][1])
                              if active_num_classes > 1 else None),
        "class_bias": [float(b) for b in class_bias],
        "test_accuracy_no_bias": test_metrics_raw_acc,
        "test_accuracy_merged_01_secondary": float(test_accuracy_merged_01),
        "majority_class_baseline_accuracy": float(majority_baseline),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": active_bucket_names,
        "excluded_buckets": excluded_buckets,
        "under_prediction_rate": float(under_rate),
        "over_prediction_rate": float(over_rate),
        "test_per_class_precision": {
            name: float(r) for name, r in zip(active_bucket_names, test_metrics["per_class_precision"])
        },
        "test_per_class_recall": {
            name: float(r) for name, r in zip(active_bucket_names, test_metrics["per_class_recall"])
        },
        "test_per_class_f1": {
            name: float(r) for name, r in zip(active_bucket_names, test_metrics["per_class_f1"])
        },
        "class_counts": class_counts_by_name,
        "selection_criterion": selection_metric,
        "class_weighting_mode": mode,
        "oversample_minority_classes": use_oversampling,
        "loss": loss_name,
        "ordinal_lambda": ordinal_lambda if loss_name == "ordinal" else None,
    }

    report_path = os.path.join(report_dir, "train_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Training complete. Report saved to {report_path}")
    return report


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "model/config.yaml"
    train(config_path)