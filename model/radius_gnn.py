"""
RadiusGNN: predicts the repair-radius bucket for an edit (methodology
Section 6, Appendix A.5).

CHANGES FROM THE PREVIOUS VERSION:
1. input_proj is now self-deriving from in_dim instead of a hardcoded
   "+6" constant. That hardcoded value already drifted out of sync once
   (extract_features grew a 7th column -- hop_distance_from_center -- and
   this file still assumed 6 "other features", causing a shape-mismatch
   crash: mat1/mat2 could not be multiplied). Deriving it from in_dim makes
   that entire class of bug structurally impossible going forward.
2. Added global_features support: a small graph-level feature vector
   (4 structural + 6 post-edit color-slack; see labels/build_labels.py)
   is concatenated
   onto the pooled node embedding, AFTER pooling, before the MLP head.
   This targets the 'full' bucket specifically: cases where repair damage
   spreads further than any local hop window can see are hypothesized to
   correlate with whole-graph structural criticality of the edited vertex,
   which mean-pooling over a bounded local window cannot capture no matter
   how wide the window is made.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATv2Conv, global_mean_pool


class RadiusGNN(nn.Module):
    """GNN for predicting repair radius bucket.

    N GraphSAGE/GCN layers -> global mean pool -> concat global features
    -> 2-layer MLP -> num_classes softmax
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int = 6,
        dropout: float = 0.2,
        conv_type: str = "sage",
        num_colors: int = 1024,
        num_global_features: int = 10,
        gat_heads: int = 4,
        pooling: str = "mean",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.num_global_features = num_global_features
        self.gat_heads = gat_heads
        if pooling not in ("mean", "hop_shell"):
            raise ValueError(f"Unknown pooling: {pooling!r} (expected 'mean' or 'hop_shell')")
        self.pooling = pooling
        # hop_radius == num_layers by convention; shells are distances 0..hop_radius.
        self.n_shells = (num_layers + 1) if pooling == "hop_shell" else 1

        # Color embedding (color indices are categorical, not ordinal).
        # Palette max is k = max_degree + 5; max_degree <= n-1 <= 499, so 1024 is a safe upper bound.
        self.color_embedding = nn.Embedding(num_colors, hidden_dim)

        # Input projection combines embedded color (hidden_dim) + all other
        # per-node feature columns (in_dim - 1, since column 0 is the color
        # index consumed by the embedding above). Derived from in_dim rather
        # than hardcoded, so adding/removing a per-node feature column in
        # extract_features() never silently breaks this shape again.
        self.other_feature_dim = in_dim - 1
        self.input_proj = nn.Linear(hidden_dim + self.other_feature_dim, hidden_dim)

        # Graph convolution layers.
        # conv_type="gatv2": attention-based aggregation -- lets the model
        # learn WHICH neighbors matter more for a given prediction, instead
        # of SAGE's uniform mean/pool aggregation over all neighbors. Using
        # concat=False so each layer's output stays hidden_dim regardless of
        # gat_heads, keeping the rest of the architecture (input_proj sizing,
        # residual-free stacking) unchanged.
        self.convs = nn.ModuleList()
        if conv_type == "sage":
            conv_class = SAGEConv
            for i in range(num_layers):
                self.convs.append(conv_class(hidden_dim, hidden_dim))
        elif conv_type == "gcn":
            conv_class = GCNConv
            for i in range(num_layers):
                self.convs.append(conv_class(hidden_dim, hidden_dim))
        elif conv_type == "gatv2":
            # NOTE: no dropout= passed to GATv2Conv itself -- that would drop
            # attention coefficients, which is more aggressive than ordinary
            # feature dropout and was suspected of near-zeroing gradients in
            # early training (loss/accuracy identical across multiple epochs).
            # Standard between-layer feature dropout (see forward()) still
            # applies, matching how SAGE/GCN are regularized.
            for i in range(num_layers):
                self.convs.append(
                    GATv2Conv(hidden_dim, hidden_dim, heads=gat_heads, concat=False)
                )
        else:
            raise ValueError(f"Unknown conv_type: {conv_type!r} (expected 'sage', 'gcn', or 'gatv2')")

        # MLP head: pooled node embedding concatenated with the graph-level
        # global features (num_global_features). hop_shell pooling produces
        # n_shells x hidden_dim (one mean vector per hop shell 0..hop_radius).
        pooled_dim = self.n_shells * hidden_dim if pooling == "hop_shell" else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(pooled_dim + num_global_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass returning logits [batch, num_classes]."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # x[:, 0] = color index (or 0 for uncolored)
        # x[:, 1:] = all other per-node features (list_size, tightness,
        #            degree_norm, is_edited, edit_type, hop_distance_from_center, ...)
        color_idx = x[:, 0].long().clamp(0, self.color_embedding.num_embeddings - 1)
        other_features = x[:, 1:]

        if other_features.shape[1] != self.other_feature_dim:
            raise RuntimeError(
                f"RadiusGNN was built for {self.other_feature_dim} per-node "
                f"'other features' (in_dim={self.other_feature_dim + 1}), but "
                f"got {other_features.shape[1]}. The label dataset's feature "
                f"layout has changed since this model was constructed -- "
                f"rebuild the model (or the labels) so they agree."
            )

        # Embed color
        color_emb = self.color_embedding(color_idx)

        # Combine with other features
        x = torch.cat([color_emb, other_features], dim=1)
        x = self.input_proj(x)

        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == "hop_shell":
            # Pool per hop shell instead of once over the whole window. r* is a
            # BFS radius by definition; a single mean over a 4-hop window is
            # numerically dominated by the outer shells and destroys the radial
            # profile that determines the label. Mass-weighted: empty shells
            # contribute exact zeros (window terminated early -- informative).
            # x[:, 6] is hop_distance / hop_radius; hop_radius matches num_layers.
            hop_r = float(self.num_layers)
            n = self.n_shells
            shell = (data.x[:, 6] * hop_r).round().long().clamp(0, n - 1)
            pooled = [global_mean_pool(x * (shell == s).float().unsqueeze(1), batch)
                      for s in range(n)]
            x = torch.cat(pooled, dim=1)                                # [B, n_shells*hidden_dim]
        else:
            # Global mean pooling over each example's node window.
            x = global_mean_pool(x, batch)

        # Concatenate whole-graph structural features, computed independently
        # of the local window (see labels/build_labels.py:compute_global_features).
        # data.global_features has shape [batch_size, num_global_features]
        # after PyG's Batch.from_data_list concatenates the per-example
        # [1, num_global_features] tensors along dim 0.
        if not hasattr(data, "global_features") or data.global_features is None:
            raise RuntimeError(
                "Batch has no 'global_features' attribute. This model requires "
                "labels built with the global-features version of "
                "labels/build_labels.py -- rebuild data/labels/labels.pt."
            )
        global_feats = data.global_features
        if global_feats.shape[1] != self.num_global_features:
            raise RuntimeError(
                f"RadiusGNN expects {self.num_global_features} global features, "
                f"got {global_feats.shape[1]}. Check labels.build_labels.NUM_GLOBAL_FEATURES "
                f"matches this model's num_global_features."
            )

        x = torch.cat([x, global_feats], dim=1)

        # MLP head
        logits = self.mlp(x)
        return logits


def create_model_from_config(config: dict, in_dim: int, num_classes: int = None,
                             num_global_features: int = None) -> RadiusGNN:
    """Create model from config dict.

    num_global_features is imported from labels.build_labels rather than
    hardcoded here, so the model and the label pipeline can never silently
    disagree about how many global features exist -- the same class of bug
    that previously hit BUCKET_NAMES across multiple files. Pass explicitly
    when training on an augmented label file (e.g. 15-d shell density).

    num_classes: pass explicitly when training on a filtered/remapped class
    set (see model/train.py's excluded_buckets experiment) -- defaults to
    RadiusGNN's own default (6, the full bucket scheme) if omitted.
    """
    from labels.build_labels import NUM_GLOBAL_FEATURES

    model_config = config["model"]
    kwargs = dict(
        in_dim=in_dim,
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        conv_type=model_config["conv_type"],
        num_global_features=(
            NUM_GLOBAL_FEATURES if num_global_features is None else num_global_features
        ),
        gat_heads=model_config.get("gat_heads", 4),  # only used when conv_type == "gatv2"
        pooling=model_config.get("pooling", "mean"),
    )
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    return RadiusGNN(**kwargs)