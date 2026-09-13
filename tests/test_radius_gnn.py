import pytest
import torch
from torch_geometric.data import Data, Batch
from labels.build_labels import NUM_GLOBAL_FEATURES
from model.radius_gnn import RadiusGNN


def _with_global_features(data: Data, batch_size: int = 1) -> Data:
    data.global_features = torch.zeros(batch_size, NUM_GLOBAL_FEATURES)
    return data


def test_radius_gnn_forward():
    """Test forward pass."""
    model = RadiusGNN(in_dim=6, hidden_dim=16, num_layers=2, num_classes=6, num_global_features=NUM_GLOBAL_FEATURES)

    # Create a simple batch
    x = torch.randn(10, 6)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)

    data = _with_global_features(Data(x=x, edge_index=edge_index, batch=batch))
    logits = model(data)
    assert logits.shape == (1, 6)


def test_radius_gnn_batch():
    """Test with multiple graphs in batch."""
    model = RadiusGNN(in_dim=6, hidden_dim=16, num_layers=2, num_classes=6, num_global_features=NUM_GLOBAL_FEATURES)

    # Two graphs
    x1 = torch.randn(5, 6)
    edge_index1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch1 = torch.zeros(5, dtype=torch.long)

    x2 = torch.randn(4, 6)
    edge_index2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch2 = torch.ones(4, dtype=torch.long)

    data1 = _with_global_features(Data(x=x1, edge_index=edge_index1, batch=batch1))
    data2 = _with_global_features(Data(x=x2, edge_index=edge_index2, batch=batch2))

    batch = Batch.from_data_list([data1, data2])
    logits = model(batch)
    assert logits.shape == (2, 6)


def test_radius_gnn_gcn():
    """Test with GCN conv type."""
    model = RadiusGNN(in_dim=6, hidden_dim=16, num_layers=2, num_classes=6, conv_type="gcn", num_global_features=NUM_GLOBAL_FEATURES)

    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = torch.zeros(5, dtype=torch.long)

    data = _with_global_features(Data(x=x, edge_index=edge_index, batch=batch))
    logits = model(data)
    assert logits.shape == (1, 6)


def test_color_embedding():
    """Test that color embedding works."""
    model = RadiusGNN(in_dim=6, hidden_dim=16, num_layers=2, num_classes=6, num_global_features=NUM_GLOBAL_FEATURES)

    # x[:, 0] should be color index
    x = torch.zeros(5, 6)
    x[:, 0] = torch.tensor([0, 1, 2, 3, 4])  # Color indices
    x[:, 1:] = torch.randn(5, 5)  # Other features

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = torch.zeros(5, dtype=torch.long)

    data = _with_global_features(Data(x=x, edge_index=edge_index, batch=batch))
    logits = model(data)
    assert logits.shape == (1, 6)


def test_hop_shell_pooling_uses_num_layers():
    """hop_shell MLP width is (num_layers+1)*hidden_dim; hop-4 stays 5 shells."""
    x = torch.randn(8, 7)
    x[:, 6] = torch.tensor([0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25, 0])  # dist/4
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.zeros(8, dtype=torch.long)
    data = _with_global_features(Data(x=x, edge_index=edge_index, batch=batch))

    m4 = RadiusGNN(in_dim=7, hidden_dim=16, num_layers=4, num_classes=6,
                   pooling="hop_shell", num_global_features=NUM_GLOBAL_FEATURES)
    assert m4.n_shells == 5
    assert m4(data).shape == (1, 6)

    m5 = RadiusGNN(in_dim=7, hidden_dim=16, num_layers=5, num_classes=6,
                   pooling="hop_shell", num_global_features=NUM_GLOBAL_FEATURES)
    x5 = x.clone()
    x5[:, 6] = torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.2, 0])  # dist/5
    data5 = _with_global_features(Data(x=x5, edge_index=edge_index, batch=batch))
    assert m5.n_shells == 6
    assert m5(data5).shape == (1, 6)


def test_missing_global_features_fails_loudly():
    """Old labels without global_features must not silently proceed."""
    model = RadiusGNN(in_dim=6, hidden_dim=16, num_layers=2, num_classes=6, num_global_features=NUM_GLOBAL_FEATURES)
    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = torch.zeros(5, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    with pytest.raises(RuntimeError, match="global_features"):
        model(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])