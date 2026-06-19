"""
Unit tests for adversarial attacks (FGSM and PGD).

Tests cover: attack execution, feature perturbation, preservation of structure,
empty malicious mask handling, and feature clipping.
"""

import unittest
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv


class SimpleGCN(nn.Module):
    """Simple GCN for testing."""

    def __init__(self, num_features: int, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class SimpleGAT(nn.Module):
    """Simple GAT for testing."""

    def __init__(self, num_features: int, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class TestAttacks(unittest.TestCase):
    """Test suite for FGSM and PGD attacks."""

    def setUp(self):
        """Set up test fixtures: models and toy graphs."""
        self.num_nodes = 20
        self.num_features = 8
        self.num_classes = 2

        # Create a toy graph
        torch.manual_seed(42)
        x = torch.randn(self.num_nodes, self.num_features, dtype=torch.float32)
        
        # Create edges with some structure
        num_edges = self.num_nodes * 3
        edge_index = torch.randint(0, self.num_nodes, (2, num_edges), dtype=torch.long)
        
        # Create mixed labels: benign (0) and attack (1)
        y = torch.randint(0, 2, (self.num_nodes,), dtype=torch.long)
        
        self.data = Data(x=x, edge_index=edge_index, y=y)
        
        # Create models
        self.gcn_model = SimpleGCN(self.num_features)
        self.gat_model = SimpleGAT(self.num_features)

    def test_fgsm_returns_data_object(self):
        """Test that FGSM returns a valid PyG Data object."""
        data_adv = fgsm_attack(self.gcn_model, self.data, epsilon=0.05)
        self.assertIsInstance(data_adv, Data)

    def test_fgsm_perturbs_features(self):
        """Test that FGSM actually perturbs node features."""
        data_adv = fgsm_attack(self.gcn_model, self.data, epsilon=0.05)
        
        # Check that at least some features changed
        feature_diff = (data_adv.x - self.data.x).abs().sum().item()
        self.assertGreater(feature_diff, 0.0, "Features should be perturbed")

    def test_fgsm_preserves_edge_index(self):
        """Test that FGSM preserves edge_index."""
        data_adv = fgsm_attack(self.gcn_model, self.data, epsilon=0.05)
        torch.testing.assert_close(data_adv.edge_index, self.data.edge_index)

    def test_fgsm_preserves_labels(self):
        """Test that FGSM preserves node labels."""
        data_adv = fgsm_attack(self.gcn_model, self.data, epsilon=0.05)
        torch.testing.assert_close(data_adv.y, self.data.y)

    def test_fgsm_malicious_only_mode(self):
        """Test FGSM attack only on malicious nodes."""
        data_adv = fgsm_attack(
            self.gcn_model, self.data, epsilon=0.1, attack_only_malicious=True
        )
        
        # Benign nodes (y==0) should have minimal perturbation
        malicious_mask = self.data.y == 1
        benign_mask = self.data.y == 0
        
        if benign_mask.sum() > 0:
            benign_diff = (data_adv.x[benign_mask] - self.data.x[benign_mask]).abs().mean()
            malicious_diff = (data_adv.x[malicious_mask] - self.data.x[malicious_mask]).abs().mean()
            
            # Malicious nodes should have more perturbation than benign nodes
            self.assertGreater(malicious_diff, benign_diff)

    def test_fgsm_all_nodes_mode(self):
        """Test FGSM attack on all nodes."""
        data_adv = fgsm_attack(
            self.gcn_model, self.data, epsilon=0.1, attack_only_malicious=False
        )
        
        # All nodes should have some perturbation
        diff = (data_adv.x - self.data.x).abs()
        nodes_with_perturbation = (diff.sum(dim=1) > 0).sum().item()
        
        self.assertGreater(nodes_with_perturbation, 0)

    def test_fgsm_no_malicious_nodes_handled_gracefully(self):
        """Test FGSM with no malicious nodes (all benign)."""
        data_benign = self.data.clone()
        data_benign.y = torch.zeros(self.num_nodes, dtype=torch.long)
        
        # Should not raise an error
        data_adv = fgsm_attack(
            self.gcn_model, data_benign, epsilon=0.05, attack_only_malicious=True
        )
        
        # Features should remain unchanged
        torch.testing.assert_close(data_adv.x, data_benign.x)

    def test_fgsm_feature_clipping(self):
        """Test FGSM with feature clipping."""
        clip_min, clip_max = -1.0, 1.0
        data_adv = fgsm_attack(
            self.gcn_model,
            self.data,
            epsilon=0.5,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        
        # All features should be within bounds
        self.assertTrue((data_adv.x >= clip_min).all())
        self.assertTrue((data_adv.x <= clip_max).all())

    def test_pgd_returns_data_object(self):
        """Test that PGD returns a valid PyG Data object."""
        data_adv = pgd_attack(self.gcn_model, self.data, epsilon=0.05, steps=5)
        self.assertIsInstance(data_adv, Data)

    def test_pgd_perturbs_features(self):
        """Test that PGD actually perturbs node features."""
        data_adv = pgd_attack(self.gcn_model, self.data, epsilon=0.05, steps=5)
        
        # Check that at least some features changed
        feature_diff = (data_adv.x - self.data.x).abs().sum().item()
        self.assertGreater(feature_diff, 0.0, "Features should be perturbed")

    def test_pgd_preserves_edge_index(self):
        """Test that PGD preserves edge_index."""
        data_adv = pgd_attack(self.gcn_model, self.data, epsilon=0.05, steps=5)
        torch.testing.assert_close(data_adv.edge_index, self.data.edge_index)

    def test_pgd_preserves_labels(self):
        """Test that PGD preserves node labels."""
        data_adv = pgd_attack(self.gcn_model, self.data, epsilon=0.05, steps=5)
        torch.testing.assert_close(data_adv.y, self.data.y)

    def test_pgd_perturbation_bounded_by_epsilon(self):
        """Test that PGD perturbation is bounded by epsilon."""
        epsilon = 0.1
        data_adv = pgd_attack(
            self.gcn_model, self.data, epsilon=epsilon, alpha=0.01, steps=5
        )
        
        # Compute L∞ norm (max absolute difference across features)
        perturbation = data_adv.x - self.data.x
        linf_perturbation = perturbation.abs().max(dim=1).values
        
        # All perturbations should be <= epsilon (with small numerical tolerance)
        self.assertTrue((linf_perturbation <= epsilon + 1e-5).all())

    def test_pgd_malicious_only_mode(self):
        """Test PGD attack only on malicious nodes."""
        data_adv = pgd_attack(
            self.gcn_model,
            self.data,
            epsilon=0.1,
            alpha=0.01,
            steps=5,
            attack_only_malicious=True,
        )
        
        # Benign nodes should have minimal perturbation
        malicious_mask = self.data.y == 1
        benign_mask = self.data.y == 0
        
        if benign_mask.sum() > 0:
            benign_diff = (data_adv.x[benign_mask] - self.data.x[benign_mask]).abs().mean()
            malicious_diff = (data_adv.x[malicious_mask] - self.data.x[malicious_mask]).abs().mean()
            
            # Malicious nodes should have more perturbation than benign nodes
            self.assertGreater(malicious_diff, benign_diff)

    def test_pgd_all_nodes_mode(self):
        """Test PGD attack on all nodes."""
        data_adv = pgd_attack(
            self.gcn_model,
            self.data,
            epsilon=0.1,
            alpha=0.01,
            steps=5,
            attack_only_malicious=False,
        )
        
        # All nodes should have some perturbation
        diff = (data_adv.x - self.data.x).abs()
        nodes_with_perturbation = (diff.sum(dim=1) > 0).sum().item()
        
        self.assertGreater(nodes_with_perturbation, 0)

    def test_pgd_no_malicious_nodes_handled_gracefully(self):
        """Test PGD with no malicious nodes (all benign)."""
        data_benign = self.data.clone()
        data_benign.y = torch.zeros(self.num_nodes, dtype=torch.long)
        
        # Should not raise an error
        data_adv = pgd_attack(
            self.gcn_model,
            data_benign,
            epsilon=0.05,
            alpha=0.01,
            steps=5,
            attack_only_malicious=True,
        )
        
        # Features should remain unchanged
        torch.testing.assert_close(data_adv.x, data_benign.x)

    def test_pgd_feature_clipping(self):
        """Test PGD with feature clipping."""
        clip_min, clip_max = -1.0, 1.0
        data_adv = pgd_attack(
            self.gcn_model,
            self.data,
            epsilon=0.5,
            alpha=0.01,
            steps=5,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        
        # All features should be within bounds
        self.assertTrue((data_adv.x >= clip_min).all())
        self.assertTrue((data_adv.x <= clip_max).all())

    def test_pgd_invalid_epsilon_raises_error(self):
        """Test that PGD raises ValueError for invalid epsilon."""
        with self.assertRaises(ValueError):
            pgd_attack(self.gcn_model, self.data, epsilon=-0.05)

    def test_pgd_invalid_alpha_raises_error(self):
        """Test that PGD raises ValueError for invalid alpha."""
        with self.assertRaises(ValueError):
            pgd_attack(self.gcn_model, self.data, alpha=-0.01)

    def test_pgd_invalid_steps_raises_error(self):
        """Test that PGD raises ValueError for invalid steps."""
        with self.assertRaises(ValueError):
            pgd_attack(self.gcn_model, self.data, steps=-5)

    def test_attacks_with_gat_model(self):
        """Test attacks work with GAT model as well."""
        data_adv_fgsm = fgsm_attack(self.gat_model, self.data, epsilon=0.05)
        data_adv_pgd = pgd_attack(
            self.gat_model, self.data, epsilon=0.05, steps=5
        )
        
        # Both should return valid Data objects with perturbed features
        self.assertIsInstance(data_adv_fgsm, Data)
        self.assertIsInstance(data_adv_pgd, Data)
        self.assertGreater((data_adv_fgsm.x - self.data.x).abs().sum().item(), 0.0)
        self.assertGreater((data_adv_pgd.x - self.data.x).abs().sum().item(), 0.0)

    def test_fgsm_vs_pgd_perturbation_difference(self):
        """Test that FGSM and PGD produce different perturbations."""
        data_adv_fgsm = fgsm_attack(self.gcn_model, self.data, epsilon=0.1)
        data_adv_pgd = pgd_attack(
            self.gcn_model, self.data, epsilon=0.1, alpha=0.01, steps=5
        )
        
        # They should produce different perturbations due to different strategies
        diff = (data_adv_fgsm.x - data_adv_pgd.x).abs().sum().item()
        self.assertGreater(diff, 0.0)


if __name__ == "__main__":
    unittest.main()
