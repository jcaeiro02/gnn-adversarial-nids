"""Unit tests for structural adversarial attacks and experiment wiring."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attacks.structural import edge_addition_attack, edge_removal_attack


class TestStructuralAttacks(unittest.TestCase):
    """Tests for structural edge perturbation attacks."""

    def setUp(self):
        torch.manual_seed(42)
        self.edge_index = torch.tensor(
            [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]],
            dtype=torch.long,
        )
        self.x = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
            dtype=torch.float32,
        )
        self.y = torch.tensor([1, 1, 0, 0], dtype=torch.long)
        self.edge_attr = torch.arange(1, self.edge_index.size(1) + 1, dtype=torch.float32).view(-1, 1)
        self.data = Data(x=self.x.clone(), edge_index=self.edge_index.clone(), y=self.y.clone(), edge_attr=self.edge_attr.clone())

    def test_edge_removal_reduces_edges(self):
        data_adv = edge_removal_attack(self.data, perturbation_rate=0.5, attack_only_malicious=True)
        self.assertLess(data_adv.edge_index.size(1), self.data.edge_index.size(1))

    def test_edge_removal_preserves_x_and_y(self):
        data_adv = edge_removal_attack(self.data, perturbation_rate=0.5, attack_only_malicious=True)
        torch.testing.assert_close(data_adv.x, self.data.x)
        torch.testing.assert_close(data_adv.y, self.data.y)

    def test_edge_removal_attack_only_malicious(self):
        edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float32)
        y = torch.tensor([1, 0, 0, 0], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)

        data_adv = edge_removal_attack(data, perturbation_rate=0.5, attack_only_malicious=True)

        # Only the edge incident to malicious node 0 should be removed.
        remaining_edges = set(tuple(pair.tolist()) for pair in data_adv.edge_index.t())
        self.assertIn((2, 3), remaining_edges)
        self.assertNotIn((0, 1), remaining_edges)
        self.assertNotIn((1, 0), remaining_edges)

    def test_edge_removal_updates_edge_attr(self):
        data_adv = edge_removal_attack(self.data, perturbation_rate=0.5, attack_only_malicious=True)
        self.assertEqual(data_adv.edge_attr.shape[0], data_adv.edge_index.size(1))
        self.assertLess(data_adv.edge_attr.shape[0], self.data.edge_attr.shape[0])

    def test_edge_addition_increases_edges(self):
        data_adv = edge_addition_attack(self.data, perturbation_rate=0.5, attack_only_malicious=True)
        self.assertGreater(data_adv.edge_index.size(1), self.data.edge_index.size(1))

    def test_edge_addition_preserves_x_and_y(self):
        data_adv = edge_addition_attack(self.data, perturbation_rate=0.5, attack_only_malicious=True)
        torch.testing.assert_close(data_adv.x, self.data.x)
        torch.testing.assert_close(data_adv.y, self.data.y)

    def test_edge_addition_avoids_self_loops(self):
        data_adv = edge_addition_attack(self.data, perturbation_rate=0.5, attack_only_malicious=True)
        self.assertFalse(torch.any(data_adv.edge_index[0] == data_adv.edge_index[1]))

    def test_edge_addition_avoids_duplicates(self):
        data_adv = edge_addition_attack(self.data, perturbation_rate=0.5, attack_only_malicious=True)
        edge_tuples = [tuple(edge.tolist()) for edge in data_adv.edge_index.t()]
        self.assertEqual(len(edge_tuples), len(set(edge_tuples)))

    def test_edge_addition_adds_reverse_edges(self):
        data_adv = edge_addition_attack(self.data, perturbation_rate=0.5, attack_only_malicious=True)
        original_edges = set(tuple(edge.tolist()) for edge in self.data.edge_index.t())
        attacked_edges = set(tuple(edge.tolist()) for edge in data_adv.edge_index.t())
        new_edges = attacked_edges - original_edges

        self.assertTrue(new_edges)
        for src, dst in new_edges:
            self.assertIn((dst, src), attacked_edges)

    def test_invalid_perturbation_rate_raises_error(self):
        with self.assertRaises(ValueError):
            edge_removal_attack(self.data, perturbation_rate=0.0)
        with self.assertRaises(ValueError):
            edge_removal_attack(self.data, perturbation_rate=1.5)
        with self.assertRaises(ValueError):
            edge_addition_attack(self.data, perturbation_rate=0.0)
        with self.assertRaises(ValueError):
            edge_addition_attack(self.data, perturbation_rate=1.5)


class TestStructuralAttackExperiment(unittest.TestCase):
    """A lightweight smoke test for the structural attack experiment script."""

    def test_experiment_smoke_test_writes_expected_outputs(self):
        spec = importlib.util.spec_from_file_location(
            "structural_experiment",
            Path(__file__).parent.parent / "experiments" / "03_structural_attacks.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        dataset = [
            Data(
                x=torch.randn(4, 3),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                y=torch.tensor([0, 1, 0, 0], dtype=torch.long),
            )
        ]

        class DummyTrainer:
            def __init__(self, *args, **kwargs):
                self.device = "cpu"

            def load_checkpoint(self, path):
                self.loaded_checkpoint = path

            def evaluate(self, graph_data):
                return {
                    "accuracy": 0.90,
                    "precision": 0.80,
                    "recall": 0.70,
                    "f1": 0.60,
                    "roc_auc": 0.50,
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "structural_outputs"
            with patch.object(module, "load_split_datasets", return_value=(None, None, dataset)), \
                 patch.object(module, "build_model", return_value=object()), \
                 patch.object(module, "Trainer", DummyTrainer), \
                 patch.object(module, "create_attack_run_directory", return_value=output_dir):
                with patch.object(sys, "argv", ["prog", "--checkpoint", "dummy.ckpt", "--dataset", "nsl-kdd"]):
                    args = module.parse_args()
                    results = module.run_structural_attacks(args)

            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "structural_attack_summary.csv").exists())
            self.assertEqual(results["dataset"], "nsl-kdd")
            self.assertEqual(args.attacks, ["edge_removal", "edge_addition"])
            self.assertTrue(results["attacks"])
            self.assertEqual(set(results["attacks"][0].keys()), {"attack", "rate", "metrics", "delta"})


if __name__ == "__main__":
    unittest.main()
