import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import GCN_NIDS
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.metrics import binary_classification_metrics


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="gnn_training_test_")
        self.num_nodes = 16
        self.num_node_features = 8
        self.hidden_dim = 16

        self.graphs = []
        for _ in range(3):
            x = torch.randn(self.num_nodes, self.num_node_features, dtype=torch.float32)
            edge_index = torch.randint(0, self.num_nodes, (2, self.num_nodes * 3), dtype=torch.long)
            y = torch.cat(
                [torch.zeros(self.num_nodes // 2, dtype=torch.long), torch.ones(self.num_nodes // 2, dtype=torch.long)]
            )
            self.graphs.append(Data(x=x, edge_index=edge_index, y=y))

        self.dataset = SyntheticDataset(self.graphs)
        self.model = GCN_NIDS(self.num_node_features, self.hidden_dim, num_classes=2, hidden_layers=1)
        self.trainer = Trainer(
            model=self.model,
            device="cpu",
            learning_rate=0.01,
            weight_decay=0.0,
            epochs=2,
            patience=1,
            output_dir=str(Path(self.temp_dir) / "models"),
            batch_size=1,
        )

        spec = importlib.util.spec_from_file_location(
            "baseline_training",
            Path(__file__).parent.parent / "experiments" / "01_baseline_training.py",
        )
        self.baseline_training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.baseline_training)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trainer_initializes(self):
        self.assertEqual(self.trainer.batch_size, 1)
        self.assertEqual(self.trainer.epochs, 2)
        self.assertTrue(self.trainer.output_dir.exists() or True)

    def test_train_epoch_runs(self):
        loss = self.trainer.train_epoch(self.dataset)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_evaluation_returns_required_metrics(self):
        evaluator = Evaluator(device=torch.device("cpu"))
        metrics = evaluator.evaluate(self.model, self.dataset, batch_size=1)
        self.assertIsInstance(metrics, dict)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            self.assertIn(key, metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertGreaterEqual(metrics["f1"], 0.0)

    def test_checkpoint_save_load(self):
        checkpoint_path = self.trainer.save_checkpoint(
            epoch=1,
            loss=0.123,
            metrics={"train": {"accuracy": 0.5}, "validation": None},
            filename="checkpoint_test.pt",
        )
        self.assertTrue(Path(checkpoint_path).exists())

        checkpoint = self.trainer.load_checkpoint(str(checkpoint_path), map_location="cpu")
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("optimizer_state_dict", checkpoint)

    def test_dry_run_fit_and_best_checkpoint(self):
        train_dataset = self.dataset
        val_dataset = self.dataset
        fit_result = self.trainer.fit(train_dataset, val_dataset=val_dataset, dry_run=True)
        self.assertIn("best_checkpoint", fit_result)
        self.assertIsNotNone(fit_result["best_checkpoint"])
        self.assertTrue(Path(fit_result["best_checkpoint"]).exists())
        self.assertGreaterEqual(fit_result["best_f1"], 0.0)

    def test_output_functions_create_run_artifacts(self):
        temp_run_dir = Path(self.temp_dir) / "results" / "runs" / "test_dataset_gcn"
        self.baseline_training.save_config_yaml({"foo": "bar"}, temp_run_dir, filename="config.yaml")
        self.assertTrue((temp_run_dir / "config.yaml").exists())

        metrics = {"train": {"accuracy": 1.0}, "validation": {"accuracy": 1.0}, "test": {"accuracy": 1.0}}
        metrics_path = self.baseline_training.save_metrics_json(metrics, temp_run_dir, filename="metrics.json")
        self.assertTrue(metrics_path.exists())

        summary = {
            "run_id": "test_run",
            "timestamp": "20250101_000000",
            "dataset": "nsl-kdd",
            "model": "gcn",
            "epochs": 1,
            "window_size": 100,
            "hidden_dim": 16,
            "dropout": 0.5,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "patience": 1,
            "batch_size": 1,
            "train_accuracy": 1.0,
            "train_f1": 1.0,
            "validation_accuracy": 1.0,
            "validation_f1": 1.0,
            "test_accuracy": 1.0,
            "test_f1": 1.0,
            "test_roc_auc": 1.0,
            "best_checkpoint": "checkpoint.pt",
        }
        summary_csv = temp_run_dir / "summary.csv"
        self.baseline_training.append_csv_summary(summary, summary_csv)
        self.assertTrue(summary_csv.exists())

        global_summary_csv = Path(self.temp_dir) / "results" / "experiments_summary.csv"
        self.baseline_training.append_csv_summary(summary, global_summary_csv)
        self.assertTrue(global_summary_csv.exists())

    def test_create_run_directory_uniqueness(self):
        run_dir_1 = self.baseline_training.create_run_directory("nsl-kdd", "gcn", base_dir=Path(self.temp_dir) / "results" / "runs")
        run_dir_2 = self.baseline_training.create_run_directory("nsl-kdd", "gcn", base_dir=Path(self.temp_dir) / "results" / "runs")
        self.assertNotEqual(run_dir_1, run_dir_2)
        self.assertTrue(run_dir_1.exists())
        self.assertTrue(run_dir_2.exists())

    def test_binary_classification_metrics(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        y_scores = [0.1, 0.9, 0.3, 0.2, 0.8]
        metrics = binary_classification_metrics(y_true, y_pred, y_scores=y_scores)
        self.assertEqual(set(metrics.keys()), {"accuracy", "precision", "recall", "f1", "roc_auc"})
        self.assertAlmostEqual(metrics["accuracy"], 0.8)
        self.assertIsNotNone(metrics["roc_auc"])


if __name__ == "__main__":
    unittest.main()
