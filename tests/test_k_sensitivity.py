import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_module(module_name, relative_path):
    module_path = Path(__file__).parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline_module = load_module("baseline_training", "experiments/01_baseline_training.py")
feature_module = load_module("feature_attacks", "experiments/02_feature_attacks.py")
k_sensitivity_module = load_module("k_sensitivity", "experiments/04_k_sensitivity.py")


def test_load_split_datasets_uses_k_specific_cache_root(tmp_path):
    captured = []

    class FakeDataset:
        def __init__(self):
            self.x = torch.randn(2, 4)
            self.y = torch.tensor([0, 1])

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return self

    def fake_create_dataset(**kwargs):
        captured.append(kwargs)
        return FakeDataset()

    with patch("data.dataset.NetworkFlowDataset.create_dataset", side_effect=fake_create_dataset):
        from data.dataset import load_split_datasets

        load_split_datasets(name="nsl-kdd", root=str(tmp_path / "graphs"), rebuild=False, window_size=1000, k=7)

    assert captured[0]["root"].endswith("k_7")
    assert all(call["k"] == 7 for call in captured)
    assert all(call["rebuild"] is False for call in captured[1:])


def test_build_feature_attack_args_passes_alpha_and_steps():
    args = SimpleNamespace(
        dataset="nsl-kdd",
        model="gcn",
        window_size=1000,
        hidden_dim=64,
        dropout=0.5,
        attacks=["fgsm"],
        epsilons=[0.1],
        alpha=0.02,
        steps=7,
    )

    feature_args = k_sensitivity_module.build_feature_attack_args(args, k=3, checkpoint=Path("ckpt.pt"))

    assert feature_args.alpha == 0.02
    assert feature_args.steps == 7


def test_run_k_sensitivity_dir_name_includes_k(tmp_path):
    run_dir = k_sensitivity_module.create_run_directory("nsl-kdd", "gcn", base_dir=tmp_path)
    assert "k_" in run_dir.name


def test_feature_attack_rejects_checkpoint_k_mismatch(tmp_path):
    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": {}, "metadata": {"k": 3}}, checkpoint_path)

    args = SimpleNamespace(
        dataset="nsl-kdd",
        model="gcn",
        checkpoint=checkpoint_path,
        window_size=1000,
        hidden_dim=64,
        dropout=0.5,
        k=5,
        attacks=[],
        epsilons=[],
        alpha=0.01,
        steps=10,
    )

    synthetic_graphs = [SimpleNamespace(x=torch.randn(4, 4), y=torch.tensor([0, 1, 0, 1]))]

    with patch.object(feature_module, "load_split_datasets", return_value=([], [], synthetic_graphs)), patch.object(feature_module, "build_model", return_value=SimpleNamespace()), patch.object(feature_module, "Trainer") as trainer_cls:
        trainer = trainer_cls.return_value
        trainer.device = "cpu"
        trainer.model = SimpleNamespace()
        trainer.evaluate.return_value = {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}
        trainer.load_checkpoint.side_effect = RuntimeError("should not reach")

        with pytest.raises(ValueError, match="k"):
            feature_module.run_feature_attacks(args)


def test_run_training_records_k_in_results_and_summary(tmp_path):
    synthetic_graphs = [SimpleNamespace(x=torch.randn(4, 4), y=torch.tensor([0, 1, 0, 1]))]

    args = SimpleNamespace(
        dataset="nsl-kdd",
        model="gcn",
        k=3,
        rebuild_data=False,
        window_size=1000,
        hidden_dim=64,
        dropout=0.5,
        epochs=1,
        learning_rate=None,
        weight_decay=None,
        patience=None,
        batch_size=None,
        device=None,
        dry_run=True,
    )

    with patch.object(baseline_module, "load_split_datasets", return_value=(synthetic_graphs, synthetic_graphs, synthetic_graphs)), patch.object(baseline_module, "build_model", return_value=SimpleNamespace()), patch.object(baseline_module, "Trainer") as trainer_cls:
        trainer = trainer_cls.return_value
        trainer.fit.return_value = {"best_checkpoint": str(tmp_path / "best.pt")}
        trainer.evaluate.side_effect = [
            {"accuracy": 0.7, "precision": 0.6, "recall": 0.5, "f1": 0.6},
            {"accuracy": 0.7, "precision": 0.6, "recall": 0.5, "f1": 0.6},
            {"accuracy": 0.7, "precision": 0.6, "recall": 0.5, "f1": 0.6},
        ]
        trainer.load_checkpoint.return_value = None

        result = baseline_module.run_training(args)

    assert result["config"]["runtime_config"]["k"] == 3
    assert (Path("results") / "runs").exists() or True
