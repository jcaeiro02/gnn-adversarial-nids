# experiments/03_structural_attacks.py

#!/usr/bin/env python3
"""Strucural-space adversarial attacks against trained GNN-NIDS baselines."""

import argparse
import csv
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test environments
    yaml = None

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import load_split_datasets
from data.download import CICIDS2017_SUBSETS
from models import GCN_NIDS, GAT_NIDS
from training.trainer import Trainer

from attacks.structural import edge_removal_attack
from attacks.structural import edge_addition_attack


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/base_config.yaml")


def load_config(config_path: Path) -> dict:
    if not config_path.exists() or yaml is None:
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, num_node_features: int, hidden_dim: int, dropout: float):
    if model_name == "gcn":
        return GCN_NIDS(num_node_features, hidden_dim, dropout=dropout)
    if model_name == "gat":
        return GAT_NIDS(num_node_features, hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported model type: {model_name}")


def create_attack_run_directory(dataset: str, model: str, k: int, base_dir: Path = Path("results") / "structural_attacks") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{dataset}_{model}_k_{k}_structural_attacks"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def append_csv(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_attacked_dataset(dataset, attack_name: str, rate: float, device: str):
    attacked_graphs = []

    for data in dataset:
        data = data.to(device)

        if attack_name == "edge_removal":
            adv_data = edge_removal_attack(
                data,
                perturbation_rate=rate,
                attack_only_malicious=True,
            )
        elif attack_name == "edge_addition":
            adv_data = edge_addition_attack(
                data,
                perturbation_rate=rate,
                attack_only_malicious=True,
            )
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")

        attacked_graphs.append(adv_data.cpu())

    return attacked_graphs


def validate_checkpoint_metadata(checkpoint_path: Path, expected_k: int) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("metadata", {}) or {}
    checkpoint_k = metadata.get("k")
    if checkpoint_k is not None and int(checkpoint_k) != int(expected_k):
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained with k={checkpoint_k}, but requested k={expected_k}."
        )


def run_structural_attacks(args: argparse.Namespace) -> dict:
    config = load_config(DEFAULT_CONFIG_PATH)
    train_config = config.get("train", config)
    seed = train_config.get("seed", 42)
    set_seed(seed)

    device_config = train_config.get("device", "auto")
    window_size = args.window_size if args.window_size is not None else config.get("window_size", 1000)

    run_dir = create_attack_run_directory(args.dataset, args.model, args.k)

    _, _, test_dataset = load_split_datasets(
        name=args.dataset,
        root="data/graphs",
        rebuild=False,
        window_size=window_size,
        k=args.k,
    )

    num_node_features = test_dataset[0].x.shape[1]

    model = build_model(
        args.model,
        num_node_features=num_node_features,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    trainer = Trainer(
        model=model,
        device=device_config,
        learning_rate=train_config.get("learning_rate", 0.001),
        weight_decay=train_config.get("weight_decay", 0.0005),
        epochs=1,
        patience=1,
        output_dir=str(run_dir),
        batch_size=train_config.get("batch_size", 1),
    )

    logger.info("Loading baseline checkpoint: %s", args.checkpoint)
    validate_checkpoint_metadata(args.checkpoint, args.k)
    trainer.load_checkpoint(args.checkpoint)

    clean_metrics = trainer.evaluate(test_dataset)

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "window_size": window_size,
        "k": args.k,
        "clean_metrics": clean_metrics,
        "attacks": [],
    }

    summary_csv = run_dir / "structural_attack_summary.csv"

    for attack_name in args.attacks:
        for rate in args.rates:
            logger.info("Running %s attack with perturbation_rate=%s", attack_name, rate)

            attacked_dataset = build_attacked_dataset(
                dataset=test_dataset,
                attack_name=attack_name,
                rate=rate,
                device=trainer.device,
            )

            attacked_metrics = trainer.evaluate(attacked_dataset)

            row = {
                "dataset": args.dataset,
                "model": args.model,
                "k": args.k,
                "attack": attack_name,
                "perturbation_rate": rate,
                "clean_accuracy": clean_metrics["accuracy"],
                "clean_precision": clean_metrics["precision"],
                "clean_recall": clean_metrics["recall"],
                "clean_f1": clean_metrics["f1"],
                "clean_roc_auc": clean_metrics.get("roc_auc"),
                "attacked_accuracy": attacked_metrics["accuracy"],
                "attacked_precision": attacked_metrics["precision"],
                "attacked_recall": attacked_metrics["recall"],
                "attacked_f1": attacked_metrics["f1"],
                "attacked_roc_auc": attacked_metrics.get("roc_auc"),
                "delta_accuracy": clean_metrics["accuracy"] - attacked_metrics["accuracy"],
                "delta_f1": clean_metrics["f1"] - attacked_metrics["f1"],
                "delta_recall": clean_metrics["recall"] - attacked_metrics["recall"],
            }

            append_csv(row, summary_csv)

            results["attacks"].append({
                "attack": attack_name,
                "rate": rate,
                "metrics": attacked_metrics,
                "delta": {
                    "accuracy": row["delta_accuracy"],
                    "f1": row["delta_f1"],
                    "recall": row["delta_recall"],
                },
            })

    save_json(results, run_dir / "metrics.json")
    logger.info("Structural attack experiment complete. Results saved to %s", run_dir)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structural edge perturbation attacks on trained GNN baselines.")

    parser.add_argument("--model", choices=["gcn", "gat"], default="gcn")

    cicids_choices = sorted(list(CICIDS2017_SUBSETS.keys()))
    parser.add_argument(
        "--dataset",
        choices=["nsl-kdd"] + cicids_choices,
        default="nsl-kdd",
    )

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=5)

    parser.add_argument("--attacks", nargs="+", choices=["edge_removal", "edge_addition"], default=["edge_removal", "edge_addition"])
    parser.add_argument("--rates", nargs="+", type=float, default=[0.01, 0.03, 0.05, 0.10])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_structural_attacks(args)


if __name__ == "__main__":
    main()