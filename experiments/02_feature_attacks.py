# experiments/02_feature_attacks.py

#!/usr/bin/env python3
"""Feature-space adversarial attacks against trained GNN-NIDS baselines."""

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
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import load_split_datasets
from data.download import CICIDS2017_SUBSETS
from models import GCN_NIDS, GAT_NIDS
from training.trainer import Trainer

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/base_config.yaml")


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
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


def create_attack_run_directory(dataset: str, model: str, base_dir: Path = Path("results") / "feature_attacks") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{dataset}_{model}_feature_attacks"
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


def build_attacked_dataset(model, dataset, attack_name: str, epsilon: float, alpha: float, steps: int, device: str):
    attacked_graphs = []

    for data in dataset:
        data = data.to(device)

        if attack_name == "fgsm":
            adv_data = fgsm_attack(model, data, epsilon=epsilon)
        elif attack_name == "pgd":
            adv_data = pgd_attack(model, data, epsilon=epsilon, alpha=alpha, steps=steps)
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")

        attacked_graphs.append(adv_data.cpu())

    return attacked_graphs


def run_feature_attacks(args: argparse.Namespace) -> dict:
    config = load_config(DEFAULT_CONFIG_PATH)
    train_config = config.get("train", config)
    seed = train_config.get("seed", 42)
    set_seed(seed)

    device_config = train_config.get("device", "auto")
    window_size = args.window_size if args.window_size is not None else config.get("window_size", 1000)

    run_dir = create_attack_run_directory(args.dataset, args.model)

    _, _, test_dataset = load_split_datasets(
        name=args.dataset,
        root="data/graphs",
        rebuild=False,
        window_size=window_size,
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
    trainer.load_checkpoint(args.checkpoint)

    clean_metrics = trainer.evaluate(test_dataset)

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "window_size": window_size,
        "clean_metrics": clean_metrics,
        "attacks": [],
    }

    summary_csv = run_dir / "feature_attack_summary.csv"

    for attack_name in args.attacks:
        for epsilon in args.epsilons:
            logger.info("Running %s attack with epsilon=%s", attack_name, epsilon)

            attacked_dataset = build_attacked_dataset(
                model=trainer.model,
                dataset=test_dataset,
                attack_name=attack_name,
                epsilon=epsilon,
                alpha=args.alpha,
                steps=args.steps,
                device=trainer.device,
            )

            attacked_metrics = trainer.evaluate(attacked_dataset)

            row = {
                "dataset": args.dataset,
                "model": args.model,
                "attack": attack_name,
                "epsilon": epsilon,
                "alpha": args.alpha if attack_name == "pgd" else None,
                "steps": args.steps if attack_name == "pgd" else None,
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
                "epsilon": epsilon,
                "metrics": attacked_metrics,
                "delta": {
                    "accuracy": row["delta_accuracy"],
                    "f1": row["delta_f1"],
                    "recall": row["delta_recall"],
                },
            })

    save_json(results, run_dir / "metrics.json")
    logger.info("Feature attack experiment complete. Results saved to %s", run_dir)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FGSM/PGD feature-space attacks on trained GNN baselines.")

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

    parser.add_argument("--attacks", nargs="+", choices=["fgsm", "pgd"], default=["fgsm", "pgd"])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.01, 0.03, 0.05, 0.10])

    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_feature_attacks(args)


if __name__ == "__main__":
    main()