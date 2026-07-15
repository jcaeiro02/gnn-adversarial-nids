#!/usr/bin/env python3
"""Run the k-NN neighborhood-size sensitivity experiment.

For each dataset, model and k value, this experiment:

1. Rebuilds or loads graphs constructed with the selected k.
2. Trains a new baseline model.
3. Loads the generated best checkpoint.
4. Runs FGSM and PGD feature attacks.
5. Measures feature-induced neighbor churn.
6. Aggregates the results across k values.
"""

import argparse
import csv
import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.download import CICIDS2017_SUBSETS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Dynamic experiment-module loading
# ----------------------------------------------------------------------

def load_experiment_module(
    module_name: str,
    file_path: Path,
) -> ModuleType:
    """Load an experiment script whose filename starts with a number."""

    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experiment module: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


BASELINE_MODULE = load_experiment_module(
    "baseline_training",
    Path(__file__).parent / "01_baseline_training.py",
)

FEATURE_ATTACK_MODULE = load_experiment_module(
    "feature_attacks",
    Path(__file__).parent / "02_feature_attacks.py",
)


# ----------------------------------------------------------------------
# Result helpers
# ----------------------------------------------------------------------

def create_run_directory(
    dataset: str,
    model: str,
    k_values: list[int] | None = None,
    base_dir: Path = Path("results") / "k_sensitivity",
) -> Path:
    """Create the directory for the complete k-sensitivity experiment."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    k_suffix = "" if not k_values else "_" + "_".join(str(k) for k in k_values)
    run_name = f"{timestamp}_{dataset}_{model}_k_sensitivity{k_suffix}"

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
        writer = csv.DictWriter(
            csvfile,
            fieldnames=list(row.keys()),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


# ----------------------------------------------------------------------
# Experiment orchestration
# ----------------------------------------------------------------------

def build_baseline_args(
    experiment_args: argparse.Namespace,
    k: int,
) -> argparse.Namespace:
    """Create the arguments expected by run_training()."""

    return argparse.Namespace(
        dataset=experiment_args.dataset,
        model=experiment_args.model,
        k=k,
        rebuild_data=experiment_args.rebuild_data,
        window_size=experiment_args.window_size,
        hidden_dim=experiment_args.hidden_dim,
        dropout=experiment_args.dropout,
        epochs=experiment_args.epochs,
        learning_rate=experiment_args.learning_rate,
        weight_decay=experiment_args.weight_decay,
        patience=experiment_args.patience,
        batch_size=experiment_args.batch_size,
        device=experiment_args.device,
        dry_run=experiment_args.dry_run,
    )


def build_feature_attack_args(
    experiment_args: argparse.Namespace,
    k: int,
    checkpoint: Path,
) -> argparse.Namespace:
    """Create the arguments expected by run_feature_attacks()."""

    return argparse.Namespace(
        dataset=experiment_args.dataset,
        model=experiment_args.model,
        checkpoint=checkpoint,
        k=k,
        window_size=experiment_args.window_size,
        hidden_dim=experiment_args.hidden_dim,
        dropout=experiment_args.dropout,
        attacks=experiment_args.attacks,
        epsilons=experiment_args.epsilons,
        alpha=experiment_args.alpha,
        steps=experiment_args.steps,
    )


def extract_checkpoint(baseline_result: dict) -> Path:
    """Extract the best-checkpoint path returned by run_training().

    Adjust this function to the exact dictionary returned by
    01_baseline_training.py.
    """

    possible_keys = [
        "best_checkpoint",
        "checkpoint",
        "checkpoint_path",
    ]

    for key in possible_keys:
        value = baseline_result.get(key)

        if value:
            return Path(value)

    fit_result = baseline_result.get("fit", {})

    if fit_result.get("best_checkpoint"):
        return Path(fit_result["best_checkpoint"])

    raise KeyError(
        "run_training() did not return the best-checkpoint path. "
        "Update run_training() to return it or adjust extract_checkpoint()."
    )


def build_summary_rows(
    k: int,
    baseline_result: dict,
    feature_result: dict,
) -> list[dict]:
    """Flatten results into one CSV row per attack and epsilon."""

    clean_metrics = feature_result["clean_metrics"]
    rows = []

    for attack_result in feature_result["attacks"]:
        attacked_metrics = attack_result["metrics"]

        row = {
            "k": k,
            "attack": attack_result["attack"],
            "epsilon": attack_result["epsilon"],

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

            "delta_accuracy": (
                clean_metrics["accuracy"]
                - attacked_metrics["accuracy"]
            ),
            "delta_f1": (
                clean_metrics["f1"]
                - attacked_metrics["f1"]
            ),
            "delta_recall": (
                clean_metrics["recall"]
                - attacked_metrics["recall"]
            ),

            "neighbor_churn_rate": attack_result.get("neighbor_churn_rate"),
            "neighbor_churn_rates": attack_result.get("neighbor_churn_rates"),
        }

        rows.append(row)

    return rows


def run_k_sensitivity(args: argparse.Namespace) -> dict:
    """Run baseline training and feature attacks for every requested k."""

    experiment_dir = create_run_directory(
        dataset=args.dataset,
        model=args.model,
        k_values=args.k_values,
    )

    summary_csv = experiment_dir / "k_sensitivity_summary.csv"

    complete_results = {
        "dataset": args.dataset,
        "model": args.model,
        "k_values": args.k_values,
        "attacks": args.attacks,
        "epsilons": args.epsilons,
        "configurations": [],
    }

    for k in args.k_values:
        logger.info("=" * 70)
        logger.info(
            "Running k-sensitivity configuration: "
            "dataset=%s, model=%s, k=%d",
            args.dataset,
            args.model,
            k,
        )
        logger.info("=" * 70)

        # --------------------------------------------------------------
        # 1. Train a new baseline for this value of k
        # --------------------------------------------------------------

        baseline_args = build_baseline_args(args, k)

        baseline_result = BASELINE_MODULE.run_training(
            baseline_args
        )

        checkpoint = extract_checkpoint(baseline_result)

        logger.info(
            "Baseline training complete for k=%d. Checkpoint: %s",
            k,
            checkpoint,
        )

        if args.dry_run:
            configuration_result = {
                "k": k,
                "checkpoint": str(checkpoint),
                "baseline": baseline_result,
                "feature_attacks": None,
                "status": "dry_run_completed",
            }

            complete_results["configurations"].append(
                configuration_result
            )

            save_json(
                complete_results,
                experiment_dir / "metrics.json",
            )

            logger.info(
                "Dry-run completed for k=%d; feature attacks were skipped.",
                k,
            )

            continue

        # Só é executado numa experiência real
        feature_args = build_feature_attack_args(
            experiment_args=args,
            k=k,
            checkpoint=checkpoint,
        )

        feature_result = FEATURE_ATTACK_MODULE.run_feature_attacks(
            feature_args
        )


        # --------------------------------------------------------------
        # 2. Run FGSM/PGD and neighbor-churn evaluation
        # --------------------------------------------------------------

        feature_args = build_feature_attack_args(
            experiment_args=args,
            k=k,
            checkpoint=checkpoint,
        )

        feature_result = FEATURE_ATTACK_MODULE.run_feature_attacks(
            feature_args
        )

        configuration_result = {
            "k": k,
            "checkpoint": str(checkpoint),
            "baseline": baseline_result,
            "feature_attacks": feature_result,
        }

        complete_results["configurations"].append(
            configuration_result
        )

        # Save partial results after each k value
        save_json(
            complete_results,
            experiment_dir / "metrics.json",
        )

        for row in build_summary_rows(
            k=k,
            baseline_result=baseline_result,
            feature_result=feature_result,
        ):
            row = {
                "dataset": args.dataset,
                "model": args.model,
                **row,
            }

            append_csv(row, summary_csv)

    logger.info(
        "k-sensitivity experiment complete. Results saved to %s",
        experiment_dir,
    )

    return complete_results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate GNN-NIDS models across multiple "
            "k-NN neighborhood sizes."
        )
    )

    parser.add_argument(
        "--model",
        choices=["gcn", "gat"],
        default="gcn",
    )

    cicids_choices = sorted(CICIDS2017_SUBSETS.keys())

    parser.add_argument(
        "--dataset",
        choices=["nsl-kdd"] + cicids_choices,
        default="nsl-kdd",
    )

    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[3, 5, 10, 20, 50],
    )

    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=["fgsm", "pgd"],
        default=["fgsm", "pgd"],
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.05, 0.10],
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help=(
            "Rebuild processed graph datasets for each k value. "
            "Existing k-specific caches are otherwise reused."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use lightweight synthetic data for a smoke test.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if any(k < 1 for k in args.k_values):
        raise ValueError("Every k value must be at least 1.")

    if len(set(args.k_values)) != len(args.k_values):
        raise ValueError("Duplicate values were provided in --k-values.")

    run_k_sensitivity(args)


if __name__ == "__main__":
    main()