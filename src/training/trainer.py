import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim import Adam

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

from .evaluator import Evaluator


logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for node classification GNN baseline models."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        epochs: int = 100,
        patience: int = 15,
        output_dir: str = "results/models",
        batch_size: int = 1,
    ):
        self.model = model
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.output_dir = Path(output_dir)
        self.checkpoint_path = self.output_dir / "best_checkpoint.pt"

        self.optimizer = Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def _resolve_device(self, device: str) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def train_epoch(self, train_dataset: Any) -> float:
        self.model.train()
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        total_loss = 0.0
        batch_count = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch)
            targets = batch.y.view(-1)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

        return total_loss / max(1, batch_count)

    def evaluate(self, dataset: Any) -> Dict[str, Optional[float]]:
        evaluator = Evaluator(device=self.device)
        return evaluator.evaluate(self.model, dataset, batch_size=self.batch_size)

    def fit(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        num_epochs = min(self.epochs, 2) if dry_run else self.epochs
        best_f1 = -1.0
        best_checkpoint_path = None
        patience_counter = 0
        history = []

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_dataset)
            train_metrics = self.evaluate(train_dataset)
            val_metrics = self.evaluate(val_dataset) if val_dataset is not None else None

            current_f1 = (
                float(val_metrics.get("f1", 0.0)) if val_metrics is not None else float(train_metrics.get("f1", 0.0))
            )

            if current_f1 > best_f1:
                best_f1 = current_f1
                patience_counter = 0
                best_checkpoint_path = self.save_checkpoint(
                    epoch=epoch,
                    loss=train_loss,
                    metrics={
                        "train": train_metrics,
                        "validation": val_metrics,
                    },
                )
            else:
                patience_counter += 1

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "best_f1": best_f1,
                }
            )

            if val_dataset is not None and patience_counter >= self.patience:
                logger.info(
                    "Early stopping triggered at epoch %d after %d epochs without improvement.",
                    epoch,
                    self.patience,
                )
                break

        return {
            "history": history,
            "best_f1": best_f1,
            "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        }

    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, Any],
        filename: str = "best_checkpoint.pt",
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "metrics": metrics,
        }
        target_path = self.output_dir / filename
        torch.save(checkpoint, target_path)
        logger.info("Saved checkpoint: %s", target_path)
        return target_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        target_path = Path(checkpoint_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            target_path,
            map_location=map_location or self.device,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
