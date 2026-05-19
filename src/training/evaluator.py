from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    from torch_geometric.loader import DataLoader
except ImportError:  # compatibility with older PyG releases
    from torch_geometric.data import DataLoader

from utils.metrics import binary_classification_metrics


class Evaluator:
    """Evaluator for node-level binary classification on PyG graphs."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self, model: torch.nn.Module, dataset: Any, batch_size: int = 1) -> Dict[str, Any]:
        model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_true = []
        all_pred = []
        all_scores = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = model(batch)
                probabilities = torch.softmax(logits, dim=-1)
                predictions = logits.argmax(dim=-1)

                labels = batch.y.view(-1).cpu().numpy()
                all_true.append(labels)
                all_pred.append(predictions.cpu().numpy())
                all_scores.append(probabilities[:, 1].cpu().numpy())

        if len(all_true) == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": None,
            }

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        y_scores = np.concatenate(all_scores)

        return binary_classification_metrics(y_true, y_pred, y_scores=y_scores)
