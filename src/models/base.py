from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import Data


class BaseGNN(nn.Module, ABC):
	"""Abstract base class for GNN models used for node classification.

	Subclasses must implement `forward(data)` and may implement
	`get_embeddings(data)` to return node-level embeddings.
	"""

	def __init__(
		self,
		num_node_features: int,
		hidden_dim: int,
		num_classes: int = 2,
		dropout: float = 0.5,
	):
		super().__init__()
		self.num_node_features = num_node_features
		self.hidden_dim = hidden_dim
		self.num_classes = num_classes
		self.dropout = float(dropout)

	@abstractmethod
	def forward(self, data: Data) -> torch.Tensor:
		"""Forward pass returning node-level logits of shape [num_nodes, num_classes]."""

	def get_config(self) -> Dict:
		return {
			"num_node_features": self.num_node_features,
			"hidden_dim": self.hidden_dim,
			"num_classes": self.num_classes,
			"dropout": self.dropout,
			"num_parameters": self.num_parameters(),
		}

	def num_parameters(self) -> int:
		return sum(p.numel() for p in self.parameters())

	def get_embeddings(self, data: Data) -> torch.Tensor:
		"""Optional: return node embeddings for a given input `data`.

		By default, models may override this to return the hidden
		representation used by the classifier.
		"""
		raise NotImplementedError()
