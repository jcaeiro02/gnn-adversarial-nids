from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from .base import BaseGNN


class GCN_NIDS(BaseGNN):
	def __init__(
		self,
		num_node_features: int,
		hidden_dim: int,
		num_classes: int = 2,
		dropout: float = 0.5,
		hidden_layers: int = 1,
	):
		super().__init__(num_node_features, hidden_dim, num_classes, dropout)

		self.convs: nn.ModuleList = nn.ModuleList()

		# Input layer
		self.convs.append(GCNConv(num_node_features, hidden_dim))

		# Hidden layers
		for _ in range(hidden_layers - 1):
			self.convs.append(GCNConv(hidden_dim, hidden_dim))

		# Classifier applied per-node
		self.classifier = nn.Linear(hidden_dim, num_classes)

	def forward(self, data: Data) -> torch.Tensor:
		x, edge_index = data.x, data.edge_index

		for conv in self.convs:
			x = conv(x, edge_index)
			x = F.relu(x)
			x = F.dropout(x, p=self.dropout, training=self.training)

		logits = self.classifier(x)
		return logits

	def get_embeddings(self, data: Data) -> torch.Tensor:
		x, edge_index = data.x, data.edge_index

		for conv in self.convs:
			x = conv(x, edge_index)
			x = F.relu(x)

		return x
