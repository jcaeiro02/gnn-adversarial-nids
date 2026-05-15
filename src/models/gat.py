from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from .base import BaseGNN


class GAT_NIDS(BaseGNN):
	def __init__(
		self,
		num_node_features: int,
		hidden_dim: int,
		num_classes: int = 2,
		dropout: float = 0.5,
		heads: int = 4,
		hidden_layers: int = 1,
	):
		super().__init__(num_node_features, hidden_dim, num_classes, dropout)

		self.heads = heads
		self.convs: nn.ModuleList = nn.ModuleList()

		# Determine per-head out channels so that concat heads produce hidden_dim
		def per_head_dim(total: int, heads: int) -> int:
			return total // heads if total % heads == 0 else total

		first_out = per_head_dim(hidden_dim, heads)
		# First layer: map input features to hidden_dim (via multi-head concat)
		self.convs.append(
			GATConv(num_node_features, first_out, heads=heads, concat=True)
		)

		# Additional hidden layers (if any)
		for _ in range(hidden_layers - 1):
			# Each hidden conv expects hidden_dim input and produces hidden_dim output
			per_head = per_head_dim(hidden_dim, heads)
			self.convs.append(GATConv(hidden_dim, per_head, heads=heads, concat=True))

		# Final classifier
		self.classifier = nn.Linear(hidden_dim, num_classes)

	def forward(self, data: Data) -> torch.Tensor:
		x, edge_index = data.x, data.edge_index

		for conv in self.convs:
			x = conv(x, edge_index)
			x = F.elu(x)
			x = F.dropout(x, p=self.dropout, training=self.training)

		logits = self.classifier(x)
		return logits

	def get_embeddings(self, data: Data) -> torch.Tensor:
		x, edge_index = data.x, data.edge_index

		for conv in self.convs:
			x = conv(x, edge_index)
			x = F.elu(x)

		return x
