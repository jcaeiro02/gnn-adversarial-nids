import unittest
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import BaseGNN, GCN_NIDS, GAT_NIDS
from torch_geometric.data import Data


class TestModels(unittest.TestCase):
	def setUp(self):
		self.num_nodes = 50
		self.num_node_features = 16
		self.num_classes = 2
		self.hidden_dim = 32

		# Random node features
		x = torch.randn(self.num_nodes, self.num_node_features, dtype=torch.float32)

		# Random edges (avoid trivial zero edges)
		num_edges = self.num_nodes * 4
		edge_index = torch.randint(0, self.num_nodes, (2, num_edges), dtype=torch.long)

		# Binary node labels
		y = torch.randint(0, 2, (self.num_nodes,), dtype=torch.long)

		self.data = Data(x=x, edge_index=edge_index, y=y)

	def _run_basic_model_checks(self, model: BaseGNN, device: torch.device):
		model.to(device)
		data = self.data.clone()
		data.x = data.x.to(device)
		data.edge_index = data.edge_index.to(device)
		data.y = data.y.to(device)

		model.train()
		logits = model(data)
		self.assertEqual(logits.shape[0], self.num_nodes)
		self.assertEqual(logits.shape[1], self.num_classes)

		embeddings = model.get_embeddings(data)
		self.assertEqual(embeddings.shape[0], self.num_nodes)

		# Backward check
		loss = logits[:, 1].mean()
		loss.backward()

		# Ensure gradients exist for some parameters
		self.assertTrue(any(p.grad is not None for p in model.parameters()))

	def test_gcn_forward_and_backward_cpu(self):
		model = GCN_NIDS(self.num_node_features, self.hidden_dim, num_classes=self.num_classes, hidden_layers=2)
		cfg = model.get_config()
		self.assertIn("num_parameters", cfg)
		self.assertGreater(model.num_parameters(), 0)
		self._run_basic_model_checks(model, torch.device("cpu"))

	def test_gat_forward_and_backward_cpu(self):
		model = GAT_NIDS(self.num_node_features, self.hidden_dim, num_classes=self.num_classes, heads=4, hidden_layers=2)
		self.assertGreater(model.num_parameters(), 0)
		self._run_basic_model_checks(model, torch.device("cpu"))

	def test_models_on_cuda_if_available(self):
		if torch.cuda.is_available():
			device = torch.device("cuda")
			model = GCN_NIDS(self.num_node_features, self.hidden_dim, num_classes=self.num_classes)
			self._run_basic_model_checks(model, device)

			model2 = GAT_NIDS(self.num_node_features, self.hidden_dim, num_classes=self.num_classes, heads=4)
			self._run_basic_model_checks(model2, device)
		else:
			self.skipTest("CUDA not available")


if __name__ == "__main__":
	unittest.main()