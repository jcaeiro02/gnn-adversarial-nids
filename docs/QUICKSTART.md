# Phase 1 Quick Start Guide

## Overview
Complete implementation of NSL-KDD data pipeline with flow-centric graph representation for GNN-based intrusion detection.

## Quick Start

### 1. Generate Sample Data (for testing)
```bash
python scripts/generate_sample_data.py --samples 1000
```
Creates synthetic NSL-KDD data in `data/raw/nsl-kdd/` for local testing.

### 2. Run Full Pipeline Verification
```bash
python scripts/verify_pipeline.py
```
Demonstrates the complete workflow:
- Data generation
- Loading and preprocessing
- Graph construction
- PyG dataset creation

### 3. Run Test Suite
```bash
# All tests
python -m pytest tests/test_data.py -v

# Specific test class
python -m pytest tests/test_data.py::TestFlowGraphBuilder -v

# With coverage
python -m pytest tests/test_data.py -v --cov=src/data
```

## Component Usage Examples

### Load and Preprocess Data
```python
from src.data.preprocess import NSLKDDPreprocessor

preprocessor = NSLKDDPreprocessor()
df = preprocessor.load_data("data/raw/nsl-kdd/KDDTrain+.txt")
X, y = preprocessor.preprocess(df, fit=True)
preprocessor.save_preprocessed(X, y, "data/processed")
```

### Build Flow-Centric Graph
```python
from src.data.graph_builder import FlowGraphBuilder
import numpy as np

builder = FlowGraphBuilder()
X = np.random.randn(1000, 41).astype(np.float32)  # 1000 flows, 41 features
y = np.random.randint(0, 2, 1000)  # Binary labels

# kNN graph (default)
graph = builder.build_graph(X, y, method="knn", k=5)

# Similarity graph
graph = builder.build_graph(X, y, method="similarity", threshold=0.8)

print(f"Graph: {graph}")
print(f"Nodes: {graph.x.shape[0]}")
print(f"Edges: {graph.edge_index.shape[1]}")
```

### Create PyG Dataset
```python
from src.data.dataset import NetworkFlowDataset

# Create dataset
dataset = NetworkFlowDataset.create_dataset(
    name="nsl-kdd",
    split="train",
    root="data/graphs"
)

# Access samples
sample = dataset[0]
print(sample)  # PyG Data object
```

## File Structure

```
src/data/
├── download.py          # DatasetDownloader
├── preprocess.py        # NSLKDDPreprocessor
├── graph_builder.py     # FlowGraphBuilder
└── dataset.py           # NetworkFlowDataset

tests/
└── test_data.py         # Test suite

scripts/
├── generate_sample_data.py    # Generate synthetic data
└── verify_pipeline.py         # Full pipeline demo

DATA_PIPELINE.md         # Complete documentation
QUICKSTART.md            # This file
```

## Key Features Implemented

✓ NSL-KDD dataset download and validation
✓ Complete preprocessing pipeline (encoding, normalization, label conversion)
✓ Flow-centric graph construction (kNN and similarity methods)
✓ PyTorch Geometric integration
✓ Comprehensive test suite (23+ tests)
✓ Logging and error handling throughout
✓ Sample data generation for testing

## Graph Representation

**Flow-Centric Model:**
- **Nodes:** Each network flow/sample
- **Node Features:** 41 preprocessed NSL-KDD attributes
- **Node Labels:** 0=normal, 1=attack
- **Edges:** Feature similarity (cosine) between flows
- **Edge Weights:** 1 - distance (normalized similarity)

**Default Graph Configuration:**
- **Method:** k-nearest neighbors
- **k:** 5 neighbors per node
- **Metric:** Cosine similarity
- **No self-loops:** Enforced

## Output Formats

### Preprocessed Data
- X: (N_samples, 41) numpy array
- y: (N_samples,) numpy array

### PyG Graph
```python
Data(
    x=[N, 41],           # Node features
    y=[N],              # Node labels
    edge_index=[2, E],  # Edge indices
    edge_attr=[E, 1],   # Edge weights
)
```

## Common Tasks

### Access graph statistics
```python
graph = builder.build_graph(X, y, method="knn", k=5)
print(f"Num nodes: {graph.x.shape[0]}")
print(f"Num edges: {graph.edge_index.shape[1]}")
print(f"Graph density: {2*graph.edge_index.shape[1]/(graph.x.shape[0]**2)}")
```

### Use different similarity metric
```python
graph = builder.build_graph(X, y, method="knn", k=5, metric="euclidean")
```

### Use different threshold
```python
graph = builder.build_graph(X, y, method="similarity", threshold=0.7)
```

### Access dataset metadata
```python
dataset = NetworkFlowDataset.create_dataset()
stats = dataset.load_statistics()
print(f"Num nodes: {stats['num_nodes']}")
print(f"Num edges: {stats['num_edges']}")
print(f"Class distribution: {stats['class_distribution']}")
```

## Docker Usage

Build and run with Docker:
```bash
cd docker
./build.sh
./run.sh
```

Inside container, run tests:
```bash
python -m pytest tests/test_data.py -v
```

## Troubleshooting

### Missing data files
```bash
# Generate sample data
python scripts/generate_sample_data.py
```

### Import errors
```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Tests fail
```bash
# Run with verbose output
python -m pytest tests/test_data.py -vv -s
```

## Performance

- **Preprocessing:** ~2-5 min for full NSL-KDD (300K samples)
- **Graph building:** ~1-2 min for kNN graph
- **Memory:** ~300MB for full dataset in memory

## Next Steps

1. **Phase 2:** Implement GNN models (GCN, GAT)
2. **Phase 3:** Implement attacks (FGSM, PGD, etc.)
3. **Phase 4:** Implement defenses (adversarial training, etc.)

## References

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl-kdd.html)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Complete Documentation](DATA_PIPELINE.md)
