# Phase 1: NSL-KDD Data Pipeline with Flow-Centric Graph Representation

## Overview

This Phase 1 implements a complete data pipeline for NSL-KDD dataset using a **flow-centric graph representation** suitable for GNN-based network intrusion detection.

### Key Concepts

**Flow-Centric Graph Representation:**
- Each network flow/sample is represented as a **NODE** in the graph
- Node features are the preprocessed NSL-KDD network flow attributes (41 features)
- Node labels indicate if the flow is normal (0) or attack (1)
- **Edges represent relationships between flows** (e.g., feature similarity), NOT communication between hosts
- This representation enables GNN models to learn from flow-level patterns and relationships

## Architecture

### Components

#### 1. **download.py** - DatasetDownloader
Downloads NSL-KDD datasets from remote sources.

**Class:** `DatasetDownloader`

**Methods:**
- `download_nsl_kdd(force=False)` - Download training and test files
- `download_all(force=False)` - Download all available datasets
- `download_file(url, filename, force=False)` - Download single file
- `validate_file(filename)` - Validate downloaded files
- `list_datasets()` - List available datasets
- `get_dataset_path(dataset_name)` - Get path to a dataset

**Features:**
- Automatic directory creation
- Validation of downloaded files
- Comprehensive error handling and logging
- Force re-download option

#### 2. **preprocess.py** - NSLKDDPreprocessor
Loads, encodes, normalizes, and converts NSL-KDD data.

**Class:** `NSLKDDPreprocessor`

**Methods:**
- `load_data(filepath)` - Load raw NSL-KDD CSV
- `preprocess(df, fit=True)` - Complete preprocessing pipeline
- `save_preprocessed(X, y, output_dir)` - Save processed data
- `load_preprocessed(state_path)` - Load preprocessor state

**Features:**
- Handles all 41 NSL-KDD features
- Label encoding: protocol_type, service, flag
- Missing value handling (NaN, inf, -inf)
- Binary label conversion: normal=0, all attacks=1
- StandardScaler normalization
- Stateful preprocessing (fit/transform pattern)

**NSL-KDD Columns:**
```
duration, protocol_type, service, flag, src_bytes, dst_bytes, land,
wrong_fragment, urgent, hot, num_failed_logins, logged_in, num_compromised,
root_shell, su_attempted, num_root, num_file_creations, num_shells,
num_access_files, num_outbound_cmds, is_host_login, is_guest_login,
count, srv_count, serror_rate, srv_serror_rate, rerror_rate,
srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate,
dst_host_count, dst_host_srv_count, dst_host_same_srv_rate,
dst_host_diff_srv_rate, dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate, dst_host_serror_rate,
dst_host_srv_serror_rate, dst_host_rerror_rate,
dst_host_srv_rerror_rate, label
```

#### 3. **graph_builder.py** - FlowGraphBuilder
Constructs flow-centric graphs with configurable edge strategies.

**Class:** `FlowGraphBuilder`

**Methods:**
- `build_graph(X, y, method='knn', **kwargs)` - Build single graph
- `build_knn_graph(X, k=5, metric='cosine')` - k-NN graph construction
- `build_similarity_graph(X, threshold=0.8, metric='cosine')` - Threshold-based graph
- `build_dataset(X_list, y_list, method='knn', ...)` - Build multiple graphs

**Features:**
- **kNN Graph (default):** k-nearest neighbors using cosine similarity
  - k=5 (default)
  - No self-loops
  - Edge weights = 1 - distance
- **Similarity Graph:** Threshold-based similarity connections
  - threshold=0.8 (default)
  - Bidirectional edges
- Graph statistics logging:
  - Number of nodes
  - Number of edges
  - Graph density
  - Average degree
  - Class distribution
- PyTorch Geometric compatible output

#### 4. **dataset.py** - NetworkFlowDataset
Manages PyG datasets with save/load and caching.

**Class:** `NetworkFlowDataset(InMemoryDataset)`

**Methods:**
- `create_dataset(name, split, root, rebuild=False)` - Factory method
- `download()` - Download raw data
- `process()` - Process data into graphs
- `get(idx)` - Get sample by index
- `load_statistics()` - Load graph statistics

**Features:**
- PyTorch Geometric InMemoryDataset integration
- Train/test split support
- Automatic caching
- Statistics tracking
- Clean separation of raw/processed data

#### 5. **test_data.py** - Comprehensive Test Suite
Unit and integration tests for all components.

**Test Classes:**
- `TestDatasetDownloader` - Download functionality
- `TestNSLKDDPreprocessor` - Preprocessing operations
- `TestFlowGraphBuilder` - Graph construction
- `TestNetworkFlowDataset` - Dataset management
- `TestIntegration` - End-to-end pipeline

**Coverage:**
- Dataset download and validation
- Preprocessing output shapes
- Label conversion correctness
- Missing value handling
- Graph creation and statistics
- PyG Data object validity
- Edge construction correctness
- No self-loops verification
- Full pipeline integration

## Data Flow Diagram

```
Raw NSL-KDD Files
    ↓
[DatasetDownloader]
    ↓
[NSLKDDPreprocessor]
    ├─ Load CSV
    ├─ Handle missing values
    ├─ Encode categorical features
    ├─ Convert labels (binary)
    └─ Normalize features (StandardScaler)
    ↓
Feature Matrix X (N_samples × 41)
Labels y (N_samples,)
    ↓
[FlowGraphBuilder]
    ├─ Build edges using similarity
    │  ├─ kNN graph (k=5)
    │  └─ Similarity graph (threshold=0.8)
    ├─ Create PyG Data object
    └─ Log statistics
    ↓
PyG Graph (Data object)
    ├─ x: node features (N × 41)
    ├─ y: node labels (N,)
    ├─ edge_index: (2, E) edges
    └─ edge_attr: (E, 1) weights
    ↓
[NetworkFlowDataset]
    ├─ Save to disk
    ├─ Cache in memory
    └─ Provide PyG interface
    ↓
Ready for GNN Training
```

## Usage Examples

### 1. Generate Sample Data

```bash
python scripts/generate_sample_data.py --samples 1000 --output-dir data/raw/nsl-kdd
```

### 2. Download Real Dataset

```python
from data.download import DatasetDownloader

downloader = DatasetDownloader()
downloader.download_nsl_kdd()
```

### 3. Preprocess Data

```python
from data.preprocess import NSLKDDPreprocessor

preprocessor = NSLKDDPreprocessor()
df = preprocessor.load_data("data/raw/nsl-kdd/KDDTrain+.txt")
X, y = preprocessor.preprocess(df, fit=True)
preprocessor.save_preprocessed(X, y, "data/processed")
```

### 4. Build Flow-Centric Graph

```python
from data.graph_builder import FlowGraphBuilder

builder = FlowGraphBuilder()

# kNN graph (default)
graph = builder.build_graph(X, y, method="knn", k=5, metric="cosine")

# Similarity graph
graph = builder.build_graph(X, y, method="similarity", threshold=0.8)
```

### 5. Create PyG Dataset

```python
from data.dataset import NetworkFlowDataset

# Create dataset
dataset = NetworkFlowDataset.create_dataset(
    name="nsl-kdd",
    split="train",
    root="data/graphs"
)

# Access sample
graph = dataset[0]
print(graph)  # PyG Data object
```

### 6. Complete Pipeline

```bash
python scripts/verify_pipeline.py
```

## Running Tests

### All tests:
```bash
python -m pytest tests/test_data.py -v
```

### Specific test class:
```bash
python -m pytest tests/test_data.py::TestFlowGraphBuilder -v
```

### Integration tests only:
```bash
python -m pytest tests/test_data.py::TestIntegration -v
```

## Directory Structure

```
data/
├── raw/
│   └── nsl-kdd/
│       ├── KDDTrain+.txt          # Raw training data
│       └── KDDTest+.txt           # Raw test data
├── processed/
│   ├── X_preprocessed.npy         # Processed features
│   ├── y_preprocessed.npy         # Processed labels
│   └── preprocessor_state.pkl     # Scaler and encoders
└── graphs/
    ├── raw/                       # Raw NSL-KDD files (mirror)
    └── processed/
        ├── data_train.pt          # PyG graph
        ├── statistics_train.pkl
        ├── data_test.pt
        └── statistics_test.pkl

src/data/
├── __init__.py
├── download.py                    # DatasetDownloader
├── preprocess.py                  # NSLKDDPreprocessor
├── graph_builder.py               # FlowGraphBuilder
└── dataset.py                     # NetworkFlowDataset

tests/
└── test_data.py                   # Test suite

scripts/
├── generate_sample_data.py        # Generate synthetic data
└── verify_pipeline.py             # End-to-end verification
```

## Key Design Decisions

### 1. Flow-Centric Representation
- **Why:** Each flow carries semantic meaning (protocol, service, attack type)
- **Benefit:** GNNs learn relationships between flows, not individual packets
- **Alternative:** Host-centric (not used) - loses flow-level information

### 2. Binary Classification
- **Why:** Simplifies detection task (normal vs attack)
- **Benefit:** Enables binary cross-entropy loss, focus on anomaly detection
- **Future:** Can extend to multi-class for attack type classification

### 3. kNN Graph Default
- **Why:** Simple, effective, preserves flow similarity structure
- **k=5:** Empirically reasonable neighborhood size
- **Alternative:** Fully connected (computationally expensive) or random (low information)

### 4. StandardScaler Normalization
- **Why:** Assumes Gaussian-distributed features (reasonable for network metrics)
- **Benefit:** Zero mean, unit variance - ideal for neural networks
- **Alternative:** MinMaxScaler (if bounded features expected)

### 5. InMemoryDataset
- **Why:** NSL-KDD is relatively small (~300K samples)
- **Benefit:** Fast access, simple caching
- **Alternative:** IterableDataset (for very large data)

## Constraints and Limitations

✓ **Implemented:**
- NSL-KDD dataset download and preprocessing
- Flow-centric graph construction (kNN and similarity)
- PyTorch Geometric compatibility
- Comprehensive testing

✗ **NOT Implemented (Per Requirements):**
- CICIDS2017 dataset
- Attacks (adversarial perturbations)
- Defenses (adversarial training, etc.)
- SDN/Mininet/Ryu infrastructure

## Performance Considerations

- **Memory:** ~300MB for full NSL-KDD processed (300K samples × 41 features)
- **Processing:** ~2-5 minutes for full preprocessing on CPU
- **Graph Building:** ~1-2 minutes for kNN graph (cosine similarity)

## Error Handling

All components include:
- Input validation
- Type checking
- Comprehensive logging
- Graceful error messages
- Recovery mechanisms

## Future Extensions

Phase 2 (Models):
- GCN, GAT, GNN implementations
- Training/evaluation pipelines

Phase 3 (Attacks):
- FGSM, PGD, Nettack, MetaAttack

Phase 4 (Defenses):
- Adversarial training
- Jaccard preprocessing
- GNNGuard

## References

- NSL-KDD Dataset: https://www.unb.ca/cic/datasets/nsl-kdd.html
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Scikit-learn: https://scikit-learn.org/

## Author Notes

This implementation prioritizes:
1. **Clarity:** Well-documented, modular code
2. **Research-Oriented:** Suitable for academic use and publication
3. **Extensibility:** Easy to add new datasets, graph types, models
4. **Robustness:** Comprehensive error handling and validation
5. **Reproducibility:** Deterministic preprocessing with proper logging
