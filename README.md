# Flow-Centric GNN-based Network Intrusion Detection: Robustness and Graph Construction Analysis

This repository contains the experimental framework developed for the dissertation **“Flow-Centric GNN-based Network Intrusion Detection: Robustness and Graph Construction Analysis”**.

The project investigates the robustness of flow-centric Graph Neural Network (GNN)-based Network Intrusion Detection Systems (NIDS) under feature-level adversarial perturbations and random structural perturbations. It also analyses neighbourhood stability and the influence of the k-nearest-neighbour (k-NN) graph construction parameter \(k\).

## Overview

Network flows are represented as graph nodes. For each fixed-size traffic window, a k-NN graph is constructed from flow-feature similarity using cosine similarity. The resulting connections are represented bidirectionally.

Two GNN architectures are evaluated:

- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)

Both models use a single message-passing layer followed by a linear classification layer.

The experimental evaluation covers:

- baseline intrusion-detection performance;
- feature-level adversarial robustness using FGSM and PGD;
- robustness to random edge removal and edge addition;
- neighbourhood stability after feature perturbation and graph reconstruction;
- sensitivity to the k-NN neighbourhood size.

## Experimental Configuration

| Component | Configuration |
|---|---|
| Graph representation | Flow-centric |
| Window definition | Fixed number of flows |
| Window size | 1000 |
| Graph construction | k-NN |
| Similarity metric | Cosine similarity |
| Default \(k\) | 5 |
| Edge representation | Bidirectional |
| Models | GCN, GAT |
| Message-passing layers | 1 |
| Primary classification metric | F1-score |
| Additional metrics | Accuracy, Precision, Recall, ROC-AUC |
| Main training seeds | 42, 43, 44, 45, 46 |
| Deterministic execution | Enabled where supported |

## Datasets

The evaluation uses NSL-KDD and four CICIDS2017 scenarios.

| Dataset / scenario | Records | Features | Benign | Malicious | Malicious (%) |
|---|---:|---:|---:|---:|---:|
| NSL-KDD | 148,517 | 41 | 77,054 | 71,463 | 48.12 |
| CICIDS2017 DDoS | 225,745 | 78 | 97,718 | 128,027 | 56.71 |
| CICIDS2017 PortScan | 286,467 | 78 | 127,537 | 158,930 | 55.48 |
| CICIDS2017 Patator | 445,909 | 78 | 432,074 | 13,835 | 3.10 |
| CICIDS2017 Selected | 958,121 | 78 | 657,329 | 300,792 | 31.39 |

### Split policy

**NSL-KDD**

- the official `KDDTest+.txt` partition is preserved as the final test set;
- `KDDTrain+.txt` is split into 85% training and 15% validation data using stratified sampling.

**CICIDS2017**

- each scenario uses a stratified 70% / 15% / 15% training, validation and test split.

Preprocessing parameters are fitted exclusively on the training data and reused for validation and test data.

## Adversarial and Robustness Evaluation

### Feature-level attacks

FGSM and PGD perturb only malicious nodes, identified using ground-truth labels.

The evaluated perturbation budgets are:

```text
epsilon = {0.01, 0.03, 0.05, 0.10}
```

PGD uses:

```text
steps        = 20
alpha        = 2.5 * epsilon / steps
random_start = True
```

During classification under feature-level attacks, the original graph connectivity is preserved. Graph reconstruction from perturbed features is performed separately for the neighbourhood-stability analysis.

### Random structural perturbations

Structural robustness is evaluated using random:

- edge removal;
- edge addition.

The perturbations target logical graph connections involving malicious nodes. Reciprocal directed edges are treated as one logical connection when determining the perturbation budget.

The evaluated perturbation rates are:

```text
r = {0.05, 0.10, 0.20, 0.30}
```

For the main structural experiments, results are aggregated across five training seeds and five perturbation seeds.

These experiments evaluate random structural perturbations and should not be interpreted as optimised or gradient-based structural adversarial attacks.

## Neighbourhood Stability

Feature perturbations may indirectly modify graph connectivity when the k-NN graph is reconstructed from the perturbed feature space.

Neighbourhood change is quantified using the **Neighbor Churn Rate (NCR)**.

Two scopes are reported:

- **Malicious-node NCR (`NCR_mal`)** — computed only over malicious nodes directly targeted by the feature attacks. This is the primary neighbourhood-stability measure.
- **Global NCR (`NCR_global`)** — computed over all nodes and retained as a complementary whole-graph measure.

For each experimental run, NCR is computed independently for each test graph. `NCR_mal` is averaged only across graphs containing at least one malicious node.

Because global NCR includes unperturbed benign nodes, it can be substantially lower than malicious-node NCR when malicious traffic represents only a small fraction of the graph.

## k-NN Sensitivity Analysis

The sensitivity analysis evaluates:

```text
k = {3, 5, 10, 20, 50}
```

For each value of \(k\), the analysis considers:

- baseline F1-score;
- FGSM robustness at `epsilon = 0.10`;
- PGD robustness at `epsilon = 0.10`;
- malicious-node NCR;
- global NCR.

The sensitivity analysis uses training seeds:

```text
42, 43, 44
```

For PGD, the attack seed is matched to the corresponding training seed:

```text
(42, 42), (43, 43), (44, 44)
```

Random structural perturbations are evaluated only for the default graph configuration (`k = 5`) and are not included in the k-sensitivity analysis.

## Main Results

Baseline test F1-scores for the default graph configuration (`k = 5`) are:

| Scenario | GCN | GAT |
|---|---:|---:|
| NSL-KDD | 0.7628 ± 0.0043 | 0.7761 ± 0.0078 |
| DDoS | 0.9984 ± 0.0002 | 0.9987 ± 0.0001 |
| PortScan | 0.9945 ± 0.0001 | 0.9964 ± 0.0003 |
| Patator | 0.8725 ± 0.0013 | 0.8519 ± 0.0160 |
| Selected | 0.9767 ± 0.0005 | 0.9743 ± 0.0007 |

The main experimental observations are:

- strong baseline performance does not necessarily imply adversarial robustness;
- feature-level performance degradation generally increases with the perturbation budget;
- neither GCN nor GAT is consistently more robust across all evaluated scenarios;
- random edge addition has a greater impact than random edge removal in the evaluated structural experiments;
- feature perturbations can modify k-NN connectivity when the graph is reconstructed from perturbed features;
- global NCR can substantially underestimate neighbourhood change relative to malicious-node NCR when malicious flows are sparse;
- the influence of \(k\) varies across datasets, models and attacks, with no single value consistently providing better baseline performance, robustness and neighbourhood stability.

## Repository Structure

```text
gnn-adversarial-nids/
├── configs/
│   └── base_config.yaml
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── src/
│   ├── analysis/
│   │   └── neighbor_churn.py
│   ├── attacks/
│   │   ├── fgsm.py
│   │   ├── pgd.py
│   │   └── structural.py
│   ├── data/
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   ├── graph_builder.py
│   │   ├── dataset.py
│   │   └── splits.py
│   ├── models/
│   │   ├── base.py
│   │   ├── gcn.py
│   │   └── gat.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── utils/
│       └── metrics.py
├── experiments/
│   ├── 01_baseline_training.py
│   ├── 02_feature_attacks.py
│   ├── 03_structural_attacks.py
│   └── 04_k_sensitivity.py
├── tests/
├── data/
│   ├── raw/
│   ├── splits/
│   └── graphs/
├── results/
├── README.md
├── .gitignore
└── setup.py
```

## Setup

### Docker

Prerequisites:

- Docker
- Docker Compose

Clone the repository:

```bash
git clone https://github.com/jcaeiro02/gnn-adversarial-nids.git
cd gnn-adversarial-nids
```

Build the experimental environment:

```bash
docker compose -f docker/docker-compose.yml build
```

### Manual installation

```bash
pip install -r docker/requirements.txt
pip install -e .
```

The final dissertation experiments were executed using CPU-based computation inside the Docker environment.

## Data Preparation

Place the required raw datasets under:

```text
data/raw/
├── nsl-kdd/
└── cicids2017/
```

CICIDS2017 CSV files are not downloaded automatically and must be placed manually under:

```text
data/raw/cicids2017/
```

Processed graph caches are isolated by dataset and neighbourhood size under `data/graphs/`. Cache metadata records the dataset, split, window size, \(k\), graph-construction method and similarity metric to prevent incompatible graph configurations from being reused.

## Running the Experiments

The commands below show representative runs for a single dataset, model and seed.

### 1. Baseline training

```bash
docker compose -f docker/docker-compose.yml run --rm gnn-experiments \
  python experiments/01_baseline_training.py \
  --dataset nsl-kdd \
  --model gcn \
  --k 5 \
  --seed 42 \
  --deterministic
```

The best validation checkpoint is stored under `results/runs/<run_id>/best_checkpoint.pt`.

### 2. FGSM evaluation

```bash
docker compose -f docker/docker-compose.yml run --rm gnn-experiments \
  python experiments/02_feature_attacks.py \
  --dataset nsl-kdd \
  --model gcn \
  --checkpoint results/runs/<run_id>/best_checkpoint.pt \
  --k 5 \
  --training-seed 42 \
  --attack-seed 42 \
  --attacks fgsm \
  --epsilons 0.01 0.03 0.05 0.10 \
  --deterministic
```

### 3. PGD evaluation

```bash
docker compose -f docker/docker-compose.yml run --rm gnn-experiments \
  python experiments/02_feature_attacks.py \
  --dataset nsl-kdd \
  --model gcn \
  --checkpoint results/runs/<run_id>/best_checkpoint.pt \
  --k 5 \
  --training-seed 42 \
  --attack-seed 42 \
  --attacks pgd \
  --epsilons 0.01 0.03 0.05 0.10 \
  --steps 20 \
  --deterministic
```

If `--alpha` is omitted, the implementation uses:

```text
alpha = 2.5 * epsilon / steps
```

### 4. Random structural perturbations

```bash
docker compose -f docker/docker-compose.yml run --rm gnn-experiments \
  python experiments/03_structural_attacks.py \
  --dataset nsl-kdd \
  --model gcn \
  --checkpoint results/runs/<run_id>/best_checkpoint.pt \
  --k 5 \
  --training-seed 42 \
  --perturbation-seed 42 \
  --attacks edge_removal edge_addition \
  --rates 0.05 0.10 0.20 0.30 \
  --deterministic
```

### 5. k-NN sensitivity analysis

```bash
docker compose -f docker/docker-compose.yml run --rm gnn-experiments \
  python experiments/04_k_sensitivity.py \
  --dataset nsl-kdd \
  --model gcn \
  --k-values 3 5 10 20 50 \
  --attacks fgsm pgd \
  --epsilons 0.10 \
  --training-seed 42 \
  --attack-seed 42 \
  --steps 20 \
  --deterministic
```

Repeat the sensitivity run for training seeds 43 and 44, matching the attack seed to the training seed.

## Reproducibility

The framework includes:

- persistent dataset splits;
- dataset- and k-specific processed graph caches;
- preprocessing fitted only on training data;
- explicit training, attack and perturbation seeds;
- deterministic PyTorch execution where supported;
- checkpoint metadata validation for \(k\) and training seed;
- saved JSON/CSV experiment outputs;
- per-run configuration metadata.

The main baseline, feature-attack and structural experiments use five training seeds (`42–46`). PGD and random structural perturbations additionally use five attack/perturbation seeds (`42–46`). The k-sensitivity analysis uses three training seeds (`42–44`).

## Limitations

The experimental scope should be interpreted within the following constraints:

- evaluation is limited to NSL-KDD and selected CICIDS2017 traffic scenarios;
- only GCN and GAT architectures are evaluated;
- both models use a single message-passing layer;
- feature-level attacks operate in standardised feature space and do not enforce feature-specific semantic or protocol constraints;
- structural robustness is evaluated using random edge addition and removal rather than targeted or adaptive structural adversarial attacks;
- neighbourhood churn measures graph changes after reconstruction but does not isolate their causal contribution to classification performance.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this repository in research, please cite:

```bibtex
@misc{caeiro2026gnn,
  title     = {Flow-Centric GNN-based Network Intrusion Detection: Robustness and Graph Construction Analysis},
  author    = {João André Rodrigues Caeiro},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/jcaeiro02/gnn-adversarial-nids}
}
```

## Acknowledgements

This project uses PyTorch and PyTorch Geometric and evaluates publicly available network intrusion detection datasets.
