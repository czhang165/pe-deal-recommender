# PE Deal Recommender System

A production-grade two-stage recommender system for matching private equity investors with investment opportunities. Built with PyTorch Lightning, with optional Ray support for distributed training and Kubernetes deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Two-Stage Recommendation                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Retrieval (Two-Tower Model)                           │
│  ┌─────────────────┐      ┌─────────────────┐                   │
│  │  Investor Tower │      │   Deal Tower    │                   │
│  │  - ID Embedding │      │  - ID Embedding │                   │
│  │  - Type/Region  │      │  - Sector/Stage │                   │
│  │  - Risk Profile │      │  - Financials   │                   │
│  │  - MLP Layers   │      │  - MLP Layers   │                   │
│  └────────┬────────┘      └────────┬────────┘                   │
│           │                        │                             │
│           └──── Dot Product ───────┘                             │
│                      │                                           │
│           ┌──────────▼──────────┐                               │
│           │    FAISS Index      │  → Top-100 Candidates         │
│           └─────────────────────┘                               │
│                                                                  │
│  Stage 2: Re-ranking (Deep Ranking Model)                       │
│  ┌─────────────────────────────────────────┐                    │
│  │  Concatenated Features → Deep MLP → Score│                   │
│  │  Trained with Pairwise RankNet Loss      │                   │
│  └─────────────────────────────────────────┘                    │
│                      │                                           │
│           ┌──────────▼──────────┐                               │
│           │   Final Top-10      │                               │
│           └─────────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Two-Tower Model**: Efficient candidate retrieval using learned embeddings
- **Deep Ranking Model**: Precise re-ranking with pairwise RankNet loss
- **FAISS Integration**: Fast approximate nearest neighbor search
- **Configurable**: Pydantic-based configuration with YAML/ENV support
- **Ray Ready**: Optional distributed training with Ray Train
- **Kubernetes Ready**: Helm charts and manifests for K8s deployment

## Project Structure

```
pe-deal-recommender/
├── src/
│   ├── config/           # Configuration management
│   │   ├── settings.py   # Pydantic settings classes
│   │   └── config.yaml   # Default configuration
│   ├── data/             # Data loading and preprocessing
│   │   ├── datasets.py   # PyTorch Dataset classes
│   │   ├── feature_encoder.py  # Feature encoding
│   │   └── data_loader.py      # Data loading utilities
│   ├── models/           # Model architectures
│   │   ├── towers.py     # Investor/Deal tower modules
│   │   ├── two_tower.py  # Two-Tower retrieval model
│   │   └── deep_ranker.py  # Deep Ranking model
│   ├── training/         # Training pipelines
│   │   ├── local_trainer.py  # PyTorch Lightning training
│   │   ├── ray_trainer.py    # Ray distributed training
│   │   └── evaluation.py     # Evaluation metrics
│   ├── inference/        # Inference pipelines
│   │   └── batch_inference.py  # FAISS + batch scoring
│   └── utils/            # Utilities
├── scripts/              # CLI scripts
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── k8s/                  # Kubernetes manifests
├── docker/               # Docker configurations
├── tests/                # Unit tests
└── notebooks/            # Jupyter notebooks
```

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/czhang165/pe-deal-recommender.git
cd pe-deal-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### With Ray Support (for distributed training)

```bash
pip install -e ".[ray]"
```

### With All Optional Dependencies

```bash
pip install -e ".[all]"
```

## Quick Start

### 1. Generate Synthetic Data

First, generate the synthetic investor-deal interaction data:

```bash
# Run the data generation notebook or script
# This creates files in data/ directory:
# - enhanced_interactions.csv
# - investor_features.csv  
# - deal_features.csv
```

Or use Python:

```python
# Use your existing data_generation.ipynb or create data programmatically
import pandas as pd
import numpy as np

# ... data generation code ...
```

### 2. Train Models

```bash
# Train both models (Two-Tower and Deep Ranking)
python scripts/train.py

# Train with custom epochs
python scripts/train.py --epochs 50

# Train only Two-Tower
python scripts/train.py --model two_tower
```

### 3. Evaluate Models

```bash
# Run evaluation
python scripts/evaluate.py

# Evaluate with different K
python scripts/evaluate.py --k 20 --retrieval-k 200
```

### 4. Use for Inference

```python
from src import load_data, load_inference_pipeline
from src.config import init_settings

# Initialize
settings = init_settings()
data_bundle = load_data(settings)

# Load trained models
pipeline = load_inference_pipeline(
    two_tower_checkpoint="models/checkpoints/two-tower-best.ckpt",
    deep_ranking_checkpoint="models/checkpoints/deep-ranking-best.ckpt",
    encoder_path="models/feature_encoder.pkl",
    investor_features=data_bundle.investor_features,
    deal_features=data_bundle.deal_features,
)

# Build FAISS index
pipeline.build_deal_index(data_bundle.all_deal_ids)

# Get recommendations
result = pipeline.recommend(
    investor_id=42,
    k=10,
    retrieval_k=100,
    exclude_deal_ids={1, 2, 3}  # Already seen deals
)

print(f"Recommendations for investor {result.investor_id}:")
for deal_id, score in zip(result.recommended_deal_ids, result.scores):
    print(f"  Deal {deal_id}: score={score:.4f}")
```

## Configuration

Configuration can be provided via:

1. **YAML file**: `src/config/config.yaml`
2. **Environment variables**: Prefixed with `RECSYS_`
3. **Python code**: Direct initialization

### Example: Custom Configuration

```yaml
# config/prod.yaml
environment: prod
storage_backend: s3
compute_backend: ray

data:
  n_investors: 1000
  n_deals: 50000

training:
  batch_size: 256
  max_epochs: 50
  
ray_training:
  num_workers: 4
  use_gpu: true
```

```bash
# Use custom config
python scripts/train.py --config config/prod.yaml
```

### Environment Variables

```bash
export RECSYS_ENVIRONMENT=prod
export RECSYS_DATA_DIR=/data/recommender
export RECSYS_STORAGE_BACKEND=s3
export RECSYS_S3_BUCKET=my-bucket
```

## Evaluation Metrics

The system evaluates models using standard ranking metrics:

| Metric | Description |
|--------|-------------|
| **Hit@K** | Whether any relevant item is in top-K |
| **NDCG@K** | Normalized Discounted Cumulative Gain |
| **MRR** | Mean Reciprocal Rank of first relevant item |
| **Recall@K** | Fraction of relevant items in top-K |
| **Precision@K** | Fraction of top-K that are relevant |

### Typical Results

```
Model                     Hit@K      NDCG@K     MRR       
-------------------------------------------------------
Two-Tower                 0.0826     0.0121     0.0250    
Deep Ranking              0.0496     0.0060     0.0141    
Two-Stage Pipeline        0.0826     0.0112     0.0250    
```

## Distributed Training with Ray

For larger datasets, use Ray for distributed training:

```python
from src.training import init_ray, train_two_tower_distributed

# Initialize Ray cluster
init_ray()

# Train with distributed data
result = train_two_tower_distributed(
    train_dataset_path="s3://bucket/train.parquet",
    val_dataset_path="s3://bucket/val.parquet",
    n_investors=1000,
    n_deals=50000,
    feature_dims=feature_dims,
)
```

## Kubernetes Deployment

See `k8s/` directory for Kubernetes manifests:

```bash
# Apply base configuration
kubectl apply -k k8s/base/

# Or use Helm
helm install pe-recommender k8s/helm/
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

### Type Checking

```bash
mypy src/
```

## License

GNU General Public License v3 - see LICENSE file for details.

## Citation

```bibtex
@software{pe_deal_recommender,
  title = {PE Deal Recommender System Distributed Production Level},
  author = {czhang165},
  year = {2026},
  url = {https://github.com/czhang165/pe-deal-recommender}
}
```
