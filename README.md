# DB-Flow: Dual-Bridge Flow Matching for Tabular Data Imputation

Anonymous code submission for ICML 2026.

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 2.0
- POT (Python Optimal Transport)
- NumPy, Pandas, Scikit-learn

## Project Structure

```
├── train.py              # Main training entry
├── configs/              # Configuration files
├── scripts/              # Benchmark and evaluation scripts
└── src/
    ├── data/             # Data loading utilities
    ├── engine/           # Training loop
    ├── losses/           # Loss functions
    ├── metrics/          # Evaluation metrics
    ├── models/           # Model architectures
    ├── ot/               # Optimal transport modules
    ├── theory/           # MNAR detection theory
    └── utils/            # Utility functions
```

## Quick Start

### Training

```bash
python train.py --config configs/sbflow.yaml
```

### Benchmark

```bash
python scripts/run_benchmark.py --config configs/sbflow.yaml --dataset bean --missing_rate 0.3
```

## Configuration

Key parameters in config files:

- `model.hidden_dim`: Hidden dimension of the network
- `model.n_layers`: Number of transformer layers
- `training.epochs`: Training epochs
- `training.lr`: Learning rate
- `ot.use_ot`: Enable optimal transport coupling
- `mnar.detect`: Enable MNAR detection

## Data

Place datasets in `./data/` directory. Supported formats: CSV, NumPy arrays.
