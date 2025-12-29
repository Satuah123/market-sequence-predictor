# Sequence Market State Prediction

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

This repository contains my personal solution for a private machine learning competition focused on **next-step prediction in multivariate time series** (market state sequences). The goal is to predict the next state vector in a sequence of high-dimensional, anonymized numeric features representing evolving market conditions.

The solution uses a **GRU (Gated Recurrent Unit)** neural network implemented in PyTorch, trained on sliding windows of historical states. It includes full training pipeline, inference code compatible with the competition's submission format, local scoring utilities, and configuration management.

## Project Overview

- **Task**: Given a sequence of past market state vectors, predict the next state vector.
- **Data**: Multiple independent sequences, each exactly 1000 steps long.
  - First 100 steps are warm-up (used for context but not scored).
  - Scoring is based on predictions for steps 100–998.
- **Evaluation Metric**: Mean R² (coefficient of determination) across all features (higher is better).
- **Key Constraint**: Model state must be reset between independent sequences.
- **Approach**: Sliding-window GRU model that processes recent history to forecast the next step.

Local validation achieves **~0.369 mean R²** (results may vary on hidden test set).

## Repository Structure

```
.
├── datasets/                   # Place train.parquet here (not included in repo)
├── examples/                   # Official baseline examples (if any)
├── config.json                 # Training and path configuration
├── Train.ipynb                 # Full training notebook with data loading, model training, validation
├── solution.py                 # Inference model compatible with competition submission format
├── utils.py                    # DataPoint class, local ScorerStepByStep for evaluation
├── model_checkpoint.pth        # Trained model weights (generated after training)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Satuah123/market-sequence-predictor.git
cd market-sequence-predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

> Note: PyTorch installation may vary by your system/CUDA setup. See [pytorch.org](https://pytorch.org/get-started/locally/) for exact command.

## Usage

### 1. Prepare the Data

Download the dataset [here](http://google.drive.com) and save it as:
```
datasets/train.parquet
```

This is a single Parquet file containing all sequences with columns:
- `seq_ix`, `step_in_seq`, `need_prediction`, and N feature columns.

### 2. Train the Model

Run the training notebook or script:

```bash
jupyter notebook Train.ipynb
```

Or convert to script and run:
```bash
jupyter nbconvert --to script Train.ipynb
python Train.py
```

The training process:
- Splits sequences into train/validation (80/20 by sequence ID)
- Creates sliding window samples
- Trains GRU with Smooth L1 loss
- Uses early stopping via ReduceLROnPlateau on validation loss
- Saves best model to `model_checkpoint.pth` (path configurable in `config.json`)

### 3. Local Evaluation

After training, the script automatically evaluates on the validation set using the official-style scorer and prints mean R².

You can also run scoring manually:
```python
from utils import ScorerStepByStep
from solution import PredictionModel
import torch

model = PredictionModel(device="cuda" if torch.cuda.is_available() else "cpu")
scorer = ScorerStepByStep("datasets/train.parquet")  # or validation split
results = scorer.score(model)
print(f"Mean R²: {results['mean_r2']:.6f}")
```

### 4. Generate Submission

The `solution.py` file contains the `PredictionModel` class required for submission.

To package for submission:
```bash
# From the repository root
zip -r submission.zip solution.py model_checkpoint.pth config.json
```

> Ensure `solution.py` is at the root of the zip (no subfolders).

## Configuration (`config.json`)

Key parameters you can tune:
```json
{
  "seed": 42,
  "device": "cuda",
  "paths": {
    "dataset": "datasets/train.parquet",
    "model_checkpoint": "model_checkpoint.pth"
  },
  "model": {
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.15
  },
  "training": {
    "window_size": 10,
    "warmup": 100,
    "batch_size": 256,
    "epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 1e-5
  },
  "regularization": {
    "gradient_clip": 1.0,
    "scheduler_patience": 10,
    "scheduler_factor": 0.5
  }
}
```

## Model Details

- **Architecture**: Multi-layer GRU → LayerNorm → Dropout → Linear
- **Input**: Last `window_size` states (default 10)
- **Output**: Predicted next state vector (same dimension as input)
- **Loss**: Smooth L1 (robust to outliers)
- **Optimizer**: Adam with weight decay
- **Inference**: Maintains rolling history per sequence, resets on new `seq_ix`

## Future Improvements (Ideas)

- Larger window sizes
- Bidirectional GRU or Transformer-based models
- Feature engineering / normalization per sequence
- Ensemble of multiple window sizes or architectures
- Residual connections or denser heads

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

Feel free to open issues or submit pull requests if you have improvements or questions!
