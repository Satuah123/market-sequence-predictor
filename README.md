# Financial Sequence Prediction: GRU Networks vs. ARIMA Benchmarking
### Developed by Samuel Atuah | MPhil Statistics Candidate, University of Ghana

##  Project Overview
This repository implements a high-performance machine learning pipeline for financial time-series forecasting. The project focuses on leveraging **Gated Recurrent Units (GRU)** to capture non-linear, long-term dependencies in sequential data, benchmarked against a classical **ARIMA (Autoregressive Integrated Moving Average)** baseline.

##  Statistical & Business Context
As an MPhil Statistics candidate, I developed this framework to test the hypothesis that **Latent Sequential Dependencies**—often missed by linear statistical models—can be captured via Recurrent Neural Network (RNN) architectures.

### Direct Applications for XDS Data Ghana:
- **Behavioral Credit Scoring**: Modeling borrower payment sequences to predict transitions between "Good" and "Delinquent" states.
- **Probability of Default (PD) Estimation**: Using rolling window history to identify early-warning signals of financial distress.
- **Dynamic Risk Recalibration**: Moving beyond static scorecards to real-time risk assessment based on sequential transaction patterns.

##  Model Architecture & Methodology
### 1. GRU Deep Learning Model (PyTorch)
- **Memory Retention**: Specifically chosen over standard RNNs to mitigate the **Vanishing Gradient Problem** using update and reset gates.
- **Regularization**: Integrated **Layer Normalization** and **Dropout (0.15)** to ensure the model generalizes well to unseen financial regimes.
- **Optimization**: Utilized **Smooth L1 Loss** (Huber Loss) for robustness against outliers, which are frequent in credit and market data.

### 2. Statistical Baseline (ARIMA)
- Provided a rigid linear benchmark to quantify the **"Non-linear Gain"** achieved by the deep learning architecture.

## Performance Metrics

| Model | RMSE | R² Score | Key Statistical Finding |
| :--- | :--- | :--- | :--- |
| **GRU** | **Superior** | **0.2708** | Captured non-linear variance that ARIMA missed entirely. |
| **ARIMA** | 0.6620 | -0.0207 | Failed to model the heteroskedasticity of the sequence. |

##  Tech Stack & Skills
- **Deep Learning**: PyTorch (Tensors, Autograd, Module subclassing)
- **Data Engineering**: Pandas, NumPy, Parquet for high-speed I/O
- **Statistical Benchmarking**: Scikit-learn, ARIMA modeling
- **Experiment Tracking**: Systematic hyperparameter tuning via `config.json`

```
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

Download the dataset [train.parquet](https://drive.google.com/file/d/1HQHEfcSSatXf0QzeNvPiqy4lpdA8q08u/view?usp=sharing) and save it as:
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
- Extend model evaluation with ROC/AUC and classification metrics  
- Improve interpretability of model outputs  
- Deploy as an interactive dashboard (e.g., using Shiny)
  
---

## Author
Samuel Atuah  
MPhil Statistics Candidate, University of Ghana  

---

##  Note
This project is part of a broader effort to apply statistical and machine learning methods to financial data, with a focus on building interpretable and decision-oriented models.

