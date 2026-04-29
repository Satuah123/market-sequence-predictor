# Market Sequence Prediction using GRU Networks and ARIMA Benchmarking
## Project Overview
This project develops a deep learning forecasting pipeline for sequential market behavior prediction using Gated Recurrent Units (GRU), with ARIMA implemented as a classical statistical baseline.

The objective is to evaluate whether deep learning architectures outperform traditional time-series methods in capturing sequential dependencies in financial-style datasets.

## Business Relevance

Although developed on sequence market data, this workflow is transferable to:

- Credit scoring and default risk prediction
- Customer behavior forecasting
- Fraud/anomaly detection
- Financial risk analytics
- Banking transaction sequence modeling

## Model Performance

| Model | RMSE | R² Score |
|------|------|---------|
| GRU | 0.53  | 0.2708 |
| ARIMA | 0.6620 | -0.0207 |

### Key Insight
GRU significantly outperformed ARIMA, demonstrating stronger capacity to capture nonlinear sequential dependencies.

## Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Jupyter Notebook

## Skills Demonstrated

- Time Series Forecasting
- Deep Learning (GRU)
- Statistical Benchmarking (ARIMA)
- Model Evaluation
- Feature Engineering
- Reproducible ML Pipelines
- Financial Data Modeling
---

# Objective
To investigate the predictive performance of classical vs deep learning approaches on high-dimensional sequential data and understand their limitations in real-world forecasting tasks.
---
##  Dataset
- Format: `.parquet`
- Structure:
  - `seq_ix`: sequence identifier
  - `step_in_seq`: time step index
  - `need_prediction`: target indicator
  - `0–31`: multivariate feature columns (32 features total)

Each sequence represents a multivariate time series instance.

---

# Methodology
The project follows a structured data science workflow:
### 1. Data Preparation
- Cleaning and structuring sequential financial data  
- Handling missing values and inconsistencies  
- Transforming data into a format suitable for sequence modeling

### 2. Feature Engineering
- Extracting time-dependent features  
- Capturing trends and transitions in the data 
### 3. Model Development
####  ARIMA (Baseline Model)
- Applied independently to each feature
- Order: (5,1,0)
- Evaluated using RMSE and R²
####  GRU (Deep Learning Model)
- Multi-layer GRU network
- Captures temporal dependencies across features
- Uses sliding window input sequences
- Optimized using Smooth L1 Loss
---

## Evaluation Metrics

- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

---


---
## Key Insights

- ARIMA struggles with multivariate and nonlinear sequential patterns
- Performance degrades significantly across all features
- GRU model captures hidden temporal relationships in the data
- Deep learning significantly outperforms classical statistical methods in this setting

---

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

