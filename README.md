# Financial Sequence Modeling for Risk Signal Detection
# Project Overview
This project focuses on modeling sequential financial data to identify patterns that can be used as early warning signals for risk-related events. In financial systems, behavior often unfolds over time, and capturing these temporal patterns is critical for making informed decisions.

The goal of this project is to demonstrate how sequence-based modeling techniques can be used to extract meaningful insights from financial data and support data-driven decision-making.

---

# Problem Statement
Financial institutions rely on timely and accurate signals to assess risk and guide decision-making. However, many traditional models fail to fully capture the temporal dependencies present in real-world data.

This project addresses this gap by:
- Modeling sequential patterns in financial data  
- Identifying signals that may indicate shifts in behaviour  
- Providing a foundation for predictive risk modeling systems

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
- Implementing sequence-based predictive models  
- Training the model to learn temporal dependencies in the data  

## Project Overview
- **Task**: Given a sequence of past market state vectors, predict the next state vector.
- **Data**: Multiple independent sequences, each exactly 1000 steps long.
  - First 100 steps are warm-up (used for context but not scored).
  - Scoring is based on predictions for steps 100–998.
- **Evaluation Metric**: Mean R² (coefficient of determination) across all features (higher is better).
- **Key Constraint**: Model state must be reset between independent sequences.
- **Approach**: Sliding-window GRU model that processes recent history to forecast the next step.

Local validation achieves **~0.369 mean R²** (results may vary on hidden test set).

---

## Key Results
- The model successfully captures sequential patterns in financial data  
- Demonstrates predictive capability in identifying future states based on past behaviour  
- Provides a framework for extending sequence modeling into risk prediction tasks  
## Business Relevance
Although this project is based on simulated/market data, the methodology is directly applicable to real-world financial systems such as:
- **Credit Risk Modeling**: Identifying behavioral patterns that precede default  
- **Fraud Detection**: Detecting unusual sequences of transactions  
- **Customer Behavior Analysis**: Understanding financial activity over time  
- **Decision Support Systems**: Enhancing data-driven decision-making in financial institutions

This approach highlights how temporal modeling can complement traditional statistical methods in high-stakes environments like banking and credit bureaus.

---

## Tools & Technologies
- Python, Jupyter Notebook
- Data manipulation libraries  
- Machine learning / sequence modeling techniques  

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

