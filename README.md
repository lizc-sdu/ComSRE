# ComSRE

A PyTorch implementation for Multi-View Urban Region Embedding via Commonality-Specificity Disentanglement.

## Project Structure

### Core Files

- **`main.py`** - Entry point for training the ComSRE model. Handles argument parsing, device configuration, and orchestrates the training process.

- **`model.py`** - Contains the main ComSRE model implementation:
  - `ComSRE` class: Main model with multi-view representation learning, contrastive alignment, orthogonality loss

- **`configure.py`** - Configuration management for different cities (Beijing, Xi'an, Chengdu) including:
  - Model architecture parameters
  - Training hyperparameters
  - Dataset paths and city-specific settings

- **`data_utils.py`** - Data loading utilities for region data including POI features, source/destination matrices.

- **`tasks.py`** - Evaluation utilities for downstream tasks:
  - Regression prediction with cross-validation
  - Performance metrics calculation (MAE, RMSE, R²)

### Dataset Structure

```
dataset/
├── bj/     # Beijing data
├── xa/     # Xi'an data  
└── cd/     # Chengdu data
```

Each city directory contains:
- POI feature matrices
- Source/destination flow matrices
- Indicators (CO₂, GDP, population)

## Usage

```bash
# Train on Beijing data
python main.py --city bj --device 0

# Train on Xi'an data  
python main.py --city xa --device cpu

# Train on Chengdu data
python main.py --city cd --device 0
```
