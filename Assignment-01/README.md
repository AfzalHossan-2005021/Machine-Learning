# Medical Student Diabetes Prediction

A comprehensive machine learning project for predicting diabetes risk in medical students using deep neural networks. This project demonstrates the complete ML pipeline: data preparation, exploratory analysis, feature engineering, model training, and evaluation.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)

## Overview

This project implements a binary classification system to predict diabetes status in medical students using health and demographic features. The solution employs multiple neural network architectures with advanced techniques such as SMOTE for handling class imbalance, stratified train-validation-test splits, and early stopping for robust model training.

### Key Features

- **Data Preprocessing**: Comprehensive cleaning, imputation, and feature engineering
- **Multiple Architectures**: Four distinct neural network designs with varying complexity
- **Regularization**: Batch normalization and dropout for improved generalization
- **Class Balancing**: SMOTE-based oversampling for imbalanced datasets
- **Evaluation Metrics**: Accuracy, precision, F1-score, and AUROC with visualization

## Project Structure

```
.
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── data/
│   ├── medical_students_diabetes_dataset.csv  # Original dataset
│   ├── train_set.csv                  # Training set (70% with SMOTE)
│   └── test_set.csv                   # Test set (15%)
├── src/
│   ├── 1_prepare_data.ipynb          # Data preparation and splitting
│   ├── 2_train_model.ipynb           # Model training and validation
│   └── 3_test_model.ipynb            # Model evaluation on test set
├── models/
│   ├── best_model.pth                # Best model from validation phase
│   └── final_model.pth               # Final model trained on full data
└── utils/
    ├── preprocessing_data.pkl        # Preprocessors and encoders
    └── hyperparameters.pkl           # Model hyperparameters
```

## Dataset

The project uses the **Medical Students Diabetes Dataset**, containing health indicators and demographic information for medical students.

### Dataset Characteristics

- **Dimensions**: Multiple features including physiological and demographic attributes
- **Target Variable**: Binary classification (Diabetes: Yes/No)
- **Data Quality**: Handles missing values, duplicates, and class imbalance
- **Split Ratio**: 70% training, 15% validation, 15% testing (stratified)

### Data Processing Steps

1. **Cleaning**: Removal of missing values in target column, duplicate row elimination
2. **Imputation**: 
   - Iterative imputation for correlated features (Height, Weight, BMI)
   - Mean imputation for numeric features
   - Mode imputation for categorical features
   - Balanced random imputation for balanced columns
3. **Encoding**: One-hot encoding for categorical variables
4. **Scaling**: StandardScaler normalization for numeric features (excluding one-hot encoded features)
5. **Feature Selection**: Top 10 features selected based on correlation with target
6. **Resampling**: SMOTE applied to training data to handle class imbalance

## Installation

### Prerequisites

- Python 3.7 or higher
- pip or conda package manager

### Setup Instructions

1. Clone or download the project:
   ```bash
   cd Assignment-01
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install torch imbalanced-learn
   ```

## Usage

The project follows a three-stage pipeline executed sequentially:

### Stage 1: Data Preparation

Run the data preparation notebook to clean and split the dataset:

```bash
jupyter notebook src/1_prepare_data.ipynb
```

**Outputs**: `train_set.csv` and `test_set.csv` in the `data/` directory

### Stage 2: Model Training

Execute the training notebook to build, train, and validate neural network models:

```bash
jupyter notebook src/2_train_model.ipynb
```

**Outputs**: 
- `preprocessing_data.pkl` and `hyperparameters.pkl` in `utils/`
- `best_model.pth` and `final_model.pth` in `models/`

**Configuration**:
- Modify `model_type` to select architecture (NeuralNet1, NeuralNet2, NeuralNet3, NeuralNet4)
- Adjust hyperparameters: `hidden_size`, `dropout_rate`, `learning_rate`, `batch_size`, `num_epochs`

### Stage 3: Model Evaluation

Evaluate the trained model on the test set:

```bash
jupyter notebook src/3_test_model.ipynb
```

**Outputs**: Evaluation metrics and AUROC curve visualization

## Methodology

### Neural Network Architectures

Four distinct architectures are implemented for comparison:

| Architecture | Hidden Layers | Regularization | Features |
|-------------|---------------|-----------------|----------|
| NeuralNet1  | 2             | Dropout         | ReLU activation |
| NeuralNet2  | 2             | Dropout + BatchNorm | Improved normalization |
| NeuralNet3  | 3             | Dropout         | Deeper network |
| NeuralNet4  | 3             | Dropout + BatchNorm | Most complex with regularization |

### Training Strategy

1. **Loss Function**: Binary Cross Entropy with Logits (BCEWithLogitsLoss)
2. **Optimizer**: Adam with learning rate 0.001
3. **Early Stopping**: Patience of 25 epochs based on validation loss
4. **Validation Split**: 15% of training data for validation monitoring
5. **Batch Size**: 128 for efficient gradient updates on CPU

### Hyperparameters

```
input_size: 10                  # Top 10 selected features
hidden_size: 1024              # First hidden layer neurons
num_epochs: 200                # Maximum training epochs
batch_size: 128                # Samples per batch
learning_rate: 0.001           # Adam optimizer learning rate
dropout_rate: 0.3              # Dropout probability
size_factor: 1                 # Hidden layer size scaling factor
```

## Results

### Evaluation Metrics

The final model evaluation on the test set includes:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Positive prediction accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **AUROC**: Area under the Receiver Operating Characteristic curve with visualization

### Model Performance Analysis

- Training loss and validation loss curves demonstrate convergence
- Early stopping prevents overfitting by monitoring validation loss
- AUROC curve visualizes classification performance across threshold ranges
- Class distribution analysis ensures balanced evaluation across both classes

## Additional Resources

- **Course**: CSE 472 - Machine Learning (Level 4, Term II)
- **Framework**: PyTorch for neural network implementation
- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## Notes

- All models are optimized to run on CPU; GPU support available via device configuration in notebooks
- Random seed (RANDOM_STATE = 2005021) ensures reproducibility across runs
- Preprocessing pipeline is saved for consistent test data transformation
- Model weights are persisted for inference on new data

---

**Last Updated**: January 2026
