# 🧠 CSE 472: Machine Learning Coursework

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn)

Welcome to the central repository for the **CSE 472: Machine Learning** coursework! 🚀 

This comprehensive repository collects all deliverables, including meticulously structured notebook-driven data pipelines, standalone **PyTorch** deep learning experiments, pre-processed datasets, saved checkpoints, and detailed project reports. Everything is structured to ensure that each project can be reviewed and reproduced entirely independently from one another.

---

## 🌟 Project Details & Comprehensive Overview

### 📊 [Assignment 01: Medical Students Diabetes Workflow](Assignment-01/)
An end-to-end machine learning pipeline focused entirely on tabular data processing and classification. 
*   **Domain Focus:** Predicting diabetes indicators in a targeted medical student demographic.
*   **Methodology:** The project incorporates comprehensive Exploratory Data Analysis (EDA), robust data imputation and cleaning, feature scaling, and extensive hyperparameter tuning across multiple models to find the optimal decision boundary.
*   **Tech Stack:** Python, Pandas, Scikit-Learn, Matplotlib, Jupyter Lab

### 🌳 [Assignment 02: Decision Tree Implementation from Scratch](Assignment-02/)
A deep dive into tree-based algorithms featuring a **custom-built Decision Tree model** written entirely from scratch without standard library ML wrappers.
*   **Domain Focus:** Algorithmic translation of Decision Tree theoretical mathematics into functional Python code.
*   **Methodology:** Focuses heavily on the core mathematics of **Information Gain** (Entropy) and **Gini Impurity** index for optimal node-splitting criteria. It handles recursive tree traversal, stopping conditions, and model evaluation metrics over continuous and categorical features.
*   **Tech Stack:** Python, NumPy, Custom ML Utilities

### 🖼️ [Online 01: Deep Learning CNN Experiment Collection](Online-01/)
A series of standalone, highly-specialized deep learning experiments demonstrating various modern **Convolutional Neural Network (CNN)** architectural patterns on standard image datasets (CIFAR-10, MNIST).
*   **A1: Network in Network (NiN):** CIFAR-10 classification utilizing $1 \times 1$ convolutions for complex cross-channel feature pooling instead of fully connected layers, capped with a Global Average Pooling layer.
*   **A2: U-Net Architecture:** Notebook-based MNIST experiment featuring a powerful encoder-decoder symmetric network with skip connections, heavily suited for complex visual structures.
*   **B1: MobileNetV1-style:** CIFAR-10 classification implementing highly efficient **Depthwise Separable Convolutions** to drastically reduce the parameter count and computational cost while preserving representational power.
*   **B2: SqueezeNet-like:** MNIST classification leveraging custom **Fire modules** (comprising a "squeeze" convolution layer paired with an "expand" layer) to aggressively compress model sizes without sacrificing accuracy.

---

## 🗺️ Project Map

| Directory | Core Focus / Topic | Essential Files | Execution Output |
| :--- | :--- | :--- | :--- |
| `📁 Assignment-01/` | E2E Data Pipeline & Classification | `src/*.ipynb` | `models/best_model.pth`, `*.pdf` |
| `📁 Assignment-02/` | Raw Custom Decision Tree | `custom/`, `utils/` | `Submission Report.pdf` |
| `📁 Online-01/A1/` | NiN CNN on CIFAR-10 | `cnn.py` | `model.ckpt` |
| `📁 Online-01/A2/` | UNet Experiment on MNIST | `Unet_Online.ipynb` | `Local Model Weights` |
| `📁 Online-01/B1/` | MobileNet CNN on CIFAR-10 | `Online-B1.py` | `model.ckpt` |
| `📁 Online-01/B2/` | SqueezeNet on MNIST | `Question.py` | `Local Model Weights` |

---

## 🛠️ Environment Setup & Installation

The notebook workflow requires standard numerical and scientific computing dependencies, whilst the deep-learning scripts heavily utilize `PyTorch` and `TorchVision`.

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install core tabular and analytical dependencies
pip install -r Assignment-01\requirements.txt

# 3. Install deep learning dependencies
pip install torch torchvision

# 4. Launch Jupyter environment for notebooks
jupyter lab
```

---

## 🚀 Running the Projects

### Assignment 01 (Tabular Pipeline)
Navigate to the `Assignment-01/src/` directory and run the notebooks in strict chronological order to ensure the data propagates correctly:
1. `1_prepare_data.ipynb` — *Cleans and sections the raw CSV datasets.*
2. `2_train_model.ipynb` — *Trains algorithms and saves the best fit to `.pth`.*
3. `3_test_model.ipynb` — *Evaluates the `best_model.pth` against hold-out sets.*

### Online 01 (Deep Learning)
For the standalone Python experiments, execute each script from its **own sub-folder**. This is crucial as their underlying scripts rely on localized relative paths to construct the `data/` caches:
```cmd
# Navigate into specific folders to run
cd Online-01\A1 && python cnn.py
cd ..\B1 && python Online-B1.py
cd ..\B2 && python Question.py
```
*Note: For `Online-01/A2/Unet_Online.ipynb`, standard top-to-bottom notebook execution applies.*

---

## 📌 Reproducibility & Technical Notes
- **Automatic Downloads:** Standard datasets (CIFAR-10, MNIST) will dynamically fetch and download via `torchvision.datasets` if they aren't detected in local `data/` caches.
- **Model Checkpoints:** Weight iterations and checkpoints are persistently saved adjacent to the training scripts that spun them up.
- **Notebook State:** Notebooks rely heavily on internal state and sequential execution. If variables behave unexpectedly, **Restart the Kernel and Run All** to clear corrupted memory footprints.
