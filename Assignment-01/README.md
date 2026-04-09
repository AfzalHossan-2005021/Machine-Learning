# 📊 Assignment 01: Medical Students Diabetes Workflow

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn)

Welcome to the **Medical Students Diabetes Workflow** project! This folder houses a complete end-to-end tabular machine learning pipeline designed to analyze and predict diabetes indicators within a specific demographic dataset.

## 🌟 Project Details & Pipeline Architecture

This assignment translates a raw medical dataset into a robust predictive engine. The workflow is segregated into three rigorous, sequential phases to ensure reproducibility and high integrity of results.

### 1️⃣ Data Preparation & Exploratory Data Analysis (EDA)
**Notebook:** `src/1_prepare_data.ipynb`
- **Data Ingestion:** Loading the primary `medical_students_diabetes_dataset.csv`.
- **EDA:** Visualizing class imbalances, feature distributions, and correlation matrices to understand the underlying demographic indicators.
- **Preprocessing:** Handling missing records via standard imputation, scaling numerical features (e.g., standardizing BMI, age, and glucose levels), and encoding categorical variables.
- **Splitting:** Generating stratified `train_set.csv` and `test_set.csv` to ensure the models are evaluated fairly on unseen holdout data.

### 2️⃣ Model Training & Hyperparameter Optimization
**Notebook:** `src/2_train_model.ipynb`
- **Algorithm Exploration:** Systematically training an array of classifiers (such as Logistic Regression, Random Forest, SVM, or Neural Networks) to establish baseline performances.
- **Cross-Validation:** Utilizing k-fold cross-validation to search the hyperparameter space (Grid Search/Random Search) and combat overfitting.
- **Checkpointing:** The model with the optimal validation score is sterilized and saved directly to the `models/` directory as `best_model.pth` (or `.pkl`) for subsequent retrieval.

### 3️⃣ Final Evaluation & Inference
**Notebook:** `src/3_test_model.ipynb`
- **Model Restoring:** Loading the pre-trained `models/best_model.pth` artifact.
- **Metric Computation:** Executing the model on the `test_set.csv` to compute decisive real-world metrics: Accuracy, Precision, Recall, F1-Score, and plotting ROC-AUC curves.
- **Reporting:** Extracting the final confusion matrix to generate the analysis found in `July25_CSE472_Assignment1.pdf`.

---

## 🗺️ At a Glance

| Item | Details |
| :--- | :--- |
| **Dataset** | `data/medical_students_diabetes_dataset.csv`, `train_set.csv`, `test_set.csv` |
| **Notebooks** | `src/1_prepare_data.ipynb`, `2_train_model.ipynb`, `3_test_model.ipynb` |
| **Checkpoint** | `models/best_model.pth` |
| **Report** | `July25_CSE472_Assignment1.pdf` |
| **Support Code** | `utils/` helper functions |

---

## 📂 Project Structure

| Path | Purpose |
| :--- | :--- |
| 📁 `data/` | Raw and processed CSV files used by the pipeline |
| 📁 `models/` | Saved model artifacts (e.g., `.pth` or `.pkl` objects) |
| 📁 `src/` | Interactive Jupyter Notebooks for step-by-step execution |
| 📁 `utils/` | Custom Python helper utilities and scripts |
| 📄 `requirements.txt` | Python dependencies for the isolated environment |

---

## 🚀 Recommended Workflow

To reproduce the exact findings and train a fresh model, execute the notebooks in this precise order:

1. **Step 1:** Run `src/1_prepare_data.ipynb` to clean the dataset and generate train/test splits.
2. **Step 2:** Run `src/2_train_model.ipynb` to iterate over architectures and train the model.
3. **Step 3:** Run `src/3_test_model.ipynb` to load `models/best_model.pth` and evaluate final metrics.

---

## 🛠️ Setup Instructions

Ensure you have a configured environment before running the notebooks:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

## 📌 Reproducibility Notes
- **Relative Paths:** Keep the CSV files inside `data/` as the notebooks rely on these relative paths.
- **State Management:** Always run notebooks sequentially from a freshly restarted kernel to recreate the final outputs reliably.
