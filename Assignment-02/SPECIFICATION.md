# CSE 472 – Assignment 2

## Decision Trees, Random Forests, and Extra Trees (From Scratch)

**Course:** CSE 472 – Machine Learning  
**Department:** Computer Science and Engineering  
**Assignment:** 2  
**Last Compiled:** January 8, 2026

---

## 📌 Overview

This project implements **tree-based learning algorithms from scratch** and empirically evaluates how ensemble methods improve generalization performance. The following models are implemented **without using any tree-based models from scikit-learn**:

- Decision Tree
- Random Forest
- Extremely Randomized Trees (Extra Trees)

The implementations are compared against their optimized counterparts from **scikit-learn** using standard classification metrics.

---

## 🎯 Objectives

- Understand the working principles of decision trees
- Implement ensemble techniques such as Random Forests and Extra Trees
- Analyze bias–variance trade-offs empirically
- Compare custom implementations with industrial-grade libraries

---

## 🧠 Implemented Algorithms

### Custom Implementations (From Scratch)

- Decision Tree (Classification)
- Random Forest (Classification)
- Extra Trees (Classification)

### Library Implementations (For Comparison)

- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.ensemble.ExtraTreesClassifier`

---

## 📊 Datasets

The following datasets are used for experimentation:

- **Iris**
- **Wine**

Both datasets are loaded from `sklearn.datasets`.

---

## ⚙️ Configurable Hyperparameters

All custom models support the following configurable hyperparameters:

- Maximum tree depth
- Minimum samples per split
- Number of trees (for ensembles)
- Number of features considered per split

All randomness is controlled using a **fixed random seed** to ensure reproducibility.

---

## 📈 Evaluation Metrics

The following metrics are used for classification performance evaluation:

- Accuracy
- F1-score
- AUROC

---

## 📊 Required Comparisons

For each dataset, results must be reported for:

- Custom Decision Tree
- Custom Random Forest
- Custom Extra Trees
- scikit-learn Decision Tree
- scikit-learn Random Forest
- scikit-learn Extra Trees

Results are reported in **clearly labeled tables** for all models and datasets.

---
