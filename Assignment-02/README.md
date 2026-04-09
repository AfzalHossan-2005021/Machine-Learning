# 🌳 Assignment 02: Decision Tree Implementation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Core%20Math-013243?style=for-the-badge&logo=numpy)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Custom%20Algorithms-8A2BE2?style=for-the-badge)

Welcome to the **Decision Tree Implementation** project! This assignment eschews high-level wrapper libraries in favor of building a complete tree-based classifier from the ground up.

## 🌟 Project Details

Unlike off-the-shelf library imports (like Scikit-Learn), this assignment required meticulously coding the core mechanics of a Decision Tree classifier entirely in Python. Key educational and technical aspects of this project include:
- **Custom Algorithmic Implementation:** Building the hierarchical tree data structure, implementing exact mathematical node-splitting criteria (utilizing both **Information Gain/Entropy** and **Gini Impurity**), and managing recursive traversal. Core logic is neatly housed within the `custom/` and `utils/` modules.
- **Handling Data Variance:** Successfully processing both continuous and categorical features while identifying non-linear decision boundaries.
- **Advanced Reporting:** A comprehensively detailed analysis (`CSE_472_Assignment_2_DT.pdf`) documenting all critical design choices, computational complexities, pruning logic, and experimental performance metrics of the custom tree deployed across various standardized datasets.

---

## 🗺️ At a Glance

| Item | Details |
| :--- | :--- |
| **Focus** | Decision Tree classifier built from scratch |
| **Core Delivery** | `CSE_472_Assignment_2_DT.pdf` |
| **Components** | `custom/`, `utils/` |

---

## 📂 Project Structure

| Path | Purpose |
| :--- | :--- |
| 📄 `CSE_472_Assignment_2_DT.pdf` | Submitted report containing methodology and results |
| 📁 `custom/` | Project-specific tree implementation source codes |
| 📁 `utils/` | Shared utility code for data loading and metric evaluation |

---

## 📌 Usage & Notes

- **Implementation Details:** The workspace is intentionally modular. Utility functions inside `utils/` support the core logic stored recursively within `custom/`.
- **Running the Code:** If subsequent testing scripts are added, they should import the tree class from `custom/` and utilize `utils/` for data preprocessing. The primary artifact for this delivery remains the comprehensive PDF report.
