# 🖼️ Online 01: Deep Learning Experiment Collection

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNNs-00A98F?style=for-the-badge)

Welcome to the **Online Deep Learning Experiments** collection! This module features rapid, highly-focused architectural experiments leveraging standard image datasets (CIFAR-10, MNIST) to demonstrate the evolution, scaling, and diversity of modern Convolutional Neural Network (CNN) strategies.

## 🌟 Project Details

Each subdirectory within this collection pairs a classic computer vision classification problem with a mathematically distinct architectural pattern authored in **PyTorch**.
- **A1 (NiN Network):** Explores the *Network in Network* concept utilizing $1 \times 1$ convolutions to build micro-networks within the filter layers, ultimately using Global Average Pooling to massively reduce parameter counts on the **CIFAR-10** dataset.
- **A2 (UNet Autoencoder):** A detailed notebook-based experiment leveraging a symmetric encoder-decoder network complete with skip connections. This structure effectively preserves spatial resolutions suited for complex feature extraction on **MNIST**.
- **B1 (MobileNetV1 style):** Implements *Depthwise Separable Convolutions*, splitting standard convolutions into depthwise and pointwise operations. This builds a highly efficient, lightweight CNN heavily tailored for speed on **CIFAR-10**.
- **B2 (SqueezeNet style):** Constructs custom *Fire Modules* (comprising a "squeeze" $1 \times 1$ convolution layer paired with an "expand" mixed $1 \times 1$ and $3 \times 3$ layer) to aggressively compress model sizes locally while maximizing accuracy on **MNIST**.

---

## 🗺️ Experiment Map

| Folder | Dataset | Source Code | Architecture / Approach | Output Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| 📁 `A1/` | **CIFAR-10** | `cnn.py` | NiN3 with Global Average Pooling | `model.ckpt` |
| 📁 `A2/` | **MNIST** | `Unet_Online.ipynb`| Notebook-based Unet Autoencoder | *Local checkpt.* |
| 📁 `B1/` | **CIFAR-10** | `Online-B1.py` | Depthwise Separable CNN | `model.ckpt` |
| 📁 `B2/` | **MNIST** | `Question.py` | SqueezeLite with Fire modules | *Local checkpt.* |

---

## 🚀 Running the Experiments

To ensure scripts correctly resolve their relative dataset paths, **you must execute each script from within its own directory.**

### Terminal Executions
```cmd
# Run A1 Experiment
cd A1
python cnn.py
cd ..

# Run B1 Experiment
cd B1
python Online-B1.py
cd ..

# Run B2 Experiment
cd B2
python Question.py
cd ..
```

### Notebook Executions
For the **A2** experiment, simply open `A2/Unet_Online.ipynb` in VS Code or Jupyter Lab and run the cells sequentially.

---

## 📌 Reproducibility Notes

- **Autodownload:** The *CIFAR-10* and *MNIST* datasets will automatically download to their respective `data/` subfolders via `torchvision.datasets` if they aren't detected locally.
- **Artifacts:** PyTorch checkpoints (`.ckpt` or `.pth`) are saved directly beside the experiment scripts that created them.
