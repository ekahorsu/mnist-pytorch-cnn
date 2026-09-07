# Handwritten Digit Recognition with PyTorch: Tensors to CNNs

A complete deep learning pipeline in PyTorch that classifies handwritten digits from MNIST, working through the full workflow from tensors and automatic differentiation to a convolutional neural network reaching about 99 percent test accuracy.

## Overview

This project builds an end-to-end image classifier in PyTorch and demonstrates every stage of the deep learning workflow taught in the DeepLearning.AI PyTorch course. It is organized to mirror the four stages of that course:

1. **Tensors and autograd**: the core PyTorch data structure and the automatic differentiation that powers learning, verified against a hand calculation.
2. **The training workflow**: the standard loop of forward pass, loss, backward pass, and optimizer step, shown first on a simple linear model.
3. **Data pipelines**: a custom `Dataset`, a `DataLoader`, and normalization transforms, the general pattern for feeding data into PyTorch at scale.
4. **Convolutional neural networks**: learnable filters and pooling assembled with `nn.Conv2d` into a model suited to images.

MNIST is used because it trains in a couple of minutes on a laptop while still exercising the full pipeline. The dataset is bundled with the repository, so the notebook runs without any download.

## Selected results

- Autograd reproduces a hand-computed derivative exactly, illustrating the mechanism behind all model training.
- The training loop recovers a known line (learned y = 2.00x + 1.02 against a true y = 2x + 1).
- The convolutional network has about 207,000 parameters and reaches roughly 99 percent test accuracy within three epochs (98.9 percent in the reference run).
- Of 10,000 test digits, only about 112 are misclassified, and these are mostly genuinely ambiguous or poorly written examples.

## Contents

1. Tensors and autograd
2. The training workflow, demonstrated on linear regression
3. The data pipeline: a custom Dataset, DataLoader, and transforms, with a batch visualization
4. A convolutional neural network: architecture, training, prediction inspection, and analysis of misclassified digits
5. Reflection and key takeaways

## How to run

```bash
git clone https://github.com/ekahorsu/mnist-pytorch-cnn.git
cd mnist-pytorch-cnn
pip install -r requirements.txt
jupyter notebook mnist_pytorch_cnn.ipynb
```

Tested with Python 3.10. No internet connection is required; the MNIST data is included under `data/`. Training runs on CPU in a couple of minutes and automatically uses a GPU if one is available.

## Tech stack

- **PyTorch** for tensors, autograd, the model, and training
- **torch.utils.data** for the custom Dataset and DataLoader
- **NumPy** for reading the raw MNIST files
- **Matplotlib** for the visualizations

## Repository contents

- `mnist_pytorch_cnn.ipynb` - the full analysis notebook
- `data/MNIST/raw/` - the bundled MNIST dataset (gzip files)
- `requirements.txt` - dependencies

## Data source

The data is the MNIST database of handwritten digits, created by Yann LeCun, Corinna Cortes, and Christopher Burges. It contains 60,000 training and 10,000 test images of 28x28 grayscale digits. The copy included here is the standard distribution, bundled for reproducibility so the notebook runs offline.
