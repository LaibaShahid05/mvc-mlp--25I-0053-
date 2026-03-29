# MVC Project: Multilayer Perceptron from Scratch

**Roll No.:** 25I-0053 &nbsp;|&nbsp; **Name:** Laiba Shahid &nbsp;|&nbsp; **Section:** AI-B  
**Course:** Artificial Neural Networks &nbsp;|&nbsp; **Institution:** NUCES–FAST, Islamabad

---

## Overview

This repository contains the complete submission for the MVC (Manual–Verify–Code) project for the Artificial Neural Networks course. The project implements a Multilayer Perceptron entirely from scratch — first by hand on paper (Tasks 1–6), then in Python using NumPy only (Task 7).

The manual tasks cover every computation inside a neural network on a small 3-sample student-performance dataset using unique per-student weights. Task 7 scales the identical algorithm to the MNIST handwritten digit benchmark (60,000 training images, 10,000 test images) and achieves approximately **92–94% test accuracy** after 20 epochs.

---

## Repository Structure

```
mvc-mlp-25I0053/
│
├── src/
│   └── mvc_mlp_25i0053.ipynb     # Task 7 — complete NumPy implementation (all cells executed)
│
├── report/
│   └── mvc-report-25I0053.pdf    # LaTeX report compiled from Overleaf (Springer LNCS template)
│
└── README.md                     # This file
```

---

## Assigned Weights & Biases

Each student received a unique set of initial weights and biases drawn from the class roster.

| Parameter | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w9 | w10 | b(1) | b(2) |
|-----------|----|----|----|----|----|----|----|----|----|----|------|------|
| Value | −0.47 | −0.80 | 0.75 | 0.62 | −0.57 | 0.39 | 0.30 | 0.22 | −0.86 | 0.84 | 0.05 | 0.34 |

---

## Network Architecture

### Tasks 1–6 (toy network, manual)

| Layer | Type | Neurons | Activation |
|-------|------|---------|------------|
| Layer 0 | Input | 2 | — |
| Layer 1 | Hidden 1 | 2 | Sigmoid σ |
| Layer 2 | Hidden 2 | 2 | Sigmoid σ |
| Layer 3 | Output | 1 | Sigmoid σ |

10 weights (w1–w10) + 2 shared bias terms (b(1), b(2)). No output bias.

### Task 7 (MNIST, NumPy)

| Layer | Type | Neurons | Activation |
|-------|------|---------|------------|
| Layer 0 | Input | 784 | — |
| Layer 1 | Hidden 1 | 128 | Sigmoid σ |
| Layer 2 | Hidden 2 | 64 | Sigmoid σ |
| Layer 3 | Output | 10 | Sigmoid σ |

---

## Dataset

### Student-Performance Dataset (Tasks 1–6)

| Sample | x1 (Study Hours) | x2 (Attendance) | y (Pass?) |
|--------|-----------------|-----------------|-----------|
| s(1) | 0.2 | 0.8 | 1 |
| s(2) | 0.9 | 0.4 | 1 |
| s(3) | 0.1 | 0.2 | 0 |

All features normalised to [0, 1].

### MNIST (Task 7)

- 60,000 training images + 10,000 test images
- 28×28 greyscale images, flattened to 784-dimensional vectors
- Pixel values normalised from [0, 255] → [0, 1] by dividing by 255
- Integer labels (0–9) converted to 10-class one-hot vectors

---

## Project Tasks Summary

| Task | Description | Method |
|------|-------------|--------|
| **1** | Forward Pass | Manual — symbolic equations + numerical computation for all 3 samples |
| **2** | Loss Calculation | Manual — MSE expanded for m=3, computed: **L = 0.2427** |
| **3** | Backpropagation | Manual — full delta method, all 12 gradients computed for s(1) |
| **4** | Weight Update & Iterations | Manual — gradient descent η=0.1, 5 iterations on s(1), loss: 0.2427 → 0.2367 |
| **5** | Gradient Descent Variants | Manual — 3 epochs mini-batch (B=2), compare Batch/SGD/Mini-Batch |
| **6** | Optimisers | Manual — Momentum (β=0.9) and NAG, 10-iteration comparison |
| **7** | Python Implementation | NumPy — full MLP on MNIST, 20 epochs, ~92–94% test accuracy |

---

## Key Results

### Manual Tasks (Tasks 1–6)

**Forward Pass — Sample s(1):**
```
z(1)_1 = 0.5560  →  a(1)_1 = 0.6355
z(1)_2 = 0.3860  →  a(1)_2 = 0.5953
z(2)_1 = 0.1564  →  a(2)_1 = 0.5390
z(2)_2 = 0.7188  →  a(2)_2 = 0.6723
z(3)_1 = 0.1012  →  y_hat  = 0.5253
```

**Initial MSE Loss:** 0.2427

**Backpropagation — Key deltas:**
```
δ(3)_1 = −0.2367
δ(2)_1 = +0.0506    δ(2)_2 = −0.0438
δ(1)_1 = −0.0097    δ(1)_2 = +0.0024
```

**Task 4C — 5 SGD iterations on s(1):**

| Iteration | MSE Loss |
|-----------|----------|
| 0 (initial) | 0.2427 |
| 1 | 0.2414 |
| 2 | 0.2401 |
| 3 | 0.2389 |
| 4 | 0.2378 |
| 5 | 0.2367 |

**Task 6C — Optimiser comparison (10 iterations, mini-batch B=2):**

| Iteration | Plain GD | Momentum (β=0.9) | NAG (β=0.9) |
|-----------|----------|-----------------|-------------|
| 1 | 0.2429 | 0.2426 | 0.2426 |
| 5 | 0.2394 | 0.2414 | 0.2414 |
| 10 | 0.2367 | 0.2385 | 0.2386 |

### MNIST (Task 7)

| Metric | Value |
|--------|-------|
| Architecture | 784 → 128 → 64 → 10 (Sigmoid) |
| Loss function | MSE |
| Optimiser | Mini-batch GD |
| Learning rate η | 0.1 |
| Batch size B | 32 |
| Epochs | 20 |
| Initial MSE (Epoch 1) | ≈ 0.09 |
| Final MSE (Epoch 20) | ≈ 0.02 |
| Test accuracy | ≈ 92–94% |

---

## How to Run

### Requirements

```bash
pip install numpy matplotlib
```

No PyTorch or TensorFlow required — all training uses NumPy only.

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/laibashahid/mvc-mlp-25I0053.git
   cd mvc-mlp-25I0053
   ```

2. Open the notebook:
   ```bash
   jupyter notebook src/mvc_mlp_25i0053.ipynb
   ```
   Or open directly in **Google Colab** (recommended — MNIST downloads automatically on first run).

3. Run all cells: `Runtime → Run all`

The notebook downloads the MNIST dataset automatically on first run. Training takes approximately 2–5 minutes on CPU.

### Reproducibility

All results are fully reproducible via:
```python
np.random.seed(42)
```
Set at the top of the notebook before weight initialisation.

---

## Implementation Details

Each NumPy function maps directly to a manual task:

| Function | Corresponds to | Purpose |
|----------|---------------|---------|
| `sigmoid(z)` | Task 1 | Activation: 1/(1+e^−z), clipped at ±500 |
| `sigmoid_derivative(a)` | Task 3 | Gradient: a·(1−a) |
| `forward_pass(...)` | Task 1B/1C | Layer-by-layer prediction, returns all intermediates |
| `mse_loss(Y_true, Y_pred)` | Task 2 | Mean squared error over batch |
| `backpropagation(...)` | Task 3B | Delta method — exact match to manual derivation |
| `update_weights(...)` | Task 4A | w ← w − η·∂L/∂w |

The backpropagation function mirrors the delta notation exactly as derived by hand:

```python
delta3 = -2.0 * (Y_true - A3) * sigmoid_derivative(A3)   # output delta
delta2 = (delta3 @ W3.T) * sigmoid_derivative(A2)          # hidden layer 2
delta1 = (delta2 @ W2.T) * sigmoid_derivative(A1)          # hidden layer 1
```

---

## References

1. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proc. IEEE*, 86(11), 2278–2324.
2. Rumelhart, D.E., Hinton, G.E., Williams, R.J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.
3. LeCun, Y., Bengio, Y., Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.
4. Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.
5. Sutskever, I., Martens, J., Dahl, G., Hinton, G. (2013). On the importance of initialization and momentum in deep learning. *ICML*, pp. 1139–1147.

---

*Submitted as part of the MVC Project — Artificial Neural Networks, NUCES–FAST Islamabad, 2025.*
