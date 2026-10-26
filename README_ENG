# 🧠 Deep Learning Framework from Scratch using NumPy

This project is a modular deep learning framework built entirely from scratch using only **NumPy** and fundamental matrix calculus—without relying on any high-level deep learning libraries like PyTorch, TensorFlow, or JAX.

The framework covers forward pass execution, backward pass logic (manual chain rule implementation), activation functions, loss computations, and adaptive optimization algorithms (Adam).

---

## 🚀 Key Features

* **Fully Modular Architecture:** Layer design inspired by PyTorch’s `nn.Module` abstraction.
* **Analytical Gradients & Chain Rule:** Hand-derived backward pass gradients computed via matrix transpositions for all layers.
* **Adaptive Optimization:** Includes the **Adam Optimizer** featuring Momentum, RMSProp, and Bias Correction.
* **Zero External Dependencies:** Built strictly using `NumPy`.

---

## 📐 Mathematical & Architectural Components

### 1. Layers & Activations (`nn/layers.py`)

* **`Linear` Layer:**
  * *Forward Pass:* $Z = X \cdot W + b$
  * *Backward Pass:* $dW = X^\top \cdot dZ$, $db = \sum dZ$, $dX = dZ \cdot W^\top$
* **`ReLU` Activation:**
  * $f(Z) = \max(0, Z)$
  * $dZ = dH \odot (Z > 0)$
* **`MSELoss` (Mean Squared Error):**
  * $\text{Loss} = \frac{1}{N} \sum (\hat{Y} - Y)^2$
  * $d\hat{Y} = \frac{2}{N} (\hat{Y} - Y)$
* **`Sequential` Module:** A chained container executing layers sequentially in the forward pass and in reverse order during backpropagation.

### 2. Optimizers (`nn/optimizers.py`)

* **`SGD`:** Parameter update using a constant learning rate ($\theta \leftarrow \theta - \eta \cdot g$).
* **`Adam`:**
  * 1st Moment (Momentum): $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
  * 2nd Moment (RMSProp): $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
  * Bias Correction: $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$
  * Update: $\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

---

## 📊 Training & Performance

A multi-layer neural network built with this framework was trained over 100 epochs, demonstrating efficient loss convergence:

* **Epoch 1:** Loss = `0.3948`
* **Epoch 50:** Loss = `0.0198`
* **Epoch 100:** Loss = **`0.0053`**

---

## 🛠️ Setup & Execution

```bash
# Clone the repository
git clone <repository_url>
cd numpy-deep-learning-from-scratch

# Install required dependencies
pip install numpy

# Run training script
python train.py
```
