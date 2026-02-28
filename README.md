# Iris MLP from Scratch

A Multi-Layer Perceptron built from scratch using Karpathy's micrograd library, trained on the classic Iris dataset for flower species prediction.

## What it does

Takes the 4 measurements of an Iris flower (sepal length, sepal width, petal length, petal width) and predicts which of the 3 species it belongs to. Achieves ~85% accuracy.

## Why I built it

After going through Andrej Karpathy's micrograd video, I wanted to go beyond just watching and actually build something real. So I implemented my own MLP from scratch — no PyTorch, no high level abstractions — just raw neurons, layers, and backprop.

## How it works

- Built `Neuron`, `Layer`, and `MLP` classes manually using micrograd's `Value` engine
- Each neuron computes `w * x + b` and passes it forward
- Loss is computed using Mean Squared Error across all 150 samples
- Gradients are computed via backpropagation and weights updated using gradient descent

## What I learned

The hardest part was getting the loss to actually go down — I ran into exploding gradients and had to normalize the input data and tune the learning rate to fix it. That taught me more about training dynamics than any tutorial could.

## Stack

- Python
- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- scikit-learn (for dataset and normalization)
- NumPy

## Usage

```python
# Install dependencies
pip install micrograd scikit-learn numpy

# Run the notebook
jupyter notebook iris_mlp.ipynb
```

## Results

| Architecture | Steps | Accuracy |
|-------------|-------|----------|
| 4 → 8 → 3 | 100 | 58.7% |
| 4 → 8 → 3 | 300 | 84.0% |
| 4 → 16 → 8 → 3 | 300 | 85.3% |
