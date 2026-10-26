import numpy as np

class ReLU:
    def __init__(self):
        # ReLU aktivasyon fonksiyonu için önbellek
        self.X = None

    def forward(self, X):
        self.X = X
        Y = np.maximum(0, X)
        return Y

    def backward(self, grad_output):
            # Pozitifleri aynen geçir, negatifleri 0 yap
            return grad_output * (self.X > 0)