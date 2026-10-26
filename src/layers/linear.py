import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Ağırlıklar (W) ve Bias (b)
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.dW = None
        self.db = None
        # Geribayılım (backpropagation) için girdiyi saklayacağımız önbellek
        self.X = None

    def forward(self, X):
        self.X = X
        Y = np.dot(X, self.W) + self.b
        return Y

    def backward(self, grad_output):
        # Ağırlıklar ve bias için gradyanları hesapla
        self.dW = np.dot(self.X.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        # Girdi için gradyanı hesapla
        grad_input = np.dot(grad_output, self.W.T)
        return grad_input   
