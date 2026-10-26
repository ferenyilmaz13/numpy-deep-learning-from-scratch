class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, X): # Girdiyi baştan sona tüm katmanlardan geçirir
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_loss): # Hata oranını baştan sona tüm katmanlardan geçirir
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
