import numpy as np 

class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.grad = None
        pass

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self):
        N = self.y_pred.size # N : matristeki toplam eleman sayısı (Örn: 3x2 matris için N=6)

        # MSE'nin y_pred'e göre türevi: (2 / N) * (y_pred - y_true) -> türevden önceki hali 1/N * (y_pred - y_true) ** 2
        grad = (2 / N) * (self.y_pred - self.y_true)
        return grad