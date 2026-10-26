import numpy as np 

class Adam: 
    def __init__(self, layers, learning_rat=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers
        self.learning_rate = learning_rat
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
        
        self.m_W = {} 
        self.v_W = {}
        self.m_b = {}
        self.v_b = {}

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                self.m_W[i] = np.zeros_like(layer.W)
                self.m_b[i] = np.zeros_like(layer.b)
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)

    def step(self):
        self.t += 1

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                # W(Ağırlık) güncellemesi
                self.dW = layer.dW
                self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.dW
                self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (layer.dW ** 2)

                # Düzeltilmiş momentler (Bias correction)
                m_hat_W = self.m_W[i] / (1 - self.beta1 ** self.t)
                v_hat_W = self.v_W[i] / (1 - self.beta2 ** self.t)

                # Parametre güncellemesi
                layer.W -= self.learning_rate * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)

                # Bias güncellemesi
                db = layer.db
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db ** 2)
            
                # Düzeltilmiş momentler (Bias correction)
                m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

                # Parametre güncellemesi
                layer.b -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)