class SGD:
    
    def __init__(self, layers, learning_rate=0.1):
        self.layers = layers # Ağımızdaki katmanların listesi
        self.learning_rate = learning_rate # Katmanların ağırlıklarının güncelleme hızı

    def step(self): 
        # Tüm katmanları gezerek, ağırlıkları ve biasları gradyanın tersi yönünde (eksili haliyle çarparak) günceller
        for layer in self.layers:
            if hasattr(layer, 'W') and layer.dW is not None:  # Eğer katman ağırlıklara sahipse ve gradyan hesaplandıysa
                layer.W -= self.learning_rate * layer.dW
            if hasattr(layer, "b") and layer.db is not None:  # Eğer katman biaslara sahipse ve gradyan hesaplandıysa
                layer.b -= self.learning_rate * layer.db