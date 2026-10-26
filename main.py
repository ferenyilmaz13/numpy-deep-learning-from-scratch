import numpy as np
from src.layers.linear import Linear
from src.layers.relu import ReLU
from src.losses.mse import MSELoss
from src.optimizers.sgd import SGD
from src.optimizers.sequential import Sequential
from src.optimizers.adam import Adam

np.random.seed(13)
 
# Oluşturduğumu başlangıç noktasını ve hangi konuma evrilmesini istediğimiz çıktıyı rasgele olarak belirliyoruz
X = np.random.rand(3, 2)
Y_true = np.random.rand(3, 4)

model = Sequential( # 2. Modelimizi oluşturduğumuz sequential sınıfı ile beraber tuğla koyar gibi inşa ediyoruz.
    Linear(input_dim=X.shape[1], output_dim=16), 
    ReLU(),
    Linear(input_dim=16, output_dim=16),
    ReLU(),
    Linear(input_dim=16, output_dim=Y_true.shape[1])
    )

loss_fn = MSELoss()
# 3. Katmanları Optimizatöre (SGS) veriyoryuz
# optimizer = SGD(layers=model.layers, learning_rate=0.1) 
# 3. Katmanları Optimizatöre (Adam) veriyoryuz
optimizer = Adam(layers=model.layers, learning_rat=0.01) 
epoch = 100
print(f"--- EĞİTİM BAŞLIYOR ({epoch} EPOCH) ---\n")

for epoch in range(1, (epoch + 1)):
    # 1-) İleri Besleme (Forward Pass) (Artık yeni sequential sınıfı ile beraber tek tek katman çağırmak yerine tek satırda katmanları liste olarak gösterip model olarak veriyoruz)
    predictions = model.forward(X) # Girdiyi baştan sona tüm katmanlardan geçirir
    loss = loss_fn.forward(predictions, Y_true) # Tahmin edilen değerler ile gerçek değerler arasındaki kayıp (loss) hesaplanır
    # 2 -) Geri Besleme (Backward Pass)
    # Hata payına göre ağırlıkları güncellemek için gradyanları hesaplıyor
    grad_loss = loss_fn.backward()
    model.backward(grad_loss) # Hata oranını baştan sona tüm katmanlardan geçirir
    # 3 -) Ağırlık Güncellemesi (Parameter Update)
    # Optimizer'ı kullanarak ağırlıkları güncelliyor
    optimizer.step()
    # Belirlediğimiz epoch kadar bekledikten sonra hata paylarını vb. şeyleri log olarak yazdırıyoruz
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Loss (Kayıp): {loss:.6f}")

print("\n--- EĞİTİM BİTTİ ---\n")

print("Girdi (X) Şekli:", X.shape)
print("Doğru Çıktı (Y_true) Şekli:", Y_true.shape)
print("Nihai Tahmin (predictions) Şekli:", predictions.shape)

print("\n--- TAHMİN VE HATA ANALİZİ ---")
print("Gerçek Hedef Değerler (Y_true):\n", Y_true)
print("\nModelin Yaptığı Tahminler (predictions):\n", predictions)
print("\nFark (Mutlak Hata):\n", np.abs(Y_true - predictions))



