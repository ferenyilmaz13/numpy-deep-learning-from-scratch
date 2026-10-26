# 🧠 NumPy ile Sıfırdan Derin Öğrenme Kütüphanesi (Deep Learning from Scratch)

Bu proje; PyTorch, TensorFlow veya JAX gibi herhangi bir hazır derin öğrenme kütüphanesi **kullanılmadan**, sadece **NumPy** ve temel matris matematiği kullanılarak sıfırdan inşa edilmiş modüler bir derin öğrenme altyapısıdır.

Proje kapsamı; ileri besleme (forward pass), geriye yayılım (backward pass / automatic differentiation logic), aktivasyon fonksiyonları, kayıp hesaplamaları ve adaptif optimizasyon algoritmalarını (Adam) içerir.

---

## 🚀 Öne Çıkan Özellikler

- **Tamamen Modüler Yapı:** PyTorch `nn.Module` mimarisine benzer modüler katman tasarımı.
- **Türev Akışı ve Zincir Kuralı (Chain Rule):** Tüm katmanlar için matris transpozları üzerinden elle türetilmiş geriye yayılım hesabı.
- **Adaptif Optimizasyon:** Momentum ve RMSProp mekanizmalarını içeren, Bias Correction destekli **Adam Optimizatörü**.
- **Sıfır Dış Bağımlılık:** Sadece `NumPy`.

---

## 📐 Matematiksel ve Mimarî Bileşenler

### 1. Katmanlar & Aktivasyonlar (`nn/layers.py`)

- **`Linear` Katmanı:**
  * *İleri Besleme:* $Z = X \cdot W + b$
  * *Geriye Yayılım:* $dW = X^\top \cdot dZ$, $db = \sum dZ$, $dX = dZ \cdot W^\top$
- **`ReLU` Katmanı:**
  * $f(Z) = \max(0, Z)$
  * $dZ = dH \odot (Z > 0)$
- **`MSELoss` (Ortalama Kare Hata):**
  * $\text{Loss} = \frac{1}{N} \sum (\hat{Y} - Y)^2$
  * $d\hat{Y} = \frac{2}{N} (\hat{Y} - Y)$
- **`Sequential` Modülü:** Katmanları ileriye doğru sırayla, geriye doğru ise ters sırada çalıştıran zincirleme mimari.

### 2. Optimizatörler (`nn/optimizers.py`)

- **`SGD`:** Sabit öğrenme oranı ile parametre güncellemesi ($\theta \leftarrow \theta - \eta \cdot g$).
- **`Adam`:**
  * 1. Moment (Momentum): $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
  * 2. Moment (RMSProp): $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
  * Bias Düzeltmesi: $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$
  * Güncelleme: $\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

---

## 📊 Eğitim ve Performans

Yazılan kütüphane üzerinde inşa edilen çok katmanlı yapay sinir ağı 100 epoch boyunca eğitilmiş ve kayıp (loss) değerinin değişimi gözlemlenmiştir:

- **Epoch 1:** Loss = `0.3948`
- **Epoch 50:** Loss = `0.0198`
- **Epoch 100:** Loss = **`0.0053`**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Depoyu klonlayın
git clone <repository_url>
cd numpy-deep-learning-from-scratch

# Gerekli paketi yükleyin
pip install numpy

# Eğitimi başlatın
python train.py
