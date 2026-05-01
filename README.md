<div align="center">
  <h1 style="color: #1a3a5f;">🌬️ Wind Turbine Active Power Prediction</h1>
  <p><strong>Digital Twin & Spatial Correlation for Blind Prediction Case Study</strong></p>
  
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/LightGBM-Expert-orange?style=for-the-badge" alt="LightGBM">
  <img src="https://img.shields.io/badge/CatBoost-Optimized-green?style=for-the-badge" alt="CatBoost">
  <img src="https://img.shields.io/badge/Energy-Digital_Twin-blueviolet?style=for-the-badge" alt="Digital Twin">
</div>

---

## 📝 <span style="color: #2980b9;">Project Overview</span>
This project focuses on predicting the **Active Power** of a target wind turbine (**T1**) in a 7-turbine farm located in Scotland (Hill of Towie). The primary challenge is **Blind Prediction**: we assume T1's wind sensors are faulty and reconstruct its power output using a **Digital Twin** approach based on neighboring turbines and global climate data.

*   **Dataset:** ~200,000 rows of high-resolution SCADA data. [Kaggle Competition Link](https://www.kaggle.com/competitions/hill-of-towie-wind-turbine-power-prediction/overview)
*   **Target Accuracy:** Ranked equivalent to **12th place** with a final **MAE of 50.36**.

---

## 🛠️ <span style="color: #c0392b;">Technical Stack & Toolkit</span>

| Tool | Purpose |
| :--- | :--- |
| **LightGBM** | Fast gradient boosting for high-speed training. |
| **CatBoost** | Managing categorical features and preventing overfitting. |
| **RidgeCV** | Meta-learning for robust model stacking/blending. |
| **Scikit-learn** | Pipeline management and robust scaling of data. |
| **Pandas/NumPy** | High-performance data manipulation and cleaning. |
| **Matplotlib** | Visualizing power curves and spatial correlations. |

---

## 🚀 <span style="color: #27ae60;">Engineering Pipeline</span>

### 1. Preprocessing & Data Integrity
*   **Leakage Prevention:** Removed all T1-specific wind and temp sensors to ensure a true blind prediction.
*   **Operational Filtering:** Used `is_valid` flags to remove downtime and maintenance anomalies.
*   **Robust Scaling:** Applied `RobustScaler` to neutralize the impact of outliers in high-variance wind data.

### 2. Physical Feature Engineering
*   **Air Density ($\rho$):** Calculated using $P / (R \cdot T)$ to model mass flow through the blades.
*   **Wind Vectors (U & V):** Converted 360° periodic wind direction into Sin/Cos components for linear model stability.
*   **Yaw Error Estimate:** Analyzed nacelle positions vs. ERA5 wind direction to assess alignment efficiency.

### 3. Model Architecture (Stacking Ensemble)
*   **Dual-Base Learners:** Used 2000-iteration deep trees for both LGBM and CatBoost.
*   **Log-Transform:** Applied `log1p` to Active Power to increase sensitivity in low-wind regimes.
*   **5-Fold CV:** Ensured stability across all seasonal cycles through cross-validation.

---

## 📊 <span style="color: #8e44ad;">Performance Results</span>

> **Final Score: 50.36 MAE**
> 
> In a 2315 kW turbine, this error represents a **~2.1% deviation**, proving that our Digital Twin can accurately "sense" the environment without direct wind sensors.

---

## ⚙️ <span style="color: #d35400;">Installation & Usage</span>

Clone the repository and install dependencies:
```bash
git clone [https://github.com/hilalay3-creator/WIND-TURBINE-ACTIVE-POWER-PREDICTION.git](https://github.com/hilalay3-creator/WIND-TURBINE-ACTIVE-POWER-PREDICTION.git)
cd WIND-TURBINE-ACTIVE-POWER-PREDICTION
pip install -r requirements.txt



DATASET:

https://www.kaggle.com/competitions/hill-of-towie-wind-turbine-power-prediction/overview 

Gerekli tüm kütüphaneler verisyon sabitlemeleriyle birlikte requirements.txt dosyasında mevcuttur.,

İndirmek için terminale yazın :  pip install -r requirements.txt

Teknik Rapor: 

Rüzgar Enerjisi Güç Tahmininde Mekansal Korelasyon ve Dijital İkiz Uygulaması
1. Proje Özeti ve Veri Mimarisi
Bu çalışma, İskoçya'daki Hill of Towie rüzgar santralinde bulunan 7 türbinlik bir filoda, sensör verisi eksik olan (Blind Prediction) Türbin 1 (T1)'in aktif güç üretimini tahmin etmeyi amaçlar.

Veri Hacmi: Yaklaşık 200.000 satırlık yüksek çözünürlüklü SCADA verisi.

Boyutluluk: Ham veri setinde yüze yakın sütun (öznitelik) bulunmakta olup, her türbin için AcWindSp, ActPower, NacelPos gibi operasyonel parametreler mevcuttur.

2. Preprocessing & Data Cleaning (Ön İşleme Stratejisi)
preprocessing.py dosyasında uyguladığımız işlemler, modelin "hile yapmasını" engellemek ve fiziksel tutarlılığı sağlamak üzerine kuruludur:

Data Leakage (Veri Sızıntısı) Önleme: T1'e ait tüm rüzgar hızı ve sıcaklık sensörleri (;1 etiketli sütunlar) veri setinden tamamen kazınmıştır. Model, hedef türbinin rüzgarını hiç görmeden tahmin yapmaya zorlanmıştır.

is_valid Filtrasyonu: Operasyonel olmayan (bakım, arıza, curtailment) durumları temsil eden hatalı kayıtlar ayıklanmıştır.

Robust Scaling: Veri setindeki aykırı değerlerin (outliers) etkisini minimize etmek için RobustScaler kullanılarak tüm özellikler normalleştirilmiştir.

3. Feature Engineering (Fiziksel Öznitelik Mühendisliği)
Sadece ham veriyi modele vermek yerine, türbin dinamiğini açıklayan Domain-Specific özellikler türetilmiştir:

Air Density (Hava Yoğunluğu) Hesabı: wind kütüphanesi ve fiziksel formüller kullanılarak ERA5 basınç ve sıcaklık verilerinden anlık hava yoğunluğu hesaplanmıştır ($ \rho = P / (R \cdot T) $). Hava yoğunluğu, güç eğrisindeki (Power Curve) kütlesel akışı belirleyen en kritik parametredir.

Wind Vectors (U & V Bileşenleri): Rüzgar yönü verisi periyodik olduğu için (0° ve 360° aynı yönü ifade eder), bu veriyi sinüs ve kosinüs bileşenlerine ayırarak modelin yönsel vektörleri anlaması sağlanmıştır.

Yaw Error Estimate: Komşu türbinlerin yönelimleri (Nacelle Position) ile ERA5 rüzgar yönü arasındaki fark analiz edilerek, türbinin rüzgarı ne kadar verimli karşıladığına dair bir "yönelim hatası" kestirimi eklenmiştir.

4. Model Architecture (Model Mimarisi)
model.py dosyasında, rüzgarın yüksek varyanslı yapısını yönetebilen hiyerarşik bir yapı kurulmuştur:

Base Learners (Zayıf Öğreniciler): Gradyan artırma algoritmalarından LGBM (hız ve doğruluk için) ve CatBoost (overfitting direnci için) seçilmiştir. Her iki model de 2000 iterasyonluk derin ağaçlarla eğitilmiştir.

Stacking Regressor: Tek bir modele güvenmek yerine, bu iki modelin tahminleri bir "Meta-Learner" (RidgeCV) aracılığıyla birleştirilmiştir. Bu sayede modellerin birbirlerinin hatalarını telafi etmesi sağlanmıştır.

Logarithmic Transformation: Aktif güç üretimi 0 ile 2315 kW arasında geniş bir yelpazeye dağıldığı için log1p dönüşümü yapılmış, tahmin aşamasında expm1 ile geri dönülmüştür. Bu işlem modelin düşük rüzgar hızlarındaki hassasiyetini artırmıştır.

5. Main Execution & Validation (Eğitim ve Doğrulama)
main.py dosyası tüm bu akışı yöneten orkestrasyon merkezidir:

5-Fold Cross Validation (CV): Veri 5 farklı parçaya bölünerek eğitilmiştir. Bu, modelin belirli bir zaman dilimine (örneğin sadece kış verisine) ezber yapmasını engelleyerek tüm mevsimsel döngülerde stabil kalmasını sağlar.

Inverse Log & Clipping: Tahmin edilen değerler logaritmik formdan kurtarıldıktan sonra, fiziksel imkansızlıkları önlemek adına np.clip(..., 0, None) ile negatif değerler sıfıra çekilmiştir.

6. Sonuç ve Metrik Analizi
Final Skoru: 50.36 MAE (Ortalama Mutlak Hata).

Bu değer, hedef türbinin rüzgar hızını (en temel girdiyi) hiç bilmeden, sadece komşu türbinlerin "ne hissettiğine" ve hava yoğunluğuna bakarak kurulan Dijital İkiz modelinin, gerçek üretimi %98'e yakın bir doğrulukla (anma gücü bazında) yakaladığını kanıtlamaktadır.
