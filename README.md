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

Mühendislik Yorumu: Bu değer, hedef türbinin rüzgar hızını (en temel girdiyi) hiç bilmeden, sadece komşu türbinlerin "ne hissettiğine" ve hava yoğunluğuna bakarak kurulan Dijital İkiz modelinin, gerçek üretimi %98'e yakın bir doğrulukla (anma gücü bazında) yakaladığını kanıtlamaktadır.
