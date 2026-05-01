import joblib
import pandas as pd
import numpy as np
import os
from src.evaluation import generate_linkedin_package
from src.model import inverse_log_transform
from src.preprocessing import handle_missing_and_clean
from sklearn.model_selection import train_test_split

def quick_generate():
    print("⚡ Hızlı Görselleştirme Başlatılıyor...")
    
    # 1. Veriyi Yükle
    df = pd.read_parquet("data/training_dataset.parquet")
    if 'is_valid' in df.columns:
        df = df[df['is_valid'] == True].copy()

    # 2. Modeli Yükle
    model_path = "models/champion_wind_model.joblib"
    if not os.path.exists(model_path):
        print("❌ Hata: Kayıtlı model bulunamadı!")
        return
    model = joblib.load(model_path)

    # 3. Veri Hazırlama (Özellikleri Hizalıyoruz)
    df_processed = handle_missing_and_clean(df)
    
    # Modelin beklediği sütunları pipeline'dan çekiyoruz
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
    else:
        # Pipeline içinde model en sondaysa ona erişiyoruz
        expected_features = model.steps[-1][1].feature_names_in_

    # Veriyi sadece modelin tanıdığı sütunlarla kısıtlıyoruz
    X = df_processed[expected_features]
    y = df.loc[df_processed.index, 'target']
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    y_test_original = inverse_log_transform(y_test_log)

    # 4. Tahmin Al (Saniyeler sürer)
    print("🔮 Tahminler üretiliyor...")
    y_pred_log = model.predict(X_test)
    y_pred = np.clip(inverse_log_transform(y_pred_log), 0, None)

    # 5. Grafikleri Oluştur
    print("📊 Grafikler çiziliyor...")
    generate_linkedin_package(df, y_test_original, y_pred)
    
    print("✅ İŞLEM TAMAM! visual_1, visual_2 ve visual_3 dosyaları hazır.")

if __name__ == "__main__":
    quick_generate()