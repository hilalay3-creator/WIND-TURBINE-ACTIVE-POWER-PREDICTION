import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from .preprocessing import handle_missing_and_clean, get_preprocessor
from .model import get_baseline_scores, apply_log_transform, inverse_log_transform
from .evaluation import generate_linkedin_package

def run_wind_power_pipeline():
    print("🚀 Hill of Towie Rüzgar Tahmin Sistemi Başlatılıyor (Mühendislik Modu)...")

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "training_dataset.parquet")

    if not os.path.exists(data_path):
        print(f"❌ Hata: Veri bulunamadı -> {data_path}")
        return

    df = pd.read_parquet(data_path)
    
    # 1. Yarışma Filtresi: is_valid == True olanları kullan[cite: 1]
    if 'is_valid' in df.columns:
        df = df[df['is_valid'] == True].copy()
        print(f"📊 {len(df)} geçerli kayıtla işlemlere başlanıyor.")

    # 2. Hedef Sütun Ayarı
    target_col = 'target'
    if target_col not in df.columns:
        target_col = [c for c in df.columns if 'activepower' in c.lower() or 'target' in c.lower()][0]

    # 3. Özellik Mühendisliği & Temizlik
    df_processed = handle_missing_and_clean(df)
    
    y = df.loc[df_processed.index, target_col]
    y_log = apply_log_transform(y)
    X = df_processed

    # 4. Eğitim/Test Ayrımı
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    y_test_original = inverse_log_transform(y_test_log)

    # 5. Model Eğitim Süreci
    numeric_features = X.columns.tolist()
    preprocessor = get_preprocessor(numeric_features)

    model_scores, trained_pipelines = get_baseline_scores(
        X_train, X_test, y_train_log, y_test_original, preprocessor
    )

    # 6. Şampiyon Seçimi
    champion_name = max(model_scores, key=lambda x: model_scores[x]['R2'])
    champion_model = trained_pipelines[champion_name]
    
    print(f"\n🏆 ŞAMPİYON: {champion_name}")
    
    # UserWarning almamak için .values kullanıyoruz
    y_pred_log = champion_model.predict(X_test) 
    y_pred = np.clip(inverse_log_transform(y_pred_log), 0, None)

    # HATA BURADAYDI: y_test yerine y_test_original gönderiyoruz
    generate_linkedin_package(df, y_test_original, y_pred)

    # 7. Model Kaydetme
    model_dir = os.path.join(base_path, "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(champion_model, os.path.join(model_dir, "champion_wind_model.joblib"))
    print(f"💾 Model kaydedildi: models/champion_wind_model.joblib")

if __name__ == "__main__":
    run_wind_power_pipeline()