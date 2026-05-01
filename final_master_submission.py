import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import KFold
from src.preprocessing import handle_missing_and_clean, get_preprocessor
from src.model import apply_log_transform, inverse_log_transform
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

def run_final_cv5_submission():
    print("🚀 Kaggle Final Süreci Başlatılıyor (CV=5 Mode)...")

    # 1. VERİLERİ YÜKLE
    train_df = pd.read_parquet("data/training_dataset.parquet")
    sub_df = pd.read_parquet("data/submission_dataset.parquet")
    
    if 'is_valid' in train_df.columns:
        train_df = train_df[train_df['is_valid'] == True].copy()

    # 2. ÖN İŞLEME
    X_train_raw = handle_missing_and_clean(train_df)
    y_train = train_df.loc[X_train_raw.index, 'target']
    y_train_log = apply_log_transform(y_train)
    
    # Sadece modelin beklediği sütunları al (Inference güvenliği için)
    X_sub_raw = handle_missing_and_clean(sub_df)
    features = X_train_raw.columns.tolist()
    X_sub = X_sub_raw[features]

    # 3. CV=5 İLE MODEL EĞİTİMİ (OOF - Out of Fold Yöntemi)
    # Yarışmalarda en sağlam yöntem, 5 farklı fold'un tahminlerini ortalamaktır.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    final_predictions = np.zeros(len(X_sub))
    
    preprocessor = get_preprocessor(features)
    
    print(f"🌀 5-Fold Cross Validation başlıyor...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_raw), 1):
        print(f"📍 Fold {fold} eğitiliyor...")
        
        X_tr, X_val = X_train_raw.iloc[train_idx], X_train_raw.iloc[val_idx]
        y_tr_log, y_val_log = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
        
        # Şampiyon Model: LGBM (Daha önce 50kW veren ayarlar)
        model = LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=127, random_state=42, verbose=-1)
        pipe = Pipeline([('pre', preprocessor), ('reg', model)])
        
        pipe.fit(X_tr, y_tr_log)
        
        # Test (submission) verisi için tahminleri biriktir
        fold_pred_log = pipe.predict(X_sub)
        final_predictions += inverse_log_transform(fold_pred_log) / 5 # Ortalamasını alıyoruz

    # 4. SON DOKUNUŞLAR
    final_predictions = np.clip(final_predictions, 0, None)

    # 5. KAGGE FORMATINA YAZMA (Sample Submission Uyumu)
    submission_df = pd.DataFrame({
        'id': sub_df['id'],
        'target': final_predictions
    })

    output_path = "submission_hilal_ay_cv5.csv"
    submission_df.to_csv(output_path, index=False)
    
    # Modeli de en son haliyle saklayalım
    joblib.dump(pipe, "models/final_cv5_model.joblib")
    
    print(f"\n✅ BAŞARILI! '{output_path}' dosyası hazır.")
    print(f"📊 Ortalama Tahmin: {final_predictions.mean():.2f} kW")

if __name__ == "__main__":
    run_final_cv5_submission()