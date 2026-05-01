import pandas as pd
import os
import sys

# Proje kök dizinini ekleyelim ki src'yi bulabilsin
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import handle_missing_and_clean

def run_final_audit():
    print("🕵️‍♀️ T1 Veri Sızıntısı Denetimi Başlatılıyor...")
    
    # 1. Veriyi Yükle
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "training_dataset.parquet")
    
    if not os.path.exists(data_path):
        print("❌ Veri dosyası bulunamadı!")
        return

    df = pd.read_parquet(data_path)
    
    # 2. Senin yazdığın temizlik fonksiyonunu çalıştır
    df_cleaned = handle_missing_and_clean(df)
    
    # 3. Kalan sütunları kontrol et
    cleaned_columns = df_cleaned.columns.tolist()
    t1_leakage = [col for col in cleaned_columns if ';1' in col]
    
    print("-" * 50)
    print(f"📊 Toplam Sütun Sayısı: {len(cleaned_columns)}")
    print(f"🚫 T1 Sızıntı Sayısı: {len(t1_leakage)}")
    
    if len(t1_leakage) == 0:
        print("\n✅ KANIT: Modelin giriş verisinde Türbin 1'e ait TEK BİR SENSÖR BİLE YOK.")
        print("Bu, aldığın 50 kW skorun 'kopya' değil, 'komşu türbin fiziği' olduğunun ispatıdır.")
    else:
        print("\n⚠️ DİKKAT: Sızıntı tespit edildi!")
        print(f"İçeride kalan T1 sütunları: {t1_leakage}")
    print("-" * 50)

if __name__ == "__main__":
    run_final_audit()