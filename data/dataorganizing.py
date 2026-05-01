import pandas as pd
import os

def rehber_olustur():
    print("⏳ Dosyalar taranıyor ve Türkçe rehber oluşturuluyor...")
    
    # 1. Dosya Yolları
    dosyalar = {
        'Training': 'data/training_dataset.parquet',
        'Submission': 'data/submission_dataset.parquet',
        'Sample': 'data/sample_model_submission.csv'
    }

    # 2. Teknik Terimlerin Türkçe Karşılık Sözlüğü
    sozluk = {
        "TimeStamp": "Zaman Damgası (10 dakikalık periyot başı)",
        "AcWindSp": "Anemometre Rüzgar Hızı (m/s)",
        "ScYawPos": "Gövde (Yaw) Yönelim Açısı (Derece)",
        "NacelPos": "Nacelle (Türbin Kafası) Pusula Yönü",
        "GenRpm": "Jeneratör Devir Sayısı (RPM)",
        "PitcPosA": "Kanat A (Pitch) Açısı",
        "PitcPosB": "Kanat B (Pitch) Açısı",
        "PitcPosC": "Kanat C (Pitch) Açısı",
        "PowerRef": "Kontrol Ünitesi Güç Referans Sınırı",
        "ScReToOp": "Operasyon Süresi (10 dk içindeki aktif saniye)",
        "ActPower": "Aktif Güç Üretimi (kW)",
        "AmbieTmp": "Ortam Sıcaklığı",
        "ShutdownDuration": "Duruş/Arıza Süresi",
        "ERA5_temperature_2m": "2 Metre Yükseklik Sıcaklığı",
        "ERA5_relative_humidity_2m": "2 Metre Bağıl Nem",
        "ERA5_dew_point_2m": "Çiğ Noktası Sıcaklığı",
        "ERA5_precipitation": "Yağış Miktarı",
        "ERA5_surface_pressure": "Yüzey Basıncı",
        "ERA5_cloud_cover": "Bulut Örtüsü Oranı (%)",
        "ERA5_wind_speed_10m": "10 Metre Yükseklik Rüzgar Hızı",
        "ERA5_wind_speed_100m": "100 Metre Yükseklik Rüzgar Hızı",
        "ERA5_wind_direction_10m": "10 Metre Rüzgar Yönü",
        "ERA5_wind_direction_100m": "100 Metre Rüzgar Yönü",
        "ERA5_wind_gusts_10m": "10 Metre Hamleli (Gust) Rüzgar Hızı",
        "mean": "Ortalama",
        "min": "Minimum",
        "max": "Maksimum",
        "stddev": "Standart Sapma (Dalgalanma)",
        "target": "HEDEF: T1 Aktif Güç (Tahmin Edilecek)",
        "is_valid": "Veri Geçerlilik Bayrağı (True/False)",
        "id": "Kaggle Kimlik Numarası"
    }

    tum_sutunlar = {}

    # 3. Dosyaları tara ve sütunları topla
    for isim, yol in dosyalar.items():
        if os.path.exists(yol):
            if yol.endswith('.parquet'):
                df = pd.read_parquet(yol).head(0) # Sadece başlığı oku
            else:
                df = pd.read_csv(yol, nrows=0)
            
            for col in df.columns:
                if col not in tum_sutunlar:
                    tum_sutunlar[col] = []
                tum_sutunlar[col].append(isim)
        else:
            print(f"⚠️ Uyarı: {yol} dosyası bulunamadı, atlanıyor.")

    # 4. Excel için veri hazırla
    data_list = []
    for sutun, dosyalar_list in tum_sutunlar.items():
        # Türkçe açıklama oluşturma mantığı
        aciklama = "Tanımlanamayan Teknik Veri"
        temiz_isim = sutun.replace(";", "_")
        
        anlam_parcalari = []
        for anahtar, deger in sozluk.items():
            if anahtar in temiz_isim:
                anlam_parcalari.append(deger)
        
        if anlam_parcalari:
            aciklama = " - ".join(list(dict.fromkeys(anlam_parcalari))) # Tekrarları silerek birleştir

        # Türbin ID ekle
        if ";" in sutun:
            t_id = sutun.split(";")[-1]
            aciklama = f"Türbin {t_id} | {aciklama}"

        data_list.append({
            "Sütun Adı": sutun,
            "Türkçe Açıklama": aciklama,
            "Bulunduğu Dosyalar": ", ".join(dosyalar_list)
        })

    # 5. Excel'e Kaydet
    final_df = pd.DataFrame(data_list)
    final_df.to_excel("3_Data_Sutun_Rehberi.xlsx", index=False)
    print("\n✅ İŞLEM TAMAM! '3_Data_Sutun_Rehberi.xlsx' dosyası oluşturuldu.")

if __name__ == "__main__":
    rehber_olustur()