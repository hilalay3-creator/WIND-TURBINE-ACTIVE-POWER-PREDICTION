import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

def handle_missing_and_clean(df):
    df = df.copy()
    
    # 1. Fiziksel Özellikler (T1 verisi içermez, sadece ERA5 kullanır)[cite: 6]
    if 'ERA5_temperature_2m' in df.columns and 'ERA5_surface_pressure' in df.columns:
        df['air_density'] = df['ERA5_surface_pressure'] / (287.05 * (df['ERA5_temperature_2m'] + 273.15))

    if 'ERA5_wind_direction_100m' in df.columns and 'ERA5_wind_speed_100m' in df.columns:
        rad = np.deg2rad(df['ERA5_wind_direction_100m'])
        df['wind_u'] = df['ERA5_wind_speed_100m'] * np.cos(rad)
        df['wind_v'] = df['ERA5_wind_speed_100m'] * np.sin(rad)

    # 2. Northing (Kuzey) Tahmini (Sadece komşu türbinlerin yönelimini kullanır)
    nacelle_cols = [c for c in df.columns if 'NacelPos_mean' in c and ';1' not in c]
    if nacelle_cols:
        df['avg_neighbor_nacelle_pos'] = df[nacelle_cols].mean(axis=1)
        df['yaw_error_estimate'] = (df['ERA5_wind_direction_100m'] - df['avg_neighbor_nacelle_pos']) % 360

    # 3. 🛡️ T1'İ TAMAMEN SİLEN RADİKAL FİLTRE
    # İçinde ';1' geçen TÜM sütunları ve yarışma yasaklılarını atıyoruz.
    cols_to_drop = [col for col in df.columns if ';1' in col] # T1'e dair rüzgar, sıcaklık vs. HER ŞEY SİLİNİR
    
    # Ekstra kontrol: target, id, is_valid sütunlarını da silelim
    extra_forbidden = ['target', 'is_valid', 'id', 'TimeStamp_StartFormat']
    for col in extra_forbidden:
        if col in df.columns:
            cols_to_drop.append(col)

    print(f"🚫 T1'e dair TÜM sütunlar ({len([c for c in cols_to_drop if ';1' in c])} adet) imha edildi.")
    
    df_cleaned = df.drop(columns=list(set(cols_to_drop)), errors='ignore')
    return df_cleaned.select_dtypes(include=[np.number])

def get_preprocessor(numeric_features):
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])