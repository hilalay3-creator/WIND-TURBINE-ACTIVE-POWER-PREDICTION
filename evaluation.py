import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_linkedin_package(df, y_true, y_pred):
    """LinkedIn ve Kurumsal Sunumlar için Mühendislik Raporu Üretir."""
    sns.set_theme(style="whitegrid")
    
    # --- BOYUT HATASINI ÇÖZEN KISIM ---
    # Sadece test setine denk gelen satırları alıyoruz
    df_test = df.loc[y_true.index].copy() 
    
    # 1. GÖRSEL: ŞAMPİYONLUK KÜRSÜSÜ
        
   # --- 2. GÖRSEL: OPERASYONEL GÜVEN ---
    df_eval = pd.DataFrame({'abs_error': np.abs(y_true.values - y_pred)})
    wind_cols = [c for c in df_test.columns if 'AcWindSp' in c and ';1' not in c]
    df_eval['ref_wind'] = df_test[wind_cols].mean(axis=1).values
    
    bins = [0, 3, 5, 10, 15, 20, 25]
    labels = ['0-3', '3-5', '5-10', '10-15', '15-20', '20+']
    df_eval['wind_range'] = pd.cut(df_eval['ref_wind'], bins=bins, labels=labels)
    
    # Boş olan kategorileri atıyoruz 
    binned_mae = df_eval.groupby('wind_range', observed=True)['abs_error'].mean()
    actual_labels = binned_mae.index.tolist()
    
    plt.figure(figsize=(12, 6))
    # range(len(labels)) yerine range(len(binned_mae)) kullanıyoruz
    plt.plot(range(len(binned_mae)), binned_mae.values, marker='o', color='#E31837', linewidth=3)
    plt.fill_between(range(len(binned_mae)), binned_mae.values, alpha=0.1, color='#E31837')
    
    plt.xticks(range(len(binned_mae)), actual_labels) # Etiketleri de dinamik yaptık
    plt.title('Operasyonel Güven: Rüzgar Hızına Göre Tahmin Kararlılığı', fontsize=14)
    plt.ylabel('MAE (kW)')
    plt.grid(True, alpha=0.3)
    plt.savefig('visual_2_reliability.png', bbox_inches='tight', dpi=150)

    # 3. GÖRSEL: MEKANSAL KORELASYON
    power_cols = [c for c in df.columns if 'ActPower' in c]
    corr = df[power_cols].corr()['wtc_ActPower_mean;1'].sort_values(ascending=False).drop('wtc_ActPower_mean;1')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr.to_frame(), annot=True, cmap='YlGnBu', cbar=False)
    plt.title('Dijital İkiz Kanıtı: Türbinler Arası Fiziksel Bağ', fontsize=14)
    plt.savefig('visual_3_spatial.png', bbox_inches='tight', dpi=150)
    
    print("🚀 LinkedIn Görsel Paketi 'visual_*.png' adlarıyla hazırlandı!")