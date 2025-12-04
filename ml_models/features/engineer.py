"""
PTF Tahmin Projesi - Öznitelik Mühendisliği Modülü
==================================================
Raporun vurguladığı önem sırasına göre öznitelikler:

1. Otoregresif Özellikler (Lag'ler) - En kritik
2. Talep/Yük Özellikleri
3. Yakıt Maliyetleri (Dolar kuru, doğalgaz)
4. Yenilenebilir Enerji (Rüzgar, Güneş)
5. Takvim Etkileri (Tatiller, Haftasonu)
6. Sistem Durumu (SMF, Dengesizlik)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    PTF tahminlemesi için öznitelik mühendisliği sınıfı.
    
    72 saatlik tahmin ufku için optimize edilmiştir.
    """
    
    # Türkiye'deki resmi tatiller (güncellenebilir)
    HOLIDAYS_2024 = [
        # Yılbaşı
        "2024-01-01",
        # Ramazan Bayramı (2024)
        "2024-04-10", "2024-04-11", "2024-04-12",
        # Ulusal Egemenlik ve Çocuk Bayramı
        "2024-04-23",
        # İşçi Bayramı
        "2024-05-01",
        # Atatürk'ü Anma, Gençlik ve Spor Bayramı
        "2024-05-19",
        # Kurban Bayramı (2024)
        "2024-06-16", "2024-06-17", "2024-06-18", "2024-06-19",
        # Demokrasi ve Milli Birlik Günü
        "2024-07-15",
        # Zafer Bayramı
        "2024-08-30",
        # Cumhuriyet Bayramı
        "2024-10-29",
    ]
    
    def __init__(
        self, 
        lag_hours: List[int] = None,
        rolling_windows: List[int] = None,
        prediction_horizon: int = 72
    ):
        """
        Args:
            lag_hours: Gecikme saatleri listesi
            rolling_windows: Hareketli ortalama pencere boyutları
            prediction_horizon: Tahmin ufku (saat)
        """
        # Varsayılan lag'ler - rapordaki önerilere göre
        self.lag_hours = lag_hours or [24, 48, 72, 168, 336]  # 168=1 hafta, 336=2 hafta
        self.rolling_windows = rolling_windows or [6, 12, 24, 48, 168]
        self.prediction_horizon = prediction_horizon
        
        # Türkiye tatillerini datetime'a çevir
        self.holidays = pd.to_datetime(self.HOLIDAYS_2024)
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm öznitelikleri ÖNEM SIRASINA GÖRE oluşturur.
        
        ÖNEM SIRASI (Rapor'dan):
        1. Otoregresif Fiyatlar (Lag'ler) - %30-40 açıklama gücü
        2. Talep (Yük) Tahminleri - Piyasa yönü
        3. Marjinal Yakıt Maliyetleri - Fiyat tabanı
        4. Yenilenebilir Enerji - Merit Order Effect
        5. Takvim ve Zaman - Davranışsal kalıplar
        6. Sistem Durumu (SMF) - Risk primi
        7. Hidroloji ve Barajlar - Fırsat maliyeti (şimdilik yok)
        8. Santral Yararlanılabilirliği (şimdilik yok)
        
        Args:
            df: Ham veri (datetime index, 'ptf' kolonu zorunlu)
            
        Returns:
            DataFrame: Özniteliklerle zenginleştirilmiş veri
        """
        df = df.copy()
        
        # Index'in datetime olduğundan emin ol
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index'i DatetimeIndex olmalıdır!")
        
        print("🔧 Öznitelik mühendisliği başlıyor...\n")
        print("   (Önem sırasına göre oluşturuluyor)\n")
        
        # 0. VERİ TİPLERİNİ TEMİZLE
        print("0️⃣  Veri tipleri kontrol ediliyor...")
        df = self._clean_data_types(df)
        
        # ═══════════════════════════════════════════════════════════════
        # 1. OTOREGRESİF ÖZELLİKLER (EN KRİTİK - %30-40 açıklama gücü)
        # ═══════════════════════════════════════════════════════════════
        print("1️⃣  [ÖNEM:1] Otoregresif özellikler (Lag'ler)...")
        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        
        # ═══════════════════════════════════════════════════════════════
        # 2. TALEP (YÜK) ÖZELLİKLERİ - Piyasa yönü
        # ═══════════════════════════════════════════════════════════════
        if 'load_forecast' in df.columns:
            print("2️⃣  [ÖNEM:2] Yük özellikleri...")
            df = self._create_load_features(df)
        
        # ═══════════════════════════════════════════════════════════════
        # 3. MARJİNAL YAKIT MALİYETLERİ - Fiyat tabanı
        # ═══════════════════════════════════════════════════════════════
        print("3️⃣  [ÖNEM:3] Yakıt maliyeti özellikleri...")
        df = self._create_fuel_cost_features(df)
        
        # ═══════════════════════════════════════════════════════════════
        # 4. YENİLENEBİLİR ENERJİ - Merit Order Effect
        # ═══════════════════════════════════════════════════════════════
        print("4️⃣  [ÖNEM:4] Yenilenebilir enerji özellikleri...")
        df = self._create_renewable_features(df)
        df = self._create_residual_load_features(df)
        
        # ═══════════════════════════════════════════════════════════════
        # 5. TAKVİM VE ZAMAN ETKİSİ - Davranışsal kalıplar
        # ═══════════════════════════════════════════════════════════════
        print("5️⃣  [ÖNEM:5] Takvim özellikleri...")
        df = self._create_calendar_features(df)
        df = self._create_cyclical_features(df)
        
        # ═══════════════════════════════════════════════════════════════
        # 6. SİSTEM DURUMU (BALANS) - Risk primi
        # ═══════════════════════════════════════════════════════════════
        if 'smf' in df.columns:
            print("6️⃣  [ÖNEM:6] Sistem durumu özellikleri...")
            df = self._create_system_features(df)
        
        # ═══════════════════════════════════════════════════════════════
        # 7. HEDEF DEĞİŞKEN
        # ═══════════════════════════════════════════════════════════════
        print("7️⃣  Hedef değişken oluşturuluyor...")
        df = self._create_target_variable(df)
        
        print(f"\n✅ Toplam {len(df.columns)} öznitelik oluşturuldu")
        
        return df
    
    # =========================================================================
    # 3. YAKIT MALİYETİ ÖZELLİKLERİ (YENİ!)
    # =========================================================================
    
    def _create_fuel_cost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Yakıt maliyeti bazlı özellikler.
        
        - Dolar kuru etkisi
        - Spark Spread (Gaz karlılığı)
        - Dark Spread (Kömür karlılığı)
        
        Bu özellikler fiyatın TABANINI belirler.
        """
        # Dolar kuru varsa
        usd_cols = [c for c in df.columns if 'usd' in c.lower()]
        if usd_cols:
            usd_col = usd_cols[0]
            usd = df[usd_col]
            
            # Kur değişimi
            df['usd_change_1d'] = usd.diff(1)
            df['usd_change_7d'] = usd.diff(7)
            df['usd_pct_change_7d'] = np.where(
                usd.shift(7).abs() > 0.1,
                (usd - usd.shift(7)) / usd.shift(7) * 100,
                0
            )
            
            # Kur hareketli ortalaması
            df['usd_ma_7d'] = usd.rolling(7).mean()
            df['usd_ma_30d'] = usd.rolling(30).mean()
            
            # Kur volatilitesi
            df['usd_volatility'] = usd.rolling(7).std()
            
            print(f"    ✓ Dolar kuru özellikleri eklendi")
        
        # Spark Spread varsa
        if 'spark_spread' in df.columns:
            spark = df['spark_spread']
            
            df['spark_spread_ma_7d'] = spark.rolling(7).mean()
            df['spark_spread_positive'] = (spark > 0).astype(int)
            
            print(f"    ✓ Spark Spread özellikleri eklendi")
        
        # Dark Spread varsa
        if 'dark_spread' in df.columns:
            dark = df['dark_spread']
            
            df['dark_spread_ma_7d'] = dark.rolling(7).mean()
            df['dark_spread_positive'] = (dark > 0).astype(int)
            
            print(f"    ✓ Dark Spread özellikleri eklendi")
        
        # Gaz maliyeti varsa
        if 'gas_input_cost' in df.columns:
            gas = df['gas_input_cost']
            df['gas_cost_ma_7d'] = gas.rolling(7).mean()
            df['gas_cost_change_7d'] = gas.diff(7)
        
        # Kömür maliyeti varsa  
        if 'coal_input_cost' in df.columns:
            coal = df['coal_input_cost']
            df['coal_cost_ma_7d'] = coal.rolling(7).mean()
            df['coal_cost_change_7d'] = coal.diff(7)
        
        return df
    
    # =========================================================================
    # 4. RESIDUAL LOAD (NET YÜK) ÖZELLİKLERİ (YENİ!)
    # =========================================================================
    
    def _create_residual_load_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Residual Load (Net Yük) özellikleri.
        
        Residual Load = Toplam Yük - (Rüzgar + Güneş)
        
        Bu değer termik santrallerin karşılaması gereken yükü gösterir.
        Merit order'da fiyatı DOĞRUDAN belirleyen budur.
        
        Residual Load Squared: Marjinal maliyet eğrisi lineer değil,
        KARESEL artar. Bu nedenle karesel terim çok önemli.
        """
        # Yük kolonu bul
        load = None
        load_cols = [c for c in df.columns if 'load' in c.lower() and 'residual' not in c.lower()]
        for col in load_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                load = df[col]
                break
        
        if load is None:
            return df
        
        # Yenilenebilir üretim topla
        renewable = pd.Series(0, index=df.index)
        
        # Rüzgar
        wind_cols = [c for c in df.columns if 'wind' in c.lower() and 
                    not any(x in c.lower() for x in ['ma', 'lag', 'change', 'variability'])]
        for col in wind_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                renewable = renewable + df[col].fillna(0)
                break
        
        # Güneş
        solar_cols = [c for c in df.columns if 'solar' in c.lower() and 
                     not any(x in c.lower() for x in ['ma', 'lag', 'change'])]
        for col in solar_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                renewable = renewable + df[col].fillna(0)
                break
        
        # === RESIDUAL LOAD ===
        df['residual_load'] = load - renewable
        
        # === RESIDUAL LOAD SQUARED (ÇOK ÖNEMLİ!) ===
        # Normalize et (GW cinsine) ki sayılar çok büyük olmasın
        residual_gw = df['residual_load'] / 1000
        df['residual_load_squared'] = residual_gw ** 2
        
        # Cubic term (opsiyonel - aşırı yüksek yüklerde)
        df['residual_load_cubed'] = residual_gw ** 3
        
        # Residual Load değişimi
        df['residual_load_change_24h'] = df['residual_load'].diff(24)
        
        # Residual Load / Toplam Yük oranı (Yenilenebilir penetrasyonu)
        df['renewable_penetration'] = np.where(
            load.abs() > 100,
            renewable / load,
            0
        )
        
        # Residual Load seviyeleri (kategorik)
        df['residual_load_level'] = pd.cut(
            df['residual_load'],
            bins=[0, 25000, 30000, 35000, 40000, 45000, np.inf],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float)
        
        # Yüksek yük flag (Residual > 40GW = fiyat spike riski)
        df['high_residual_load'] = (df['residual_load'] > 40000).astype(int)
        
        print(f"    ✓ Residual Load özellikleri eklendi")
        print(f"      Ort Residual: {df['residual_load'].mean():.0f} MW")
        print(f"      Ort Yenilenebilir Penetrasyon: {df['renewable_penetration'].mean()*100:.1f}%")
        
        return df
    
    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm kolonların veri tiplerini kontrol eder ve düzeltir.
        
        - Tarih kolonlarını KALDIRIR (XGBoost işleyemez)
        - Sayısal olması gereken kolonları float'a çevirir
        - String kolonları temizler
        """
        cols_to_drop = []
        
        for col in df.columns:
            col_lower = col.lower()
            dtype = df[col].dtype
            
            # Datetime kolonlarını işaretle
            if 'datetime' in str(dtype) or dtype.name.startswith('datetime'):
                cols_to_drop.append(col)
                print(f"    ⚠ '{col}' datetime kolonu kaldırılacak")
                continue
            
            # Tarih/saat içerebilecek kolon isimlerini kontrol et
            datetime_patterns = ['date', 'time', 'tarih', 'saat', 'period', 'timestamp']
            is_datetime_col = any(p in col_lower for p in datetime_patterns)
            
            if is_datetime_col and col not in ['hour', 'day_of_week', 'day_of_month', 'month', 'year', 'week_of_year']:
                # Bu bir datetime kolonu, kaldır
                cols_to_drop.append(col)
                print(f"    ⚠ '{col}' tarih kolonu kaldırılacak")
                continue
            
            # Object tipindeki kolonları sayısala çevirmeye çalış
            if dtype == 'object':
                try:
                    # Önce tarih olup olmadığını kontrol et
                    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if sample and isinstance(sample, str) and ('T' in sample or '-' in sample):
                        try:
                            pd.to_datetime(sample)
                            cols_to_drop.append(col)
                            print(f"    ⚠ '{col}' tarih string kolonu kaldırılacak")
                            continue
                        except:
                            pass
                    
                    # Sayısala çevirmeyi dene
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"    ✓ '{col}' sayısala çevrildi")
                except Exception as e:
                    print(f"    ⚠ '{col}' çevrilemedi: {e}")
                    cols_to_drop.append(col)
        
        # Kolonları kaldır
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors='ignore')
            print(f"    → {len(cols_to_drop)} kolon kaldırıldı")
        
        return df
    
    # =========================================================================
    # 1. OTOREGRESIF ÖZELLİKLER
    # =========================================================================
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gecikmeli (lag) özellikler oluşturur.
        
        Önem: En yüksek - SHAP analizlerinde %30-40 açıklama gücü
        
        72 saatlik tahmin için, en az 72 saat önceki veriler kullanılmalı!
        """
        # PTF Lag'leri
        if 'ptf' in df.columns:
            for lag in self.lag_hours:
                # 72 saat sonrasını tahmin edeceğimiz için,
                # lag + 72 kadar geriye gitmemiz gerekiyor
                effective_lag = lag + self.prediction_horizon
                df[f'ptf_lag_{lag}h'] = df['ptf'].shift(effective_lag)
            
            # Özel lag'ler
            # Dünün aynı saati (gün öncesi piyasası için kritik)
            df['ptf_lag_24h'] = df['ptf'].shift(24 + self.prediction_horizon)
            
            # Geçen haftanın aynı saati (haftalık döngü)
            df['ptf_lag_168h'] = df['ptf'].shift(168 + self.prediction_horizon)
            
            # Bir önceki günün ortalama fiyatı
            df['ptf_prev_day_mean'] = df['ptf'].shift(self.prediction_horizon).rolling(24).mean()
            
            # Bir önceki günün max/min fiyatı
            df['ptf_prev_day_max'] = df['ptf'].shift(self.prediction_horizon).rolling(24).max()
            df['ptf_prev_day_min'] = df['ptf'].shift(self.prediction_horizon).rolling(24).min()
            
            # Geçen haftanın aynı günü ortalama
            df['ptf_prev_week_same_day'] = df['ptf'].shift(168 + self.prediction_horizon).rolling(24).mean()
            
            # ═══════════════════════════════════════════════════════════════
            # LOG TRANSFORM - Fiyat dağılımını normalize et
            # ═══════════════════════════════════════════════════════════════
            ptf_shifted = df['ptf'].shift(self.prediction_horizon)
            
            # Log fiyat (daha normal dağılım sağlar, yüksek fiyat spike'larını yumuşatır)
            df['ptf_log'] = np.log1p(ptf_shifted.clip(lower=1))
            
            # Log fiyat lag'leri
            df['ptf_log_lag_24h'] = df['ptf_log'].shift(24)
            df['ptf_log_lag_168h'] = df['ptf_log'].shift(168)
            
            # Log fiyat hareketli ortalaması
            df['ptf_log_ma_24h'] = df['ptf_log'].rolling(24).mean()
            df['ptf_log_ma_168h'] = df['ptf_log'].rolling(168).mean()
            
            print(f"    ✓ Log transform özellikleri eklendi")
            
            # ═══════════════════════════════════════════════════════════════
            # TREND ÖZELLİKLERİ - Fiyat yönünü yakala (ÇOK ÖNEMLİ!)
            # ═══════════════════════════════════════════════════════════════
            
            # 7 günlük trend (kısa vadeli)
            ma_7d = ptf_shifted.rolling(24 * 7).mean()
            ma_3d = ptf_shifted.rolling(24 * 3).mean()
            ma_7d_lagged = ma_7d.shift(24 * 7)
            df['trend_7d'] = np.where(
                ma_7d_lagged.abs() > 1,
                (ma_7d - ma_7d_lagged) / ma_7d_lagged * 100,
                0
            )
            
            # 30 günlük trend (orta vadeli) - TREND YAKALAMAK İÇİN KRİTİK
            ma_30d = ptf_shifted.rolling(24 * 30).mean()
            ma_30d_lagged = ma_30d.shift(24 * 30)
            df['trend_30d'] = np.where(
                ma_30d_lagged.abs() > 1,
                (ma_30d - ma_30d_lagged) / ma_30d_lagged * 100,
                0
            )
            
            # Trend yönü (binary flags)
            df['trend_up_7d'] = (df['trend_7d'] > 0).astype(int)
            df['trend_up_30d'] = (df['trend_30d'] > 0).astype(int)
            
            # Trend gücü (momentum) - Yüksek değer = güçlü trend
            df['trend_strength'] = df['trend_7d'].abs() + df['trend_30d'].abs()
            
            # MA crossover (kısa MA > uzun MA = yükseliş sinyali)
            df['ma_crossover'] = (ma_3d > ma_7d).astype(int)
            df['ma_crossover_30d'] = (ma_7d > ma_30d).astype(int)
            
            # Fiyat pozisyonu (mevcut fiyat / 30 günlük ortalama)
            # > 1 = ortalamanın üstünde, < 1 = altında
            df['price_position_30d'] = np.where(
                ma_30d.abs() > 1,
                ptf_shifted / ma_30d,
                1
            )
            
            print(f"    ✓ Trend özellikleri eklendi (7d, 30d)")
            
            # ═══════════════════════════════════════════════════════════════
            # RECENCY WEIGHTING - Son verilere daha fazla ağırlık ver
            # (Model son dönem fiyat seviyesini yakalasın)
            # ═══════════════════════════════════════════════════════════════
            
            # Exponential Weighted Moving Average - Son verilere üssel ağırlık
            df['ptf_ewm_7d'] = ptf_shifted.ewm(span=24*7, adjust=False).mean()
            df['ptf_ewm_30d'] = ptf_shifted.ewm(span=24*30, adjust=False).mean()
            df['ptf_ewm_90d'] = ptf_shifted.ewm(span=24*90, adjust=False).mean()
            
            # EWM farkı (kısa vadeli vs uzun vadeli trend)
            df['ewm_diff_7d_30d'] = df['ptf_ewm_7d'] - df['ptf_ewm_30d']
            df['ewm_diff_30d_90d'] = df['ptf_ewm_30d'] - df['ptf_ewm_90d']
            
            # Recency ratio - Son 3 gün / Son 30 gün
            ma_3d_clean = ptf_shifted.rolling(24 * 3).mean()
            ma_30d_clean = ptf_shifted.rolling(24 * 30).mean()
            df['recency_ratio'] = np.where(
                ma_30d_clean.abs() > 1,
                ma_3d_clean / ma_30d_clean,
                1
            )
            
            # Son 7 gün ortalaması (güncel seviye göstergesi)
            df['ptf_recent_7d_mean'] = ptf_shifted.rolling(24 * 7).mean()
            
            # Fiyat ivmesi (acceleration) - Trend değişim hızı
            trend_7d_shifted = df['trend_7d'].shift(24 * 7)
            df['price_acceleration'] = df['trend_7d'] - trend_7d_shifted.fillna(0)
            
            print(f"    ✓ Recency weighting özellikleri eklendi (EWM)")
        
        # SMF Lag'leri (varsa)
        if 'smf' in df.columns:
            for lag in [24, 48, 168]:
                effective_lag = lag + self.prediction_horizon
                df[f'smf_lag_{lag}h'] = df['smf'].shift(effective_lag)
        
        return df
    
    # =========================================================================
    # 2. HAREKETLİ İSTATİSTİKLER
    # =========================================================================
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hareketli ortalama, standart sapma ve diğer istatistikler.
        
        Momentum ve volatilite göstergeleri.
        """
        if 'ptf' not in df.columns:
            return df
        
        # Shift edilmiş PTF (tahmin anında bilinen değerler)
        ptf_known = df['ptf'].shift(self.prediction_horizon)
        
        for window in self.rolling_windows:
            # Hareketli ortalama
            df[f'ptf_ma_{window}h'] = ptf_known.rolling(window).mean()
            
            # Hareketli standart sapma (volatilite)
            df[f'ptf_std_{window}h'] = ptf_known.rolling(window).std()
            
            # Hareketli min/max
            df[f'ptf_min_{window}h'] = ptf_known.rolling(window).min()
            df[f'ptf_max_{window}h'] = ptf_known.rolling(window).max()
        
        # Fiyat momentum göstergeleri
        # Son 24 saatteki değişim
        df['ptf_change_24h'] = ptf_known.diff(24)
        
        # pct_change güvenli hesaplama
        ptf_24h_ago = ptf_known.shift(24)
        df['ptf_pct_change_24h'] = np.where(
            ptf_24h_ago.abs() > 1,
            (ptf_known - ptf_24h_ago) / ptf_24h_ago * 100,
            0
        )
        
        # Son 168 saatteki değişim (haftalık)
        df['ptf_change_168h'] = ptf_known.diff(168)
        
        ptf_168h_ago = ptf_known.shift(168)
        df['ptf_pct_change_168h'] = np.where(
            ptf_168h_ago.abs() > 1,
            (ptf_known - ptf_168h_ago) / ptf_168h_ago * 100,
            0
        )
        
        # Volatilite oranı (CV - Coefficient of Variation)
        # INF önlemek için mean'in çok küçük olduğu yerleri kontrol et
        mean_24h = df['ptf_ma_24h']
        std_24h = df['ptf_std_24h']
        df['ptf_cv_24h'] = np.where(
            mean_24h.abs() > 1,  # Mean 1'den büyükse hesapla
            std_24h / mean_24h,
            0  # Yoksa 0 koy
        )
        
        return df
    
    # =========================================================================
    # 3. TAKVİM ÖZELLİKLERİ
    # =========================================================================
    
    def _create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tarih/saat bazlı özellikler.
        
        Türkiye piyasasına özgü tatiller ve mevsimsellik.
        """
        # Temel zaman özellikleri
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek  # 0=Pazartesi, 6=Pazar
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Binary özellikler
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Saat dilimleri (piyasa döngüsü)
        df['is_peak_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 20)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
        
        # Mevsimler
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['is_winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(int)
        
        # Resmi tatiller
        df['is_holiday'] = df.index.normalize().isin(self.holidays).astype(int)
        
        # Tatil öncesi/sonrası (köprü günleri)
        holiday_dates = self.holidays.normalize()
        df['is_day_before_holiday'] = df.index.normalize().isin(
            holiday_dates - pd.Timedelta(days=1)
        ).astype(int)
        df['is_day_after_holiday'] = df.index.normalize().isin(
            holiday_dates + pd.Timedelta(days=1)
        ).astype(int)
        
        # Ayın başı/sonu (fatura dönemleri)
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        
        return df
    
    # =========================================================================
    # 4. DÖNGÜSEL KODLAMA
    # =========================================================================
    
    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Saat ve gün gibi döngüsel değişkenleri sin/cos ile kodlar.
        
        Önemli: Saat 23 ile 0 birbirine yakın olmalı!
        """
        # Saat (24 saatlik döngü)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Haftanın günü (7 günlük döngü)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Ay (12 aylık döngü)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Yılın günü (365 günlük döngü - mevsimsellik)
        day_of_year = df.index.dayofyear
        df['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        df['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        
        return df
    
    # =========================================================================
    # 5. SİSTEM DURUMU ÖZELLİKLERİ
    # =========================================================================
    
    def _create_system_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SMF bazlı sistem dengesizlik özellikleri.
        
        SMF > PTF → Sistem enerji açığı (yukarı baskı)
        SMF < PTF → Sistem enerji fazlası (aşağı baskı)
        """
        if 'smf' not in df.columns or 'ptf' not in df.columns:
            return df
        
        # PTF-SMF Spread (Shift edilmiş - tahmin anında bilinen)
        ptf_known = df['ptf'].shift(self.prediction_horizon)
        smf_known = df['smf'].shift(self.prediction_horizon)
        
        df['ptf_smf_spread'] = ptf_known - smf_known
        
        # Ratio hesabı - inf önleme
        df['ptf_smf_ratio'] = np.where(
            smf_known.abs() > 1,
            ptf_known / smf_known,
            1  # SMF çok küçükse oran 1 kabul et
        )
        
        # Sistem yönü göstergeleri
        df['system_long'] = (smf_known < ptf_known).astype(int)  # Enerji fazlası
        df['system_short'] = (smf_known > ptf_known).astype(int)  # Enerji açığı
        
        # Son 24 saatte sistem yönü
        df['system_short_count_24h'] = df['system_short'].rolling(24).sum()
        df['system_long_count_24h'] = df['system_long'].rolling(24).sum()
        
        # SMF volatilitesi
        df['smf_std_24h'] = smf_known.rolling(24).std()
        df['smf_ma_24h'] = smf_known.rolling(24).mean()
        
        return df
    
    # =========================================================================
    # 6. YÜK ÖZELLİKLERİ
    # =========================================================================
    
    def _create_load_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Yük tahmini bazlı özellikler.
        
        Yük ile PTF arasında üstel (exponential) ilişki var!
        """
        if 'load_forecast' not in df.columns:
            return df
        
        load = df['load_forecast']
        
        # Yük istatistikleri
        df['load_ma_24h'] = load.rolling(24).mean()
        df['load_ma_168h'] = load.rolling(168).mean()
        df['load_std_24h'] = load.rolling(24).std()
        
        # Yük değişimi
        df['load_change_24h'] = load.diff(24)
        
        # pct_change inf üretebilir, güvenli hesapla
        load_24h_ago = load.shift(24)
        df['load_pct_change_24h'] = np.where(
            load_24h_ago.abs() > 100,
            (load - load_24h_ago) / load_24h_ago * 100,
            0
        )
        
        # Yük seviyeleri (kategorik - lineer olmayan ilişki için)
        # Türkiye için tipik yük aralıkları
        df['load_level'] = pd.cut(
            load,
            bins=[0, 30000, 35000, 40000, 45000, 50000, np.inf],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float)
        
        # Yük peak flag
        load_mean = load.rolling(168).mean()
        load_std = load.rolling(168).std()
        df['load_is_high'] = (load > load_mean + load_std).astype(int)
        df['load_is_low'] = (load < load_mean - load_std).astype(int)
        
        return df
    
    # =========================================================================
    # 7. YENİLENEBİLİR ENERJİ ÖZELLİKLERİ
    # =========================================================================
    
    def _create_renewable_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rüzgar ve güneş üretimi özellikleri.
        
        Merit Order Effect: Yenilenebilir ↑ → PTF ↓
        """
        # Rüzgar özellikleri
        wind_cols = [c for c in df.columns if 'wind' in c.lower() or 'ruzgar' in c.lower()]
        
        # Sayısal rüzgar kolonu bul
        wind_numeric_col = None
        for col in wind_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                wind_numeric_col = col
                break
            # Object tipindeyse sayısala çevirmeyi dene
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > 0:
                    wind_numeric_col = col
                    break
            except:
                continue
        
        # Alternatif: 'forecast' veya 'generation' kolonlarını ara
        if wind_numeric_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if ('forecast' in col_lower or 'generation' in col_lower) and 'wind' not in col_lower:
                    if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        wind_numeric_col = col
                        break
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if df[col].notna().sum() > 0:
                            wind_numeric_col = col
                            break
                    except:
                        continue
        
        if wind_numeric_col is not None:
            wind = df[wind_numeric_col].shift(self.prediction_horizon)
            
            df['wind_ma_24h'] = wind.rolling(24).mean()
            df['wind_ma_168h'] = wind.rolling(168).mean()
            df['wind_change_24h'] = wind.diff(24)
            
            # Variability hesabı - inf önleme
            wind_mean = wind.rolling(24).mean()
            wind_std = wind.rolling(24).std()
            df['wind_variability'] = np.where(
                wind_mean.abs() > 1,
                wind_std / wind_mean,
                0
            )
            
            print(f"    ✓ Rüzgar özellikleri oluşturuldu: {wind_numeric_col}")
        else:
            print("    ⚠ Rüzgar verisi bulunamadı veya sayısal değil")
        
        # Güneş özellikleri (varsa)
        solar_cols = [c for c in df.columns if 'solar' in c.lower() or 'sun' in c.lower() or 'gunes' in c.lower()]
        
        solar_numeric_col = None
        for col in solar_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                solar_numeric_col = col
                break
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > 0:
                    solar_numeric_col = col
                    break
            except:
                continue
        
        if solar_numeric_col is not None:
            solar = df[solar_numeric_col].shift(self.prediction_horizon)
            
            df['solar_ma_24h'] = solar.rolling(24).mean()
            df['solar_change_24h'] = solar.diff(24)
            
            print(f"    ✓ Güneş özellikleri oluşturuldu: {solar_numeric_col}")
        
        # Toplam yenilenebilir oranı
        if 'wind_ma_24h' in df.columns and 'load_forecast' in df.columns:
            load_ma = df['load_forecast'].rolling(24).mean()
            # INF önleme
            df['renewable_ratio'] = np.where(
                load_ma.abs() > 100,  # Yük 100 MW'dan büyükse
                df['wind_ma_24h'] / load_ma,
                0
            )
        
        return df
    
    # =========================================================================
    # 8. HEDEF DEĞİŞKEN
    # =========================================================================
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        72 saatlik tahmin için hedef değişken oluşturur.
        
        Birden fazla hedef: Saatlik, günlük ortalama, peak/off-peak
        """
        if 'ptf' not in df.columns:
            return df
        
        # Ana hedef: 72 saat sonraki PTF
        df['target_ptf_72h'] = df['ptf'].shift(-self.prediction_horizon)
        
        # Alternatif hedefler (multi-output için)
        # 24 saat sonraki PTF
        df['target_ptf_24h'] = df['ptf'].shift(-24)
        
        # 48 saat sonraki PTF
        df['target_ptf_48h'] = df['ptf'].shift(-48)
        
        # 72 saat sonraki günlük ortalama
        df['target_ptf_72h_daily_mean'] = df['ptf'].shift(-self.prediction_horizon).rolling(24).mean()
        
        return df
    
    # =========================================================================
    # YARDIMCI METODLAR
    # =========================================================================
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Öznitelikleri gruplarına göre döndürür."""
        return {
            'lag': [c for c in self._feature_cols if 'lag' in c],
            'rolling': [c for c in self._feature_cols if any(x in c for x in ['ma_', 'std_', 'min_', 'max_'])],
            'calendar': [c for c in self._feature_cols if any(x in c for x in ['hour', 'day', 'month', 'week', 'is_'])],
            'cyclical': [c for c in self._feature_cols if any(x in c for x in ['_sin', '_cos'])],
            'system': [c for c in self._feature_cols if any(x in c for x in ['smf', 'spread', 'system'])],
            'load': [c for c in self._feature_cols if 'load' in c],
            'renewable': [c for c in self._feature_cols if any(x in c for x in ['wind', 'solar', 'renewable'])]
        }
    
    @staticmethod
    def remove_leaky_features(df: pd.DataFrame, target_col: str = 'target_ptf_72h') -> pd.DataFrame:
        """
        Data leakage'a neden olabilecek özellikleri kaldırır.
        
        ÖNEMLİ: Geleceğe ait bilgi içeren kolonlar modele dahil edilmemeli!
        """
        # Hedef değişkenler
        target_cols = [c for c in df.columns if 'target' in c]
        
        # Ham PTF (shift edilmemiş)
        leaky_cols = ['ptf', 'smf'] + target_cols
        
        # Sadece var olan kolonları düşür
        cols_to_drop = [c for c in leaky_cols if c in df.columns and c != target_col]
        
        return df.drop(columns=cols_to_drop, errors='ignore')


# Test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ÖZNİTELİK MÜHENDİSLİĞİ TESTİ")
    print("="*60 + "\n")
    
    # Örnek veri oluştur
    dates = pd.date_range('2024-01-01', periods=500, freq='h')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'ptf': 100 + np.random.randn(500).cumsum() + 50*np.sin(np.arange(500)*2*np.pi/24),
        'smf': 100 + np.random.randn(500).cumsum() + 50*np.sin(np.arange(500)*2*np.pi/24) + np.random.randn(500)*10,
        'load_forecast': 35000 + np.random.randn(500)*1000 + 5000*np.sin(np.arange(500)*2*np.pi/24),
        'wind_generation': 3000 + np.random.randn(500)*500
    }, index=dates)
    
    print(f"Örnek veri shape: {df.shape}")
    print(f"Kolonlar: {list(df.columns)}\n")
    
    # Feature engineer uygula
    fe = FeatureEngineer(prediction_horizon=72)
    df_features = fe.create_all_features(df)
    
    print(f"\nSonuç shape: {df_features.shape}")
    print(f"\nÖrnek öznitelikler:")
    print(df_features[['ptf_lag_24h', 'ptf_ma_24h', 'hour_sin', 'is_weekend', 'target_ptf_72h']].head(10))
    
    # Eksik veri özeti
    print(f"\nEksik veri oranları (ilk 10):")
    missing = (df_features.isnull().sum() / len(df_features) * 100).round(2)
    print(missing.head(10))
