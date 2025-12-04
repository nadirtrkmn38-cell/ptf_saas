"""
PTF Tahmin Projesi - Emtia ve Döviz Verileri Modülü
===================================================
- TCMB API'den USD/TRY kuru
- TTF Doğalgaz fiyatları
- Kömür fiyatları (API2)
- Spark Spread ve Dark Spread hesaplamaları
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging
import requests
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class CommodityDataFetcher:
    """
    Emtia ve döviz verilerini çeken sınıf.
    
    Veri Kaynakları:
    - TCMB EVDS API: USD/TRY, EUR/TRY
    - Investing.com / Yahoo Finance: TTF Gaz, API2 Kömür
    """
    
    # Türkiye santral verimlilik ortalamaları
    GAS_EFFICIENCY = 0.50  # %50 - DGKÇS
    COAL_EFFICIENCY = 0.38  # %38 - İthal Kömür
    
    # Enerji dönüşüm sabitleri
    # 1 MWh = 3.6 GJ
    # TTF: EUR/MWh (doğrudan)
    # API2 Kömür: USD/ton, 1 ton = ~6.0 MWh (6000 kcal/kg)
    COAL_ENERGY_CONTENT = 6.0  # MWh/ton
    
    def __init__(self, tcmb_api_key: Optional[str] = None):
        """
        Args:
            tcmb_api_key: TCMB EVDS API anahtarı (opsiyonel)
        """
        self.tcmb_api_key = tcmb_api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_usd_try(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        TCMB'den USD/TRY kurunu çeker.
        
        Alternatif: Yahoo Finance veya ExchangeRate API
        """
        logger.info("💱 USD/TRY kuru çekiliyor...")
        
        # Yöntem 1: Yahoo Finance (ücretsiz, güvenilir)
        try:
            df = self._fetch_yahoo_currency("USDTRY=X", start_date, end_date)
            if not df.empty:
                df = df.rename(columns={'Close': 'usd_try'})
                logger.info(f"  ✓ Yahoo Finance'den {len(df)} gün USD/TRY çekildi")
                return df[['usd_try']]
        except Exception as e:
            logger.warning(f"  Yahoo Finance hatası: {e}")
        
        # Yöntem 2: Sentetik veri oluştur (fallback)
        logger.warning("  ⚠ Gerçek kur verisi çekilemedi, trend bazlı veri oluşturuluyor")
        return self._create_synthetic_usd_try(start_date, end_date)
    
    def fetch_eur_try(self, start_date: str, end_date: str) -> pd.DataFrame:
        """EUR/TRY kurunu çeker."""
        logger.info("💱 EUR/TRY kuru çekiliyor...")
        
        try:
            df = self._fetch_yahoo_currency("EURTRY=X", start_date, end_date)
            if not df.empty:
                df = df.rename(columns={'Close': 'eur_try'})
                logger.info(f"  ✓ {len(df)} gün EUR/TRY çekildi")
                return df[['eur_try']]
        except Exception as e:
            logger.warning(f"  EUR/TRY hatası: {e}")
        
        return pd.DataFrame()
    
    def fetch_ttf_gas(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        TTF Doğalgaz fiyatını çeker (EUR/MWh).
        
        TTF = Title Transfer Facility (Hollanda doğalgaz hub'ı)
        """
        logger.info("🔥 TTF Doğalgaz fiyatı çekiliyor...")
        
        # Yahoo Finance TTF futures
        try:
            df = self._fetch_yahoo_commodity("TTF=F", start_date, end_date)
            if not df.empty:
                df = df.rename(columns={'Close': 'ttf_eur_mwh'})
                logger.info(f"  ✓ {len(df)} gün TTF fiyatı çekildi")
                return df[['ttf_eur_mwh']]
        except Exception as e:
            logger.warning(f"  TTF hatası: {e}")
        
        # Alternatif: Dutch TTF Gas
        try:
            df = self._fetch_yahoo_commodity("NG=F", start_date, end_date)
            if not df.empty:
                # Henry Hub USD/MMBtu -> EUR/MWh dönüşümü
                # 1 MMBtu = 0.293 MWh
                df['ttf_eur_mwh'] = df['Close'] / 0.293 * 0.92  # USD->EUR yaklaşık
                logger.info(f"  ✓ Henry Hub'dan yaklaşık TTF hesaplandı")
                return df[['ttf_eur_mwh']]
        except:
            pass
        
        logger.warning("  ⚠ TTF verisi çekilemedi, sentetik veri oluşturuluyor")
        return self._create_synthetic_gas(start_date, end_date)
    
    def fetch_coal_api2(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        API2 Kömür fiyatını çeker (USD/ton).
        
        API2 = Amsterdam-Rotterdam-Antwerp kömür endeksi
        """
        logger.info("�ite Kömür fiyatı çekiliyor...")
        
        # Yahoo Finance Coal futures
        try:
            df = self._fetch_yahoo_commodity("MTF=F", start_date, end_date)
            if not df.empty:
                df = df.rename(columns={'Close': 'coal_usd_ton'})
                logger.info(f"  ✓ {len(df)} gün kömür fiyatı çekildi")
                return df[['coal_usd_ton']]
        except Exception as e:
            logger.warning(f"  Kömür hatası: {e}")
        
        logger.warning("  ⚠ Kömür verisi çekilemedi, sentetik veri oluşturuluyor")
        return self._create_synthetic_coal(start_date, end_date)
    
    def _fetch_yahoo_currency(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Yahoo Finance'den döviz verisi çeker."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)
            if not df.empty:
                df.index = df.index.tz_localize(None)
            return df
        except ImportError:
            logger.warning("yfinance yüklü değil, pip install yfinance")
            return pd.DataFrame()
    
    def _fetch_yahoo_commodity(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Yahoo Finance'den emtia verisi çeker."""
        return self._fetch_yahoo_currency(symbol, start, end)
    
    def _create_synthetic_usd_try(self, start: str, end: str) -> pd.DataFrame:
        """Gerçekçi USD/TRY sentetik verisi oluşturur."""
        dates = pd.date_range(start, end, freq='D')
        n = len(dates)
        
        # 2022 başı: ~13 TL, 2024 sonu: ~35 TL
        # Yıllık ~%70 artış trendi
        t = np.arange(n)
        base = 13
        trend = base * (1 + 0.7 * t / 365)  # Yıllık %70 artış
        noise = np.random.randn(n) * 0.3  # Günlük volatilite
        
        usd_try = trend + noise
        usd_try = np.maximum(usd_try, 10)  # Minimum 10 TL
        
        return pd.DataFrame({'usd_try': usd_try}, index=dates)
    
    def _create_synthetic_gas(self, start: str, end: str) -> pd.DataFrame:
        """Gerçekçi TTF gaz sentetik verisi oluşturur."""
        dates = pd.date_range(start, end, freq='D')
        n = len(dates)
        
        # 2022: Enerji krizi ~100-300 EUR/MWh
        # 2023-2024: Normalleşme ~30-50 EUR/MWh
        t = np.arange(n)
        
        # Kriz spike'ı 2022 ortasında
        crisis_peak = 200 * np.exp(-((t - 180) ** 2) / (2 * 90 ** 2))
        base = 40 + crisis_peak
        
        # Mevsimsellik (kış yüksek)
        seasonal = 15 * np.sin(2 * np.pi * t / 365 + np.pi)
        
        noise = np.random.randn(n) * 5
        
        ttf = base + seasonal + noise
        ttf = np.maximum(ttf, 20)
        
        return pd.DataFrame({'ttf_eur_mwh': ttf}, index=dates)
    
    def _create_synthetic_coal(self, start: str, end: str) -> pd.DataFrame:
        """Gerçekçi API2 kömür sentetik verisi oluşturur."""
        dates = pd.date_range(start, end, freq='D')
        n = len(dates)
        
        # 2022 krizi: ~400 USD/ton peak
        # Normal: ~100-150 USD/ton
        t = np.arange(n)
        
        crisis_peak = 250 * np.exp(-((t - 200) ** 2) / (2 * 100 ** 2))
        base = 120 + crisis_peak
        
        noise = np.random.randn(n) * 10
        
        coal = base + noise
        coal = np.maximum(coal, 80)
        
        return pd.DataFrame({'coal_usd_ton': coal}, index=dates)
    
    def fetch_all_commodities(
        self, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Tüm emtia ve döviz verilerini çeker ve birleştirir.
        """
        logger.info("\n" + "="*50)
        logger.info("EMTİA VE DÖVİZ VERİLERİ ÇEKİLİYOR")
        logger.info("="*50 + "\n")
        
        # Verileri çek
        usd = self.fetch_usd_try(start_date, end_date)
        eur = self.fetch_eur_try(start_date, end_date)
        gas = self.fetch_ttf_gas(start_date, end_date)
        coal = self.fetch_coal_api2(start_date, end_date)
        
        # Birleştir
        dfs = [usd, eur, gas, coal]
        dfs = [df for df in dfs if not df.empty]
        
        if not dfs:
            logger.warning("Hiç emtia verisi çekilemedi!")
            return pd.DataFrame()
        
        combined = pd.concat(dfs, axis=1)
        
        # Eksik günleri doldur (forward fill)
        combined = combined.resample('D').last().ffill()
        
        logger.info(f"\n✓ Toplam {len(combined)} gün emtia verisi hazır")
        logger.info(f"  Kolonlar: {list(combined.columns)}")
        
        return combined
    
    def calculate_spreads(
        self, 
        df: pd.DataFrame,
        ptf_col: str = 'ptf',
        usd_col: str = 'usd_try',
        eur_col: str = 'eur_try',
        gas_col: str = 'ttf_eur_mwh',
        coal_col: str = 'coal_usd_ton'
    ) -> pd.DataFrame:
        """
        Spark Spread ve Dark Spread hesaplar.
        
        Spark Spread (Gaz): PTF - (Gaz Maliyeti / Verimlilik)
        Dark Spread (Kömür): PTF - (Kömür Maliyeti / Verimlilik)
        
        Pozitif spread = Santral karlı
        Negatif spread = Santral zararlı
        """
        df = df.copy()
        
        # EUR/TRY yoksa USD'den tahmin et (EUR ≈ 1.08 * USD)
        if eur_col not in df.columns and usd_col in df.columns:
            df[eur_col] = df[usd_col] * 1.08
        
        # Gaz maliyeti (TL/MWh)
        if gas_col in df.columns and eur_col in df.columns:
            # TTF EUR/MWh -> TL/MWh
            gas_cost_tl = df[gas_col] * df[eur_col]
            
            # Santral giriş maliyeti (verimlilik hesabı)
            # 1 MWh elektrik üretmek için 1/verimlilik MWh gaz gerekir
            df['gas_input_cost'] = gas_cost_tl / self.GAS_EFFICIENCY
            
            # Spark Spread
            if ptf_col in df.columns:
                df['spark_spread'] = df[ptf_col] - df['gas_input_cost']
                logger.info("  ✓ Spark Spread hesaplandı")
        
        # Kömür maliyeti (TL/MWh)
        if coal_col in df.columns and usd_col in df.columns:
            # USD/ton -> TL/ton
            coal_cost_tl = df[coal_col] * df[usd_col]
            
            # TL/ton -> TL/MWh
            coal_cost_mwh = coal_cost_tl / self.COAL_ENERGY_CONTENT
            
            # Santral giriş maliyeti
            df['coal_input_cost'] = coal_cost_mwh / self.COAL_EFFICIENCY
            
            # Dark Spread
            if ptf_col in df.columns:
                df['dark_spread'] = df[ptf_col] - df['coal_input_cost']
                logger.info("  ✓ Dark Spread hesaplandı")
        
        # Clean Spark Spread (CO2 maliyeti dahil - opsiyonel)
        # Türkiye'de henüz karbon vergisi düşük, şimdilik atlıyoruz
        
        return df


def calculate_residual_load(
    df: pd.DataFrame,
    load_col: str = 'load_forecast',
    wind_col: str = 'wind_generation',
    solar_col: str = 'solar_generation'
) -> pd.DataFrame:
    """
    Residual Load (Net Yük) hesaplar.
    
    Residual Load = Toplam Yük - Yenilenebilir Üretim
    
    Bu değer, termik santrallerin karşılaması gereken yükü gösterir.
    Merit order'da fiyatı belirleyen budur.
    """
    df = df.copy()
    
    # Yenilenebilir toplam
    renewable = 0
    
    wind_cols = [c for c in df.columns if 'wind' in c.lower() and 'ma' not in c.lower() and 'lag' not in c.lower()]
    solar_cols = [c for c in df.columns if 'solar' in c.lower() and 'ma' not in c.lower() and 'lag' not in c.lower()]
    
    if wind_cols:
        # İlk uygun rüzgar kolonunu bul
        for col in wind_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                renewable = renewable + df[col].fillna(0)
                break
    
    if solar_cols:
        for col in solar_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                renewable = renewable + df[col].fillna(0)
                break
    
    # Yük kolonu
    load = None
    if load_col in df.columns:
        load = df[load_col]
    else:
        load_cols = [c for c in df.columns if 'load' in c.lower() and 'ma' not in c.lower()]
        if load_cols:
            load = df[load_cols[0]]
    
    if load is not None:
        # Residual Load
        df['residual_load'] = load - renewable
        
        # Residual Load Squared (karesel maliyet eğrisi)
        # Normalize et ki çok büyük sayılar olmasın
        residual_normalized = df['residual_load'] / 1000  # GW cinsine çevir
        df['residual_load_squared'] = residual_normalized ** 2
        
        # Log transform (opsiyonel - çok yüksek değerler için)
        df['residual_load_log'] = np.log1p(np.maximum(df['residual_load'], 0))
        
        logger.info(f"  ✓ Residual Load hesaplandı (Ort: {df['residual_load'].mean():.0f} MW)")
    
    return df


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("EMTİA VERİ TESTİ")
    print("="*60 + "\n")
    
    fetcher = CommodityDataFetcher()
    
    # Test verisi çek
    commodities = fetcher.fetch_all_commodities("2024-01-01", "2024-12-01")
    
    if not commodities.empty:
        print("\nÖrnek veri:")
        print(commodities.tail(10))
        
        # Spread hesapla (örnek PTF ile)
        commodities['ptf'] = 2500  # Örnek PTF
        commodities = fetcher.calculate_spreads(commodities)
        
        print("\nSpread'ler:")
        print(commodities[['ptf', 'gas_input_cost', 'spark_spread', 
                          'coal_input_cost', 'dark_spread']].tail(5))
