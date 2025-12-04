"""
PTF Tahmin Projesi - Veri Çekme Modülü
======================================
EPİAŞ Şeffaflık Platformu'ndan PTF tahminlemesi için 
gerekli tüm verileri çeker ve birleştirir.

Veri Kaynakları:
- PTF (Piyasa Takas Fiyatı) - mcp
- SMF (Sistem Marjinal Fiyatı) - smp  
- Yük Tahmini - load-plan
- Rüzgar Tahmini - wind-forecast
- Gerçek Zamanlı Üretim - realtime-generation
- KGÜP (Günlük Üretim Planı) - dpp-org
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
import pickle
import warnings

warnings.filterwarnings('ignore')

# Proje path'ini ayarla
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings, Settings

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EPIASDataFetcher:
    """
    EPİAŞ Şeffaflık Platformu'ndan veri çeken sınıf.
    
    Attributes:
        eptr: EPTR2 client instance
        settings: Proje ayarları
        cache_dir: Cache dizini
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Args:
            settings: Proje ayarları. None ise config'den yüklenir.
        """
        self.settings = settings or get_settings()
        self.cache_dir = PROJECT_ROOT / self.settings.data.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # EPTR2 client'ı başlat
        self._init_eptr_client()
    
    def _init_eptr_client(self):
        """EPTR2 client'ı kimlik bilgileriyle başlatır."""
        try:
            from eptr2 import EPTR2
            
            self.eptr = EPTR2(
                username=self.settings.epias.username,
                password=self.settings.epias.password
            )
            logger.info("✓ EPİAŞ bağlantısı başarılı")
            
        except Exception as e:
            logger.error(f"✗ EPİAŞ bağlantı hatası: {e}")
            raise ConnectionError(
                "EPİAŞ'a bağlanılamadı. Lütfen config/config.yaml dosyasındaki "
                "kullanıcı adı ve şifreyi kontrol edin."
            )
    
    def _get_cache_path(self, data_type: str, start: str, end: str) -> Path:
        """Cache dosyası path'ini oluşturur."""
        filename = f"{data_type}_{start}_{end}.pkl"
        return self.cache_dir / filename
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Cache'ten veri yükler."""
        if self.settings.data.cache_enabled and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                logger.info(f"  ↳ Cache'ten yüklendi: {cache_path.name}")
                return df
            except Exception as e:
                logger.warning(f"  ↳ Cache okuma hatası: {e}")
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        """Veriyi cache'e kaydeder."""
        if self.settings.data.cache_enabled:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(df, f)
                logger.info(f"  ↳ Cache'e kaydedildi: {cache_path.name}")
            except Exception as e:
                logger.warning(f"  ↳ Cache yazma hatası: {e}")
    
    def _fetch_data(
        self, 
        call_name: str, 
        start_date: str, 
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        EPİAŞ'tan veri çeker.
        
        NOT: EPİAŞ API maksimum 1 yıllık veri çekmeye izin verir.
        Bu fonksiyon otomatik olarak tarihleri parçalara böler.
        
        Args:
            call_name: API endpoint adı (mcp, smp, load-plan vb.)
            start_date: Başlangıç tarihi (YYYY-MM-DD)
            end_date: Bitiş tarihi (YYYY-MM-DD)
            use_cache: Cache kullanılsın mı?
            
        Returns:
            DataFrame: Çekilen veri
        """
        cache_path = self._get_cache_path(call_name, start_date, end_date)
        
        # Cache kontrolü
        if use_cache:
            cached_df = self._load_from_cache(cache_path)
            if cached_df is not None:
                return cached_df
        
        # Tarihleri parçalara böl (EPİAŞ max 1 yıl izin veriyor)
        date_chunks = self._split_date_range(start_date, end_date, max_days=364)
        
        all_data = []
        
        for chunk_start, chunk_end in date_chunks:
            logger.info(f"  → {call_name} çekiliyor: {chunk_start} - {chunk_end}")
            
            try:
                df = self.eptr.call(
                    call_name,
                    start_date=chunk_start,
                    end_date=chunk_end
                )
                
                if df is not None and not df.empty:
                    all_data.append(df)
                    logger.info(f"    ✓ {len(df)} satır çekildi")
                else:
                    logger.warning(f"    ⚠ Boş veri: {chunk_start} - {chunk_end}")
                    
            except Exception as e:
                logger.error(f"    ✗ Hata ({chunk_start} - {chunk_end}): {e}")
                continue
        
        # Tüm parçaları birleştir
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates()
            self._save_to_cache(combined_df, cache_path)
            logger.info(f"  ✓ Toplam {len(combined_df)} satır çekildi")
            return combined_df
        else:
            logger.warning(f"  ⚠ Hiç veri çekilemedi: {call_name}")
            return pd.DataFrame()
    
    def _split_date_range(
        self, 
        start_date: str, 
        end_date: str, 
        max_days: int = 364
    ) -> list:
        """
        Tarih aralığını parçalara böler.
        
        EPİAŞ API maksimum 1 yıl (365 gün) veri çekmeye izin verir.
        Güvenli olmak için 364 gün kullanıyoruz.
        
        Returns:
            List of (start, end) tuples
        """
        from datetime import datetime, timedelta
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        chunks = []
        current_start = start
        
        while current_start < end:
            current_end = min(current_start + timedelta(days=max_days), end)
            chunks.append((
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d")
            ))
            current_start = current_end + timedelta(days=1)
        
        logger.info(f"  📅 Tarih aralığı {len(chunks)} parçaya bölündü")
        return chunks
    
    def fetch_ptf(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        PTF (Piyasa Takas Fiyatı) verilerini çeker.
        
        Returns:
            DataFrame: datetime index, ptf kolonu
        """
        logger.info("📊 PTF verileri çekiliyor...")
        df = self._fetch_data("mcp", start_date, end_date)
        
        if not df.empty:
            # Debug: kolonları göster
            logger.info(f"  PTF kolonları: {list(df.columns)}")
            
            # Kolon isimlerini standartlaştır
            df = self._standardize_datetime(df)
            
            # PTF kolonunu bul ve yeniden adlandır
            ptf_patterns = ['marketclearingprice', 'mcp', 'price', 'ptf', 'fiyat']
            ptf_col = None
            
            for col in df.columns:
                if col.lower() in ptf_patterns or any(p in col.lower() for p in ptf_patterns):
                    ptf_col = col
                    break
            
            if ptf_col and ptf_col != 'ptf':
                df = df.rename(columns={ptf_col: 'ptf'})
                logger.info(f"  '{ptf_col}' -> 'ptf' olarak yeniden adlandırıldı")
            
            # PTF kolonunu sayısala çevir
            if 'ptf' in df.columns:
                df['ptf'] = pd.to_numeric(df['ptf'], errors='coerce')
        
        return df
    
    def fetch_smf(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        SMF (Sistem Marjinal Fiyatı) verilerini çeker.
        
        Returns:
            DataFrame: datetime index, smf kolonu
        """
        logger.info("📊 SMF verileri çekiliyor...")
        df = self._fetch_data("smp", start_date, end_date)
        
        if not df.empty:
            logger.info(f"  SMF kolonları: {list(df.columns)}")
            
            df = self._standardize_datetime(df)
            
            # SMF kolonunu bul
            smf_patterns = ['systemmarginalprice', 'smp', 'smf', 'price', 'fiyat']
            smf_col = None
            
            for col in df.columns:
                if col.lower() in smf_patterns or any(p in col.lower() for p in smf_patterns):
                    smf_col = col
                    break
            
            if smf_col and smf_col != 'smf':
                df = df.rename(columns={smf_col: 'smf'})
                logger.info(f"  '{smf_col}' -> 'smf' olarak yeniden adlandırıldı")
            
            if 'smf' in df.columns:
                df['smf'] = pd.to_numeric(df['smf'], errors='coerce')
        
        return df
    
    def fetch_load_plan(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Yük tahmin planını çeker.
        
        Returns:
            DataFrame: datetime index, load_forecast kolonu
        """
        logger.info("📊 Yük tahmini çekiliyor...")
        df = self._fetch_data("load-plan", start_date, end_date)
        
        if not df.empty:
            logger.info(f"  Yük kolonları: {list(df.columns)}")
            
            df = self._standardize_datetime(df)
            
            # Yük kolonunu bul
            load_patterns = ['lep', 'loadestimationplan', 'load', 'forecast', 'demand', 'consumption', 'tuketim', 'talep']
            load_col = None
            
            for col in df.columns:
                if col.lower() in load_patterns or any(p in col.lower() for p in load_patterns):
                    load_col = col
                    break
            
            if load_col and load_col != 'load_forecast':
                df = df.rename(columns={load_col: 'load_forecast'})
                logger.info(f"  '{load_col}' -> 'load_forecast' olarak yeniden adlandırıldı")
            
            if 'load_forecast' in df.columns:
                df['load_forecast'] = pd.to_numeric(df['load_forecast'], errors='coerce')
        
        return df
    
    def fetch_wind_forecast(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Rüzgar üretim tahminini çeker.
        
        Returns:
            DataFrame: datetime index, wind_generation/forecast kolonları
        """
        logger.info("📊 Rüzgar tahmini çekiliyor...")
        df = self._fetch_data("wind-forecast", start_date, end_date)
        
        if not df.empty:
            logger.info(f"  Rüzgar kolonları: {list(df.columns)}")
            
            df = self._standardize_datetime(df)
            
            # Sayısal kolonları dönüştür
            numeric_patterns = ['forecast', 'generation', 'quarter', 'wind', 'actual', 'uretim']
            
            for col in df.columns:
                col_lower = col.lower()
                if any(p in col_lower for p in numeric_patterns):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ana rüzgar kolonlarını yeniden adlandır
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == 'forecast':
                    df = df.rename(columns={col: 'wind_forecast'})
                    logger.info(f"  '{col}' -> 'wind_forecast' olarak yeniden adlandırıldı")
                elif col_lower == 'generation':
                    df = df.rename(columns={col: 'wind_generation'})
                    logger.info(f"  '{col}' -> 'wind_generation' olarak yeniden adlandırıldı")
            
            # Gereksiz quarter kolonlarını kaldır (15 dakikalık detay gereksiz)
            quarter_cols = [c for c in df.columns if 'quarter' in c.lower()]
            if quarter_cols:
                df = df.drop(columns=quarter_cols, errors='ignore')
                logger.info(f"  Quarter kolonları kaldırıldı: {quarter_cols}")
            
            # Saatlik gruplama yap (15 dakikalık veri varsa)
            if not isinstance(df.index, pd.DatetimeIndex):
                return df
            
            # Eğer veri 15 dakikalık ise saatliğe çevir
            time_diff = df.index.to_series().diff().median()
            if time_diff and time_diff < pd.Timedelta(hours=1):
                logger.info(f"  15 dakikalık veri saatliğe dönüştürülüyor...")
                # Sadece sayısal kolonları grupla
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    df = df[numeric_cols].resample('h').mean()
                    logger.info(f"  Saatlik veri: {len(df)} satır")
        
        return df
    
    def fetch_realtime_generation(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Gerçek zamanlı üretim verilerini çeker (kaynak bazlı).
        
        NOT: Bu endpoint bazı hesaplarda aktif olmayabilir.
        Hata alınırsa boş DataFrame döner.
        
        Returns:
            DataFrame: Kaynak bazlı üretim verileri
        """
        logger.info("📊 Gerçek zamanlı üretim çekiliyor...")
        
        # Farklı endpoint isimleri dene
        possible_calls = ["rt-gen", "generation", "real-time-generation"]
        
        for call_name in possible_calls:
            try:
                df = self._fetch_data(call_name, start_date, end_date)
                if not df.empty:
                    df = self._standardize_datetime(df)
                    return df
            except Exception as e:
                logger.warning(f"  {call_name} çalışmadı: {e}")
                continue
        
        logger.warning("  ⚠ Gerçek zamanlı üretim verisi çekilemedi (bu normal olabilir)")
        return pd.DataFrame()
    
    def fetch_dpp(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        KGÜP (Kesinleşmiş Günlük Üretim Planı) verilerini çeker.
        
        Returns:
            DataFrame: Organizasyon bazlı üretim planları
        """
        logger.info("📊 KGÜP verileri çekiliyor...")
        df = self._fetch_data("dpp-org", start_date, end_date)
        
        if not df.empty:
            df = self._standardize_datetime(df)
        
        return df
    
    def _standardize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame'deki tarih/saat kolonlarını standartlaştırır.
        
        EPİAŞ API'sinden gelen farklı kolon formatlarını handle eder.
        
        Returns:
            DataFrame: datetime indeksli standart format
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # Debug: Gelen kolonları logla
        logger.debug(f"  Gelen kolonlar: {list(df.columns)}")
        
        # Olası tarih kolon isimleri (küçük harfe dönüştürerek arayacağız)
        date_patterns = ['date', 'tarih', 'period', 'gun']
        hour_patterns = ['hour', 'saat', 'period']
        datetime_patterns = ['datetime', 'timestamp', 'time']
        
        # Kolon isimlerini küçük harfe çevir
        col_mapping = {c: c.lower() for c in df.columns}
        
        # datetime kolonu varsa direkt kullan
        for col in df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in datetime_patterns) and 'date' in col_lower:
                try:
                    df['datetime'] = pd.to_datetime(df[col])
                    df = df.set_index('datetime').sort_index()
                    logger.debug(f"  datetime index oluşturuldu: {col}")
                    return df
                except Exception as e:
                    logger.debug(f"  {col} datetime'a çevrilemedi: {e}")
        
        # date ve hour ayrı ayrı ara
        date_col = None
        hour_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if date_col is None and any(p in col_lower for p in date_patterns):
                date_col = col
            if hour_col is None and any(p in col_lower for p in hour_patterns) and 'date' not in col_lower:
                hour_col = col
        
        # date ve hour bulunduysa birleştir
        if date_col:
            try:
                df['datetime'] = pd.to_datetime(df[date_col])
                
                if hour_col:
                    # Saat kolonunu ekle
                    df['datetime'] = df['datetime'] + pd.to_timedelta(df[hour_col], unit='h')
                
                df = df.set_index('datetime').sort_index()
                
                # Gereksiz kolonları temizle
                cols_to_drop = [date_col]
                if hour_col:
                    cols_to_drop.append(hour_col)
                df = df.drop(columns=cols_to_drop, errors='ignore')
                
                logger.debug(f"  datetime index oluşturuldu: {date_col} + {hour_col}")
                return df
                
            except Exception as e:
                logger.warning(f"  datetime oluşturma hatası: {e}")
        
        # Hiçbir şey bulunamadıysa, ilk kolonu index yap
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                # İlk kolonun tarih olup olmadığını kontrol et
                first_col = df.columns[0]
                df['datetime'] = pd.to_datetime(df[first_col])
                df = df.set_index('datetime').sort_index()
                logger.debug(f"  İlk kolon datetime olarak kullanıldı: {first_col}")
            except:
                logger.warning("  datetime index oluşturulamadı, ham veri döndürülüyor")
        
        return df
    
    def fetch_all(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Tüm gerekli verileri çeker.
        
        Args:
            start_date: Başlangıç tarihi (None ise config'den alınır)
            end_date: Bitiş tarihi (None ise bugün)
            
        Returns:
            Dict: Veri türü -> DataFrame
        """
        start_date = start_date or self.settings.data.start_date
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"\n{'='*50}")
        logger.info(f"VERİ ÇEKME BAŞLADI: {start_date} → {end_date}")
        logger.info(f"{'='*50}\n")
        
        data = {}
        
        # 1. PTF (Ana hedef değişken)
        data['ptf'] = self.fetch_ptf(start_date, end_date)
        
        # 2. SMF (Sistem dengesizlik sinyali)
        data['smf'] = self.fetch_smf(start_date, end_date)
        
        # 3. Yük Tahmini
        data['load'] = self.fetch_load_plan(start_date, end_date)
        
        # 4. Rüzgar Tahmini
        data['wind'] = self.fetch_wind_forecast(start_date, end_date)
        
        # 5. Gerçek Zamanlı Üretim
        data['generation'] = self.fetch_realtime_generation(start_date, end_date)
        
        logger.info(f"\n{'='*50}")
        logger.info("VERİ ÇEKME TAMAMLANDI")
        for name, df in data.items():
            if not df.empty:
                logger.info(f"  {name}: {len(df)} satır")
        logger.info(f"{'='*50}\n")
        
        return data


class DataMerger:
    """
    Farklı veri kaynaklarını birleştiren sınıf.
    PTF tahminlemesi için tek bir DataFrame oluşturur.
    """
    
    @staticmethod
    def merge_datasets(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Tüm veri setlerini birleştirir.
        
        Args:
            data: fetch_all() çıktısı
            
        Returns:
            DataFrame: Birleştirilmiş veri seti
        """
        logger.info("🔗 Veri setleri birleştiriliyor...")
        
        # PTF ana tablo olacak
        if 'ptf' not in data or data['ptf'].empty:
            # Alternatif: diğer veri setlerinden birini kullan
            for key in ['smf', 'load', 'wind']:
                if key in data and not data[key].empty:
                    logger.warning(f"PTF verisi bulunamadı, {key} ana tablo olarak kullanılıyor")
                    data['ptf'] = data[key].copy()
                    break
            else:
                raise ValueError("PTF verisi bulunamadı ve alternatif veri de yok!")
        
        merged = data['ptf'].copy()
        
        # Index'in datetime olduğundan emin ol
        if not isinstance(merged.index, pd.DatetimeIndex):
            # datetime kolonu ara
            dt_cols = [c for c in merged.columns if 'date' in c.lower() or 'time' in c.lower()]
            if dt_cols:
                merged['datetime'] = pd.to_datetime(merged[dt_cols[0]])
                merged = merged.set_index('datetime')
        
        # Diğer verileri birleştir
        for name, df in data.items():
            if name == 'ptf' or df.empty:
                continue
            
            try:
                # Index'in uyumlu olduğundan emin ol
                if not isinstance(df.index, pd.DatetimeIndex):
                    dt_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
                    if dt_cols:
                        df['datetime'] = pd.to_datetime(df[dt_cols[0]])
                        df = df.set_index('datetime')
                
                # Index bazlı merge (datetime index)
                merged = merged.join(df, how='left', rsuffix=f'_{name}')
                logger.info(f"  ✓ {name} birleştirildi ({len(df)} satır)")
            except Exception as e:
                logger.warning(f"  ⚠ {name} birleştirilemedi: {e}")
        
        # Duplicate kolonları temizle
        merged = merged.loc[:, ~merged.columns.duplicated()]
        
        # Index'i sırala
        merged = merged.sort_index()
        
        logger.info(f"  → Final shape: {merged.shape}")
        
        return merged
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Veri kalitesini kontrol eder.
        
        Returns:
            Tuple[bool, List[str]]: (Geçerli mi?, Uyarılar listesi)
        """
        warnings = []
        
        # 1. Eksik veri kontrolü
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        high_missing = missing_pct[missing_pct > 5]
        if not high_missing.empty:
            warnings.append(f"Yüksek eksik veri: {high_missing.to_dict()}")
        
        # 2. PTF değer aralığı kontrolü
        if 'ptf' in df.columns:
            ptf_min, ptf_max = df['ptf'].min(), df['ptf'].max()
            if ptf_min < 0:
                warnings.append(f"Negatif PTF değeri: {ptf_min}")
            if ptf_max > 10000:  # Anormal yüksek
                warnings.append(f"Anormal yüksek PTF: {ptf_max}")
        
        # 3. Zaman sürekliliği kontrolü
        if isinstance(df.index, pd.DatetimeIndex):
            time_diff = df.index.to_series().diff()
            gaps = time_diff[time_diff > pd.Timedelta(hours=1)]
            if not gaps.empty:
                warnings.append(f"Zaman boşlukları: {len(gaps)} adet")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings


# Test ve kullanım örneği
if __name__ == "__main__":
    print("\n" + "="*60)
    print("EPİAŞ VERİ ÇEKME TESTİ")
    print("="*60 + "\n")
    
    try:
        # Settings yükle
        settings = get_settings()
        print(f"✓ Config yüklendi")
        print(f"  Username: {settings.epias.username[:3]}***")
        print(f"  Tarih aralığı: {settings.data.start_date} - bugün")
        
        # Fetcher oluştur
        fetcher = EPIASDataFetcher(settings)
        
        # Test için kısa bir aralık çek
        test_start = "2024-01-01"
        test_end = "2024-01-07"
        
        print(f"\n📥 Test verisi çekiliyor: {test_start} - {test_end}")
        
        data = fetcher.fetch_all(test_start, test_end)
        
        # Verileri birleştir
        merger = DataMerger()
        merged_df = merger.merge_datasets(data)
        
        # Validasyon
        is_valid, warnings = merger.validate_data(merged_df)
        
        print("\n" + "="*60)
        print("SONUÇ")
        print("="*60)
        print(f"Veri geçerliliği: {'✓ Geçerli' if is_valid else '⚠ Uyarılar var'}")
        for w in warnings:
            print(f"  - {w}")
        print(f"\nFinal DataFrame:")
        print(merged_df.head())
        
    except Exception as e:
        print(f"\n✗ Hata: {e}")
        print("\nLütfen config/config.yaml dosyasını kontrol edin.")
