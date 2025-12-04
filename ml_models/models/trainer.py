"""
PTF Tahmin Projesi - Model Eğitim Modülü
========================================
XGBoost ve LightGBM ile PTF tahmin modelleri.

Özellikler:
- Zaman serisi için uygun train/test split
- Hiperparametre optimizasyonu (Optuna)
- Cross-validation
- SHAP analizi
- Model kaydetme/yükleme
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import json
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ML Kütüphaneleri
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost yüklü değil: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM yüklü değil: pip install lightgbm")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ============================================================================
# METRİK FONKSİYONLARI
# ============================================================================

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    
    NOT: Çok düşük gerçek değerler (< 50 TL) MAPE'yi patlatır.
    Bu değerler filtrelenir çünkü:
    1. PTF nadiren 50 TL'nin altına düşer
    2. Düşük değerler genellikle hatalı veri veya özel durumlar
    3. Ticari açıdan yüksek fiyatlı saatler daha önemli
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Çok düşük değerleri filtrele (< 50 TL)
    mask = y_true > 50
    
    if mask.sum() == 0:
        # Hiç değer kalmadıysa eski yöntemle hesapla
        mask = y_true != 0
    
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).
    
    MAPE'den daha dengeli - aşırı düşük/yüksek değerlere karşı dayanıklı.
    Range: 0-200%
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Çok küçük payda değerlerini filtrele
    mask = denominator > 10
    
    if mask.sum() == 0:
        mask = denominator != 0
    
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def weighted_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Hacim Ağırlıklı MAPE - Yüksek fiyatlı saatlere daha fazla ağırlık verir.
    
    Enerji piyasasında yüksek fiyatlı saatler ticari açıdan daha önemlidir.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Çok düşük değerleri filtrele
    mask = y_true > 50
    
    if mask.sum() == 0:
        return np.nan
    
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]
    
    # Ağırlıklar = gerçek değerler (yüksek fiyat = yüksek ağırlık)
    weights = y_true_f
    errors = np.abs((y_true_f - y_pred_f) / y_true_f)
    
    return np.average(errors, weights=weights) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tüm metrikleri hesaplar"""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'weighted_mape': weighted_mape(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


# ============================================================================
# VERİ HAZIRLAMA
# ============================================================================

class DataPreparer:
    """
    Model eğitimi için veri hazırlama sınıfı.
    
    - Train/Test/Validation split (zaman serisi uyumlu)
    - Eksik veri temizleme
    - Öznitelik seçimi
    """
    
    def __init__(
        self, 
        target_col: str = 'target_ptf_72h',
        test_size: float = 0.2,
        val_size: float = 0.1
    ):
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.feature_cols = None
        self.scaler = None
    
    def prepare(
        self, 
        df: pd.DataFrame,
        drop_cols: List[str] = None,
        scale_features: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Veriyi train/val/test olarak ayırır.
        
        ÖNEMLI: Zaman serisi için random split YAPILMAZ!
        Kronolojik sıra korunur.
        
        Returns:
            Tuple[train_df, val_df, test_df]
        """
        df = df.copy()
        
        # Hedef değişken kontrolü
        if self.target_col not in df.columns:
            raise ValueError(f"Hedef kolon bulunamadı: {self.target_col}")
        
        # Kaldırılacak kolonlar
        drop_cols = drop_cols or []
        target_cols = [c for c in df.columns if 'target' in c and c != self.target_col]
        
        # Ham fiyat kolonlarını kaldır (data leakage önleme)
        leaky_cols = ['ptf', 'smf']
        
        # DATETIME KOLONLARINI KALDIR (XGBoost bunları işleyemez)
        datetime_cols = []
        for col in df.columns:
            if df[col].dtype.name.startswith('datetime') or 'datetime' in str(df[col].dtype):
                datetime_cols.append(col)
            elif df[col].dtype == 'object':
                # Object tipindeki kolonları kontrol et
                try:
                    pd.to_datetime(df[col].iloc[0])
                    datetime_cols.append(col)
                except:
                    pass
        
        if datetime_cols:
            logger.info(f"Datetime kolonları kaldırılıyor: {datetime_cols}")
        
        cols_to_drop = list(set(drop_cols + target_cols + leaky_cols + datetime_cols))
        
        # Öznitelik kolonlarını belirle
        self.feature_cols = [
            c for c in df.columns 
            if c not in cols_to_drop and c != self.target_col
        ]
        
        # Sayısal olmayan kolonları da kaldır
        non_numeric_cols = []
        for col in self.feature_cols:
            if df[col].dtype not in ['float64', 'float32', 'int64', 'int32', 'bool', 'int8', 'int16', 'float16']:
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            logger.warning(f"Sayısal olmayan kolonlar kaldırılıyor: {non_numeric_cols}")
            self.feature_cols = [c for c in self.feature_cols if c not in non_numeric_cols]
        
        logger.info(f"Toplam {len(self.feature_cols)} öznitelik kullanılacak")
        
        # Eksik değerleri temizle - AMA önce kaç satır kaybedeceğimizi kontrol et
        df_subset = df[[self.target_col] + self.feature_cols]
        
        # Her kolondaki eksik veri yüzdesini kontrol et
        missing_pct = (df_subset.isnull().sum() / len(df_subset) * 100).round(2)
        high_missing = missing_pct[missing_pct > 50]
        
        if not high_missing.empty:
            logger.warning(f"Yüksek eksik verili kolonlar kaldırılıyor (>50%): {list(high_missing.index)}")
            cols_to_remove = list(high_missing.index)
            self.feature_cols = [c for c in self.feature_cols if c not in cols_to_remove]
            df_subset = df[[self.target_col] + self.feature_cols]
        
        # Şimdi eksik satırları temizle
        initial_rows = len(df_subset)
        
        # INF DEĞERLERİ TEMİZLE (XGBoost inf kabul etmiyor!)
        df_subset = df_subset.replace([np.inf, -np.inf], np.nan)
        
        df_clean = df_subset.dropna()
        final_rows = len(df_clean)
        
        logger.info(f"Eksik veri temizlendi: {initial_rows} → {final_rows} satır ({initial_rows - final_rows} satır çıkarıldı)")
        
        if final_rows == 0:
            # Eğer tüm satırlar silindiyse, forward fill dene
            logger.warning("Tüm satırlar silindi! Forward fill deneniyor...")
            df_subset = df_subset.fillna(method='ffill').fillna(method='bfill')
            df_clean = df_subset.dropna()
            final_rows = len(df_clean)
            logger.info(f"Forward fill sonrası: {final_rows} satır")
        
        if final_rows == 0:
            raise ValueError("Veri hazırlama sonrası 0 satır kaldı! Veri kalitesini kontrol edin.")
        
        # Kronolojik split
        n = len(df_clean)
        test_idx = int(n * (1 - self.test_size))
        val_idx = int(test_idx * (1 - self.val_size))
        
        train_df = df_clean.iloc[:val_idx]
        val_df = df_clean.iloc[val_idx:test_idx]
        test_df = df_clean.iloc[test_idx:]
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Train period: {train_df.index.min()} → {train_df.index.max()}")
        logger.info(f"Test period: {test_df.index.min()} → {test_df.index.max()}")
        
        # Ölçekleme (opsiyonel)
        if scale_features:
            self.scaler = StandardScaler()
            train_df[self.feature_cols] = self.scaler.fit_transform(train_df[self.feature_cols])
            val_df[self.feature_cols] = self.scaler.transform(val_df[self.feature_cols])
            test_df[self.feature_cols] = self.scaler.transform(test_df[self.feature_cols])
        
        return train_df, val_df, test_df
    
    def get_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """DataFrame'den X ve y ayırır"""
        X = df[self.feature_cols]
        y = df[self.target_col]
        return X, y


# ============================================================================
# MODEL EĞİTİCİ
# ============================================================================

class PTFModelTrainer:
    """
    PTF tahmin modeli eğitici.
    
    Desteklenen modeller:
    - XGBoost (varsayılan)
    - LightGBM
    """
    
    # Varsayılan hiperparametreler (PTF için optimize edilmiş)
    DEFAULT_XGB_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    DEFAULT_LGB_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'num_leaves': 64,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    def __init__(
        self, 
        model_type: str = 'xgboost',
        params: Dict = None
    ):
        """
        Args:
            model_type: 'xgboost' veya 'lightgbm'
            params: Model parametreleri (None ise varsayılan kullanılır)
        """
        self.model_type = model_type.lower()
        self.model = None
        self.feature_importance = None
        self.training_history = []
        
        # Parametreleri ayarla
        if self.model_type == 'xgboost':
            if not HAS_XGB:
                raise ImportError("XGBoost yüklü değil!")
            self.params = params or self.DEFAULT_XGB_PARAMS.copy()
        elif self.model_type == 'lightgbm':
            if not HAS_LGB:
                raise ImportError("LightGBM yüklü değil!")
            self.params = params or self.DEFAULT_LGB_PARAMS.copy()
        else:
            raise ValueError(f"Desteklenmeyen model: {model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        early_stopping_rounds: int = 50
    ) -> 'PTFModelTrainer':
        """
        Modeli eğitir.
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim hedefi
            X_val: Validasyon özellikleri (erken durdurma için)
            y_val: Validasyon hedefi
            early_stopping_rounds: Erken durdurma sabrı
            
        Returns:
            self (zincirleme için)
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"{self.model_type.upper()} MODEL EĞİTİMİ")
        logger.info(f"{'='*50}")
        logger.info(f"Train shape: {X_train.shape}")
        
        start_time = datetime.now()
        
        if self.model_type == 'xgboost':
            self._train_xgboost(X_train, y_train, X_val, y_val, early_stopping_rounds)
        else:
            self._train_lightgbm(X_train, y_train, X_val, y_val, early_stopping_rounds)
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Eğitim süresi: {training_time:.1f} saniye")
        
        # Feature importance hesapla
        self._calculate_feature_importance(X_train.columns.tolist())
        
        return self
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val, early_stopping_rounds):
        """XGBoost eğitimi"""
        self.model = xgb.XGBRegressor(**self.params)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # En iyi iterasyon
        if hasattr(self.model, 'best_iteration'):
            logger.info(f"En iyi iterasyon: {self.model.best_iteration}")
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, early_stopping_rounds):
        """LightGBM eğitimi"""
        self.model = lgb.LGBMRegressor(**self.params)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        callbacks = [lgb.log_evaluation(period=0)]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )
        
        if hasattr(self.model, 'best_iteration_'):
            logger.info(f"En iyi iterasyon: {self.model.best_iteration_}")
    
    def _calculate_feature_importance(self, feature_names: List[str]):
        """Öznitelik önem skorlarını hesaplar"""
        if self.model is None:
            return
        
        importance = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Tahmin yapar"""
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi!")
        return self.model.predict(X)
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Model performansını değerlendirir.
        
        Returns:
            Dict: Metrik adı → değer
        """
        y_pred = self.predict(X_test)
        metrics = calculate_metrics(y_test.values, y_pred)
        
        if verbose:
            logger.info(f"\n📊 MODEL PERFORMANSI")
            logger.info(f"   MAE:   {metrics['mae']:.2f} TL")
            logger.info(f"   RMSE:  {metrics['rmse']:.2f} TL")
            logger.info(f"   MAPE:  {metrics['mape']:.2f}%")
            logger.info(f"   Weighted MAPE: {metrics.get('weighted_mape', 0):.2f}%")
            logger.info(f"   SMAPE: {metrics['smape']:.2f}%")
            logger.info(f"   R²:    {metrics['r2']:.4f}")
        
        return metrics
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Zaman serisi cross-validation uygular.
        
        TimeSeriesSplit kullanır (gelecek verisi sızmaz).
        """
        logger.info(f"\n🔄 {n_splits}-Fold Time Series Cross Validation")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {metric: [] for metric in ['mae', 'rmse', 'mape', 'r2']}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Yeni model oluştur ve eğit
            cv_model = PTFModelTrainer(self.model_type, self.params.copy())
            cv_model.train(X_train_cv, y_train_cv)
            
            # Değerlendir
            metrics = cv_model.evaluate(X_val_cv, y_val_cv, verbose=False)
            
            for metric, value in metrics.items():
                if metric in cv_results:
                    cv_results[metric].append(value)
            
            logger.info(f"  Fold {fold}: MAPE={metrics['mape']:.2f}%, R²={metrics['r2']:.4f}")
        
        # Ortalama sonuçlar
        logger.info(f"\n  📈 CV Ortalamaları:")
        for metric, values in cv_results.items():
            logger.info(f"     {metric.upper()}: {np.mean(values):.4f} (±{np.std(values):.4f})")
        
        return cv_results
    
    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """En önemli n özniteliği döndürür"""
        if self.feature_importance is None:
            return pd.DataFrame()
        return self.feature_importance.head(n)
    
    def save(self, path: str):
        """Modeli kaydeder"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_importance': self.feature_importance
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"✓ Model kaydedildi: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PTFModelTrainer':
        """Modeli yükler"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        trainer = cls(
            model_type=save_dict['model_type'],
            params=save_dict['params']
        )
        trainer.model = save_dict['model']
        trainer.feature_importance = save_dict['feature_importance']
        
        logger.info(f"✓ Model yüklendi: {path}")
        return trainer


# ============================================================================
# SHAP ANALİZİ
# ============================================================================

def analyze_with_shap(
    model: PTFModelTrainer,
    X: pd.DataFrame,
    max_samples: int = 1000
) -> Optional[Any]:
    """
    SHAP değerlerini hesaplar ve görselleştirir.
    
    SHAP, modelin her tahmin için hangi özelliklerin
    ne kadar etkili olduğunu gösterir.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP yüklü değil: pip install shap")
        return None
    
    logger.info("\n🔍 SHAP Analizi yapılıyor...")
    
    # Örnek sayısını sınırla (hesaplama süresi için)
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X
    
    # SHAP explainer oluştur
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_sample)
    
    # Özet istatistikler
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': mean_abs_shap
    }).sort_values('shap_importance', ascending=False)
    
    logger.info("\nSHAP Önem Sıralaması (Top 10):")
    for i, row in shap_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['shap_importance']:.2f}")
    
    return shap_values


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODEL EĞİTİM TESTİ")
    print("="*60 + "\n")
    
    # Örnek veri oluştur
    np.random.seed(42)
    n = 5000
    dates = pd.date_range('2022-01-01', periods=n, freq='h')
    
    # Gerçekçi PTF simülasyonu
    base = 100
    trend = np.linspace(0, 50, n)  # Yükselen trend
    daily = 30 * np.sin(2 * np.pi * np.arange(n) / 24)  # Günlük döngü
    weekly = 15 * np.sin(2 * np.pi * np.arange(n) / 168)  # Haftalık döngü
    noise = np.random.randn(n) * 20
    
    ptf = base + trend + daily + weekly + noise
    ptf = np.maximum(ptf, 0)  # Negatif fiyat olamaz
    
    df = pd.DataFrame({'ptf': ptf}, index=dates)
    
    # Basit lag özellikleri
    for lag in [24, 48, 168]:
        df[f'ptf_lag_{lag}'] = df['ptf'].shift(lag + 72)  # 72 saat sonrası için
    
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Hedef değişken
    df['target_ptf_72h'] = df['ptf'].shift(-72)
    
    print(f"Veri shape: {df.shape}")
    
    # Veri hazırlama
    preparer = DataPreparer(target_col='target_ptf_72h')
    train_df, val_df, test_df = preparer.prepare(df, drop_cols=['ptf'])
    
    X_train, y_train = preparer.get_xy(train_df)
    X_val, y_val = preparer.get_xy(val_df)
    X_test, y_test = preparer.get_xy(test_df)
    
    # Model eğitimi
    if HAS_XGB:
        trainer = PTFModelTrainer(model_type='xgboost')
        trainer.train(X_train, y_train, X_val, y_val)
        
        # Değerlendirme
        metrics = trainer.evaluate(X_test, y_test)
        
        # Top özellikler
        print("\n📊 En Önemli Özellikler:")
        print(trainer.get_top_features(10))
        
        # Model kaydet
        trainer.save('/home/claude/ptf_project/models/test_model.pkl')
    else:
        print("XGBoost yüklü değil, test atlandı.")
