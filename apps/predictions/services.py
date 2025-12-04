"""
PTF Prediction Service - Gerçek Model
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone

logger = logging.getLogger(__name__)


class PTFPredictionService:
    
    def __init__(self):
        self.model = None
        self.feature_cols = []
        self._load_model()
    
    def _load_model(self):
        try:
            ml_models_dir = Path(settings.BASE_DIR) / 'ml_models'
            pkl_files = list(ml_models_dir.glob('*.pkl'))
            
            if not pkl_files:
                logger.warning("Model dosyasi bulunamadi, demo mod aktif")
                return
            
            model_file = pkl_files[0]
            
            with open(model_file, 'rb') as f:
                saved_data = pickle.load(f)
            
            if isinstance(saved_data, dict):
                self.model = saved_data.get('model')
                self.feature_cols = saved_data.get('feature_cols', [])
            else:
                self.model = saved_data
            
            logger.info("Model yuklendi: " + str(model_file.name))
            
        except Exception as e:
            logger.error("Model yukleme hatasi: " + str(e))
    
    def _fetch_epias_data(self):
        """EPİAŞ'tan gerçek veri çek"""
        try:
            from eptr2 import EPTR
            
            eptr = EPTR()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)
            
            ptf_df = eptr.ptf(start_date=start_date, end_date=end_date)
            
            if ptf_df is None or ptf_df.empty:
                return self._get_demo_data()
            
            if not isinstance(ptf_df.index, pd.DatetimeIndex):
                ptf_df.index = pd.to_datetime(ptf_df.index)
            
            if 'ptf' not in ptf_df.columns:
                ptf_df.columns = ['ptf']
            
            logger.info("EPIAS'tan " + str(len(ptf_df)) + " satir veri cekildi")
            return ptf_df
            
        except ImportError:
            logger.warning("eptr2 yuklu degil")
            return self._get_demo_data()
        except Exception as e:
            logger.error("EPIAS hatasi: " + str(e))
            return self._get_demo_data()
    
    def _get_demo_data(self):
        """Demo veri"""
        dates = pd.date_range(
            end=datetime.now(),
            periods=336,
            freq='h'
        )
        
        np.random.seed(42)
        base_price = 2500
        prices = []
        
        for dt in dates:
            hour = dt.hour
            weekday = dt.weekday()
            
            price = base_price
            
            if 0 <= hour < 6:
                price -= 400
            elif 6 <= hour < 9:
                price += 200
            elif 9 <= hour < 18:
                price += 300
            elif 18 <= hour < 22:
                price += 500
            else:
                price -= 100
            
            if weekday >= 5:
                price -= 300
            
            price += np.random.randn() * 150
            prices.append(max(price, 500))
        
        return pd.DataFrame({'ptf': prices}, index=dates)
    
    def _create_features(self, df):
        """Öznitelik mühendisliği"""
        df = df.copy()
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_peak_hour'] = df['hour'].isin([18, 19, 20, 21]).astype(int)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        for lag in [1, 2, 3, 24, 48, 168]:
            col_name = 'ptf_lag_' + str(lag) + 'h'
            df[col_name] = df['ptf'].shift(lag)
        
        df['ptf_ma_24h'] = df['ptf'].rolling(24).mean()
        df['ptf_ma_168h'] = df['ptf'].rolling(168).mean()
        df['ptf_std_24h'] = df['ptf'].rolling(24).std()
        
        return df.dropna()
    
    def predict_next_72h(self):
        """72 saatlik tahmin"""
        cache_key = 'ptf_predictions_72h'
        cached = cache.get(cache_key)
        
        if cached:
            return cached
        
        try:
            df = self._fetch_epias_data()
            df_features = self._create_features(df)
            
            if df_features.empty:
                return self._get_demo_predictions()
            
            last_price = float(df_features['ptf'].iloc[-1])
            last_std = float(df_features['ptf_std_24h'].iloc[-1]) if 'ptf_std_24h' in df_features else last_price * 0.1
            
            predictions = []
            now = timezone.now()
            
            for i in range(72):
                target_dt = now + timedelta(hours=i+1)
                target_hour = target_dt.hour
                
                if self.model is not None:
                    pred_value = self._predict_with_model(df_features, target_hour)
                else:
                    hour_effect = 200 * np.sin(2 * np.pi * target_hour / 24 - np.pi/2)
                    pred_value = last_price + hour_effect + np.random.randn() * 50
                
                confidence = max(0.5, 0.95 - (i * 0.005))
                std_multiplier = 1 + (i * 0.02)
                
                lower = pred_value - 1.96 * last_std * std_multiplier
                upper = pred_value + 1.96 * last_std * std_multiplier
                
                predictions.append({
                    'datetime': target_dt.isoformat(),
                    'date': target_dt.strftime('%Y-%m-%d'),
                    'hour': target_hour,
                    'predicted_price': round(pred_value, 2),
                    'lower_bound': round(max(lower, 0), 2),
                    'upper_bound': round(upper, 2),
                    'confidence': round(confidence * 100, 0),
                })
            
            cache.set(cache_key, predictions, 3600)
            return predictions
            
        except Exception as e:
            logger.error("Tahmin hatasi: " + str(e))
            return self._get_demo_predictions()
    
    def _predict_with_model(self, df_features, target_hour):
        """Model ile tahmin"""
        try:
            last_row = df_features.iloc[-1:].copy()
            last_row['hour'] = target_hour
            last_row['hour_sin'] = np.sin(2 * np.pi * target_hour / 24)
            last_row['hour_cos'] = np.cos(2 * np.pi * target_hour / 24)
            last_row['is_peak_hour'] = 1 if target_hour in [18, 19, 20, 21] else 0
            
            if self.feature_cols:
                for col in self.feature_cols:
                    if col not in last_row.columns:
                        last_row[col] = 0
                X = last_row[self.feature_cols]
            else:
                numeric_cols = last_row.select_dtypes(include=[np.number]).columns.tolist()
                if 'ptf' in numeric_cols:
                    numeric_cols.remove('ptf')
                X = last_row[numeric_cols]
            
            return float(self.model.predict(X)[0])
            
        except Exception as e:
            logger.error("Model tahmin hatasi: " + str(e))
            return float(df_features['ptf'].iloc[-1])
    
    def _get_demo_predictions(self):
        """Demo tahminler"""
        predictions = []
        now = timezone.now()
        base_price = 2700
        
        for i in range(72):
            target_dt = now + timedelta(hours=i+1)
            hour = target_dt.hour
            weekday = target_dt.weekday()
            
            if 0 <= hour < 6:
                hour_effect = -400
            elif 6 <= hour < 9:
                hour_effect = 200
            elif 9 <= hour < 18:
                hour_effect = 300
            elif 18 <= hour < 22:
                hour_effect = 500
            else:
                hour_effect = -100
            
            weekend_effect = -300 if weekday >= 5 else 0
            noise = np.random.randn() * 100
            
            pred_value = base_price + hour_effect + weekend_effect + noise
            std = 200 + (i * 5)
            
            predictions.append({
                'datetime': target_dt.isoformat(),
                'date': target_dt.strftime('%Y-%m-%d'),
                'hour': hour,
                'predicted_price': round(pred_value, 2),
                'lower_bound': round(pred_value - 1.96 * std, 2),
                'upper_bound': round(pred_value + 1.96 * std, 2),
                'confidence': round(max(50, 95 - i * 0.5), 0),
            })
        
        return predictions
    
    def get_daily_summary(self, target_date=None):
        """Günlük özet"""
        if target_date is None:
            target_date = (timezone.now() + timedelta(days=1)).date()
        
        predictions = self.predict_next_72h()
        target_str = str(target_date)
        daily_preds = [p for p in predictions if p['date'] == target_str]
        
        if not daily_preds:
            return None
        
        prices = [p['predicted_price'] for p in daily_preds]
        
        return {
            'date': target_str,
            'min_price': round(min(prices), 2),
            'max_price': round(max(prices), 2),
            'avg_price': round(sum(prices) / len(prices), 2),
            'min_hour': prices.index(min(prices)),
            'max_hour': prices.index(max(prices)),
            'predictions': daily_preds,
        }
    
    def refresh_cache(self):
        """Cache yenile"""
        cache.delete('ptf_predictions_72h')
        return self.predict_next_72h()