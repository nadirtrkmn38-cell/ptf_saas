"""
Predictions App - Models
========================
PTF tahmin verileri ve geçmiş fiyatlar.
"""

from django.db import models
from django.utils import timezone
from datetime import date, timedelta


class PTFPrediction(models.Model):
    """
    PTF tahmin verileri.
    Her gün için 24 saatlik tahminler.
    """
    
    date = models.DateField('Tarih', db_index=True)
    hour = models.IntegerField('Saat', db_index=True)  # 0-23
    
    # Tahmin değerleri
    predicted_price = models.DecimalField('Tahmin fiyatı (TL/MWh)', max_digits=10, decimal_places=2)
    lower_bound = models.DecimalField('Alt sınır', max_digits=10, decimal_places=2, null=True, blank=True)
    upper_bound = models.DecimalField('Üst sınır', max_digits=10, decimal_places=2, null=True, blank=True)
    confidence = models.FloatField('Güven skoru', default=0.0)  # 0-1 arası
    
    # Gerçekleşen değer (sonradan güncellenir)
    actual_price = models.DecimalField('Gerçekleşen fiyat', max_digits=10, decimal_places=2, null=True, blank=True)
    
    # Model bilgileri
    model_version = models.CharField('Model versiyonu', max_length=50, default='v1')
    
    # Meta
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'PTF Tahmini'
        verbose_name_plural = 'PTF Tahminleri'
        ordering = ['date', 'hour']
        unique_together = ['date', 'hour']
        indexes = [
            models.Index(fields=['date', 'hour']),
            models.Index(fields=['-date']),
        ]
    
    def __str__(self):
        return f"{self.date} {self.hour:02d}:00 - {self.predicted_price} TL"
    
    @property
    def datetime(self):
        """Tarih ve saat birleşik"""
        from datetime import datetime
        return datetime.combine(self.date, datetime.min.time().replace(hour=self.hour))
    
    @property
    def error(self):
        """Tahmin hatası (gerçek değer varsa)"""
        if self.actual_price:
            return float(self.actual_price - self.predicted_price)
        return None
    
    @property
    def error_percent(self):
        """Yüzde hata"""
        if self.actual_price and self.actual_price > 0:
            return abs(self.error) / float(self.actual_price) * 100
        return None
    
    @classmethod
    def get_daily_summary(cls, target_date):
        """Günlük özet"""
        predictions = cls.objects.filter(date=target_date).order_by('hour')
        
        if not predictions.exists():
            return None
        
        prices = [float(p.predicted_price) for p in predictions]
        
        return {
            'date': target_date,
            'min_price': min(prices),
            'max_price': max(prices),
            'avg_price': sum(prices) / len(prices),
            'min_hour': prices.index(min(prices)),
            'max_hour': prices.index(max(prices)),
            'predictions': list(predictions.values('hour', 'predicted_price', 'confidence')),
        }
    
    @classmethod
    def get_next_72h(cls):
        """Önümüzdeki 72 saatlik tahminler"""
        now = timezone.now()
        predictions = []
        
        for i in range(72):
            target_dt = now + timedelta(hours=i)
            target_date = target_dt.date()
            target_hour = target_dt.hour
            
            pred = cls.objects.filter(date=target_date, hour=target_hour).first()
            if pred:
                predictions.append(pred)
        
        return predictions


class HistoricalPTF(models.Model):
    """
    Geçmiş PTF verileri (EPİAŞ'tan çekilen).
    Model eğitimi ve karşılaştırma için kullanılır.
    """
    
    date = models.DateField('Tarih', db_index=True)
    hour = models.IntegerField('Saat')
    
    # Fiyatlar
    ptf = models.DecimalField('PTF (TL/MWh)', max_digits=10, decimal_places=2)
    smf = models.DecimalField('SMF (TL/MWh)', max_digits=10, decimal_places=2, null=True, blank=True)
    
    # Sistem verileri
    load = models.DecimalField('Yük (MWh)', max_digits=12, decimal_places=2, null=True, blank=True)
    wind_generation = models.DecimalField('Rüzgar üretimi (MWh)', max_digits=10, decimal_places=2, null=True, blank=True)
    solar_generation = models.DecimalField('Güneş üretimi (MWh)', max_digits=10, decimal_places=2, null=True, blank=True)
    
    class Meta:
        verbose_name = 'Geçmiş PTF'
        verbose_name_plural = 'Geçmiş PTF Verileri'
        ordering = ['-date', 'hour']
        unique_together = ['date', 'hour']
    
    def __str__(self):
        return f"{self.date} {self.hour:02d}:00 - {self.ptf} TL"


class ModelPerformance(models.Model):
    """
    Model performans metrikleri.
    Günlük/haftalık takip için.
    """
    
    date = models.DateField('Tarih', unique=True)
    
    # Metrikler
    mape = models.FloatField('MAPE (%)', null=True, blank=True)
    rmse = models.FloatField('RMSE (TL)', null=True, blank=True)
    mae = models.FloatField('MAE (TL)', null=True, blank=True)
    r2_score = models.FloatField('R² Skoru', null=True, blank=True)
    
    # İstatistikler
    total_predictions = models.IntegerField('Toplam tahmin', default=0)
    accurate_predictions = models.IntegerField('Doğru tahmin (<%10 hata)', default=0)
    
    model_version = models.CharField('Model versiyonu', max_length=50)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Model Performansı'
        verbose_name_plural = 'Model Performansları'
        ordering = ['-date']
    
    def __str__(self):
        return f"{self.date} - MAPE: {self.mape:.2f}%"
    
    @property
    def accuracy_rate(self):
        """Doğruluk oranı"""
        if self.total_predictions > 0:
            return self.accurate_predictions / self.total_predictions * 100
        return 0
