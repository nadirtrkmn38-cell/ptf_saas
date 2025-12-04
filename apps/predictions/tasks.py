"""
Predictions App - Celery Tasks
==============================
Zamanlanmış görevler: günlük tahmin güncelleme, cache yenileme, performans hesaplama.
"""

from celery import shared_task
from django.core.cache import cache
from django.utils import timezone
from django.conf import settings
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

@shared_task
def send_daily_report():
    """Her gün sabah 06:00'da rapor gönder"""
    from apps.users.models import CustomUser
    from .services import PTFPredictionService
    
    service = PTFPredictionService()
    summary = service.get_daily_summary()
    
    if not summary:
        return
    
    # Premium kullanıcılara email gönder
    premium_users = CustomUser.objects.filter(
        subscription_plan__in=['basic', 'pro', 'enterprise'],
        subscription_expires__gte=timezone.now()
    )
    
    subject = 'Günlük PTF Tahmin Raporu - ' + summary['date']
    
    message = """
    Günlük PTF Tahmin Özeti
    =======================
    
    Tarih: {date}
    
    Minimum Fiyat: {min_price} TL/MWh (Saat {min_hour}:00)
    Maximum Fiyat: {max_price} TL/MWh (Saat {max_hour}:00)
    Ortalama Fiyat: {avg_price} TL/MWh
    
    Detaylı tahminler için: {url}
    """.format(
        date=summary['date'],
        min_price=summary['min_price'],
        max_price=summary['max_price'],
        min_hour=summary['min_hour'],
        max_hour=summary['max_hour'],
        avg_price=summary['avg_price'],
        url=settings.SITE_URL + '/predictions/'
    )
    
    for user in premium_users:
        try:
            send_mail(
                subject=subject,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
            )
        except Exception as e:
            logger.error("Email error: " + str(e))
    
    logger.info("Daily report sent to " + str(premium_users.count()) + " users")

@shared_task(bind=True, max_retries=3)
def update_daily_predictions(self):
    """
    Günlük PTF tahminlerini güncelle.
    Her gün 06:00'da çalışır.
    """
    try:
        from .services import PTFPredictionService
        from .models import PTFPrediction
        
        logger.info("Günlük tahmin güncelleme başladı...")
        
        service = PTFPredictionService()
        predictions = service.predict_next_72h()
        
        # Veritabanına kaydet
        saved_count = 0
        for pred in predictions:
            PTFPrediction.objects.update_or_create(
                date=pred['date'],
                hour=pred['hour'],
                defaults={
                    'predicted_price': pred['predicted_price'],
                    'lower_bound': pred.get('lower_bound'),
                    'upper_bound': pred.get('upper_bound'),
                    'confidence': pred.get('confidence', 0),
                    'model_version': 'v1.0',
                }
            )
            saved_count += 1
        
        # Cache'i temizle
        cache.delete('ptf_predictions_72h')
        
        logger.info(f"Günlük tahmin güncelleme tamamlandı: {saved_count} tahmin kaydedildi")
        
        return {
            'status': 'success',
            'predictions_saved': saved_count,
            'timestamp': timezone.now().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"Tahmin güncelleme hatası: {exc}")
        self.retry(exc=exc, countdown=300)  # 5 dakika sonra tekrar dene


@shared_task
def refresh_prediction_cache():
    """
    Tahmin cache'ini yenile.
    Her saat başı çalışır.
    """
    try:
        from .services import PTFPredictionService
        
        service = PTFPredictionService()
        predictions = service.predict_next_72h()
        
        # Cache'e yaz
        cache.set('ptf_predictions_72h', predictions, 3600)
        
        logger.info("Tahmin cache'i yenilendi")
        return {'status': 'success'}
        
    except Exception as e:
        logger.error(f"Cache yenileme hatası: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def update_actual_prices():
    """
    Gerçekleşen fiyatları EPİAŞ'tan çek ve tahminlerle karşılaştır.
    Her gün 00:30'da çalışır (önceki günün verileri için).
    """
    try:
        from .models import PTFPrediction, HistoricalPTF
        
        yesterday = timezone.now().date() - timedelta(days=1)
        
        # EPİAŞ'tan gerçek değerleri çek
        try:
            from eptr2 import EPTR
            eptr = EPTR()
            ptf_data = eptr.ptf(start_date=yesterday, end_date=yesterday)
            
            for idx, row in ptf_data.iterrows():
                hour = idx.hour if hasattr(idx, 'hour') else 0
                
                # Historical kaydet
                HistoricalPTF.objects.update_or_create(
                    date=yesterday,
                    hour=hour,
                    defaults={'ptf': row.get('ptf', 0)}
                )
                
                # Tahmin ile karşılaştır
                prediction = PTFPrediction.objects.filter(
                    date=yesterday,
                    hour=hour
                ).first()
                
                if prediction:
                    prediction.actual_price = row.get('ptf', 0)
                    prediction.save()
            
            logger.info(f"Gerçek fiyatlar güncellendi: {yesterday}")
            
        except ImportError:
            logger.warning("eptr2 yüklü değil, gerçek fiyatlar güncellenemedi")
        
        return {'status': 'success', 'date': str(yesterday)}
        
    except Exception as e:
        logger.error(f"Gerçek fiyat güncelleme hatası: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def calculate_daily_performance():
    """
    Günlük model performansını hesapla.
    Her gün 01:00'da çalışır.
    """
    try:
        from .models import PTFPrediction, ModelPerformance
        import numpy as np
        
        yesterday = timezone.now().date() - timedelta(days=1)
        
        predictions = PTFPrediction.objects.filter(
            date=yesterday,
            actual_price__isnull=False
        )
        
        if not predictions.exists():
            logger.info(f"Performans hesabı için yeterli veri yok: {yesterday}")
            return {'status': 'skipped', 'reason': 'no data'}
        
        # Metrikleri hesapla
        y_true = [float(p.actual_price) for p in predictions]
        y_pred = [float(p.predicted_price) for p in predictions]
        
        errors = [abs(t - p) for t, p in zip(y_true, y_pred)]
        percent_errors = [abs(t - p) / t * 100 for t, p in zip(y_true, y_pred) if t > 0]
        
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        mape = np.mean(percent_errors) if percent_errors else None
        
        # Doğru tahmin sayısı (<%10 hata)
        accurate = sum(1 for pe in percent_errors if pe < 10)
        
        # Kaydet
        ModelPerformance.objects.update_or_create(
            date=yesterday,
            defaults={
                'mape': mape,
                'rmse': rmse,
                'mae': mae,
                'total_predictions': len(predictions),
                'accurate_predictions': accurate,
                'model_version': 'v1.0',
            }
        )
        
        logger.info(f"Performans hesaplandı: {yesterday} - MAPE: {mape:.2f}%")
        
        return {
            'status': 'success',
            'date': str(yesterday),
            'mape': mape,
            'rmse': rmse
        }
        
    except Exception as e:
        logger.error(f"Performans hesaplama hatası: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def generate_weekly_report():
    """
    Haftalık performans raporu oluştur ve email gönder.
    Her Pazartesi 08:00'da çalışır.
    """
    try:
        from .models import ModelPerformance
        from django.core.mail import send_mail
        from apps.users.models import CustomUser
        
        # Son 7 günün performansı
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=7)
        
        performances = ModelPerformance.objects.filter(
            date__gte=start_date,
            date__lt=end_date
        ).order_by('date')
        
        if not performances.exists():
            return {'status': 'skipped', 'reason': 'no data'}
        
        # Özet hesapla
        avg_mape = sum(p.mape for p in performances if p.mape) / performances.count()
        avg_rmse = sum(p.rmse for p in performances if p.rmse) / performances.count()
        
        report = f"""
        PTF Tahmin Haftalık Performans Raporu
        =====================================
        Dönem: {start_date} - {end_date}
        
        Ortalama MAPE: {avg_mape:.2f}%
        Ortalama RMSE: {avg_rmse:.2f} TL
        
        Günlük Detay:
        """
        
        for perf in performances:
            report += f"\n{perf.date}: MAPE={perf.mape:.2f}%, Doğruluk={perf.accuracy_rate:.1f}%"
        
        # Admin kullanıcılara email gönder
        admins = CustomUser.objects.filter(is_staff=True, is_active=True)
        
        for admin in admins:
            send_mail(
                subject=f'PTF Tahmin Haftalık Rapor - {start_date} / {end_date}',
                message=report,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[admin.email],
                fail_silently=True,
            )
        
        logger.info(f"Haftalık rapor gönderildi: {admins.count()} admin")
        
        return {
            'status': 'success',
            'avg_mape': avg_mape,
            'emails_sent': admins.count()
        }
        
    except Exception as e:
        logger.error(f"Haftalık rapor hatası: {e}")
        return {'status': 'error', 'message': str(e)}
