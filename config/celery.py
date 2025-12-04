"""
Celery Configuration - PTF Tahmin SaaS Platform
================================================
Günlük tahmin güncelleme ve periyodik görevler için.
"""

import os
from celery import Celery
from celery.schedules import crontab

# Django settings modülünü ayarla
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

# Celery app oluştur
app = Celery('ptf_saas')

# Django settings'den konfigürasyonu oku
app.config_from_object('django.conf:settings', namespace='CELERY')

# Tüm Django app'lerden tasks'ları otomatik keşfet
app.autodiscover_tasks()

# ============================================================================
# ZAMANLANMIŞ GÖREVLER
# ============================================================================
app.conf.beat_schedule = {
    # Her gün saat 06:00'da PTF tahminlerini güncelle
    'update-predictions-daily': {
        'task': 'apps.predictions.tasks.update_daily_predictions',
        'schedule': crontab(hour=6, minute=0),
    },
    
    # Her saat başı tahmin cache'ini güncelle
    'refresh-prediction-cache': {
        'task': 'apps.predictions.tasks.refresh_prediction_cache',
        'schedule': crontab(minute=0),
        'options': {'queue': 'default'}
    },
    
    # Her gün gece yarısı abonelik durumlarını kontrol et
    'check-subscription-expirations': {
        'task': 'apps.subscriptions.tasks.check_subscription_expirations',
        'schedule': crontab(hour=0, minute=5),
        'options': {'queue': 'subscriptions'}
    },
    
    # Her gün aboneliği bitmek üzere olanlara email gönder
    'send-expiration-reminders': {
        'task': 'apps.subscriptions.tasks.send_expiration_reminders',
        'schedule': crontab(hour=10, minute=0),
        'options': {'queue': 'emails'}
    },
    
    # Her hafta performans raporu oluştur
    'generate-weekly-report': {
        'task': 'apps.predictions.tasks.generate_weekly_report',
        'schedule': crontab(hour=8, minute=0, day_of_week=1),  # Pazartesi 08:00
        'options': {'queue': 'reports'}
    },

    'send-daily-report': {
        'task': 'apps.predictions.tasks.send_daily_report',
        'schedule': crontab(hour=7, minute=0),
    }
}  # <--- KRİTİK DÜZELTME: BURADAKİ KAPATMA PARANTEZİ EKSİKTİ!

# Queue'ları tanımla
app.conf.task_queues = {
    'default': {},
    'predictions': {},
    'subscriptions': {},
    'emails': {},
    'reports': {},
}

# Task routing
app.conf.task_routes = {
    'apps.predictions.tasks.*': {'queue': 'predictions'},
    'apps.subscriptions.tasks.*': {'queue': 'subscriptions'},
}


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    """Debug için test task"""
    print(f'Request: {self.request!r}')