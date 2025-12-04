"""
PTF Tahmin SaaS Platform - Config Package
"""

# Celery app'i Django başlatıldığında yükle
from .celery import app as celery_app

__all__ = ('celery_app',)
