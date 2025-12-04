"""
Subscriptions App - Models
==========================
Abonelik planları, aktif abonelikler ve ödemeler.
"""

from django.db import models
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
import uuid


class Plan(models.Model):
    """
    Abonelik planları.
    """
    
    INTERVAL_CHOICES = [
        ('monthly', 'Aylık'),
        ('yearly', 'Yıllık'),
    ]
    
    name = models.CharField('Plan adı', max_length=100)
    slug = models.SlugField('Slug', unique=True)
    description = models.TextField('Açıklama', blank=True)
    
    # Fiyatlandırma
    price = models.DecimalField('Fiyat (TL)', max_digits=10, decimal_places=2)
    interval = models.CharField('Ödeme aralığı', max_length=20, choices=INTERVAL_CHOICES, default='monthly')
    
    # Özellikler (JSON)
    features = models.JSONField('Özellikler', default=dict)
    # Örnek: {"api_calls": 1000, "history_days": 30, "priority_support": false}
    
    # API limitleri
    api_calls_per_day = models.IntegerField('Günlük API limiti', default=10)
    history_days = models.IntegerField('Geçmiş veri (gün)', default=7)
    
    # iyzico referansları
    iyzico_product_ref = models.CharField('iyzico ürün referansı', max_length=100, blank=True)
    iyzico_plan_ref = models.CharField('iyzico plan referansı', max_length=100, blank=True)
    
    # Durum
    is_active = models.BooleanField('Aktif', default=True)
    is_featured = models.BooleanField('Öne çıkan', default=False)
    sort_order = models.IntegerField('Sıralama', default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Plan'
        verbose_name_plural = 'Planlar'
        ordering = ['sort_order', 'price']
    
    def __str__(self):
        return f"{self.name} - {self.price} TL/{self.get_interval_display()}"
    
    def get_yearly_price(self):
        """Yıllık ödeme indirimi ile fiyat"""
        if self.interval == 'monthly':
            return self.price * 10  # 2 ay bedava
        return self.price
    
    def get_features_list(self):
        """Özellik listesi"""
        return self.features.get('feature_list', [])


class Subscription(models.Model):
    """
    Kullanıcı abonelikleri.
    """
    
    STATUS_CHOICES = [
        ('active', 'Aktif'),
        ('pending', 'Beklemede'),
        ('cancelled', 'İptal edildi'),
        ('expired', 'Süresi doldu'),
        ('unpaid', 'Ödenmemiş'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='subscriptions'
    )
    plan = models.ForeignKey(Plan, on_delete=models.PROTECT, related_name='subscriptions')
    
    status = models.CharField('Durum', max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Dönem bilgileri
    current_period_start = models.DateTimeField('Dönem başlangıcı')
    current_period_end = models.DateTimeField('Dönem sonu')
    
    # iyzico referansları
    iyzico_subscription_ref = models.CharField('iyzico abonelik referansı', max_length=100, unique=True, null=True, blank=True)
    iyzico_customer_ref = models.CharField('iyzico müşteri referansı', max_length=100, blank=True)
    
    # İptal bilgileri
    cancel_at_period_end = models.BooleanField('Dönem sonunda iptal', default=False)
    cancelled_at = models.DateTimeField('İptal tarihi', null=True, blank=True)
    cancellation_reason = models.TextField('İptal nedeni', blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Abonelik'
        verbose_name_plural = 'Abonelikler'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.email} - {self.plan.name}"
    
    def is_active(self):
        """Abonelik aktif mi?"""
        return self.status == 'active' and self.current_period_end > timezone.now()
    
    def days_until_expiry(self):
        """Bitiş tarihine kalan gün"""
        if not self.current_period_end:
            return 0
        delta = self.current_period_end - timezone.now()
        return max(0, delta.days)
    
    def renew(self):
        """Aboneliği yenile"""
        if self.plan.interval == 'monthly':
            self.current_period_end += timedelta(days=30)
        else:
            self.current_period_end += timedelta(days=365)
        self.status = 'active'
        self.save()
    
    def cancel(self, reason='', immediate=False):
        """Aboneliği iptal et"""
        self.cancellation_reason = reason
        self.cancelled_at = timezone.now()
        
        if immediate:
            self.status = 'cancelled'
            self.current_period_end = timezone.now()
        else:
            self.cancel_at_period_end = True
        
        self.save()


class Payment(models.Model):
    """
    Ödeme kayıtları.
    """
    
    STATUS_CHOICES = [
        ('pending', 'Beklemede'),
        ('completed', 'Tamamlandı'),
        ('failed', 'Başarısız'),
        ('refunded', 'İade edildi'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    subscription = models.ForeignKey(
        Subscription,
        on_delete=models.CASCADE,
        related_name='payments'
    )
    
    amount = models.DecimalField('Tutar', max_digits=10, decimal_places=2)
    currency = models.CharField('Para birimi', max_length=3, default='TRY')
    status = models.CharField('Durum', max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # iyzico bilgileri
    iyzico_payment_id = models.CharField('iyzico ödeme ID', max_length=100, unique=True, null=True, blank=True)
    iyzico_payment_transaction_id = models.CharField('iyzico işlem ID', max_length=100, blank=True)
    
    # Kart bilgileri (maskeli)
    card_last_four = models.CharField('Kart son 4 hane', max_length=4, blank=True)
    card_brand = models.CharField('Kart markası', max_length=20, blank=True)
    
    # Fatura bilgileri
    invoice_number = models.CharField('Fatura no', max_length=50, blank=True)
    invoice_url = models.URLField('Fatura URL', blank=True)
    
    # Tarihler
    payment_date = models.DateTimeField('Ödeme tarihi', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Hata bilgisi
    error_message = models.TextField('Hata mesajı', blank=True)
    
    class Meta:
        verbose_name = 'Ödeme'
        verbose_name_plural = 'Ödemeler'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.subscription.user.email} - {self.amount} TL - {self.status}"


class SubscriptionEvent(models.Model):
    """
    Abonelik olayları logu.
    """
    
    EVENT_TYPES = [
        ('created', 'Oluşturuldu'),
        ('activated', 'Aktifleştirildi'),
        ('renewed', 'Yenilendi'),
        ('payment_failed', 'Ödeme başarısız'),
        ('cancelled', 'İptal edildi'),
        ('expired', 'Süresi doldu'),
        ('upgraded', 'Yükseltildi'),
        ('downgraded', 'Düşürüldü'),
    ]
    
    subscription = models.ForeignKey(
        Subscription,
        on_delete=models.CASCADE,
        related_name='events'
    )
    event_type = models.CharField('Olay tipi', max_length=30, choices=EVENT_TYPES)
    data = models.JSONField('Veri', default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Abonelik olayı'
        verbose_name_plural = 'Abonelik olayları'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.subscription} - {self.event_type}"
