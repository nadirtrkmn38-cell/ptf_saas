"""
Users App - Models
==================
Custom User model ve Profile.
"""

from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from django.utils import timezone


class CustomUserManager(BaseUserManager):
    """
    Custom user manager - email ile authentication.
    """
    
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('Email adresi zorunludur')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user
    
    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        
        return self.create_user(email, password, **extra_fields)


class CustomUser(AbstractUser):
    """
    Custom User Model - Email based authentication.
    """
    
    SUBSCRIPTION_PLANS = [
        ('free', 'Ücretsiz'),
        ('basic', 'Başlangıç'),
        ('pro', 'Profesyonel'),
        ('enterprise', 'Kurumsal'),
    ]
    
    # Username'i kaldır, email kullan
    username = None
    email = models.EmailField('Email adresi', unique=True, db_index=True)
    
    # Profil bilgileri
    phone = models.CharField('Telefon', max_length=20, blank=True)
    company_name = models.CharField('Şirket adı', max_length=255, blank=True)
    job_title = models.CharField('Ünvan', max_length=100, blank=True)
    
    # Abonelik bilgileri
    subscription_plan = models.CharField(
        'Abonelik planı',
        max_length=20,
        choices=SUBSCRIPTION_PLANS,
        default='free'
    )
    subscription_expires = models.DateTimeField(
        'Abonelik bitiş tarihi',
        null=True,
        blank=True
    )
    
    # API erişimi
    api_key = models.CharField('API anahtarı', max_length=64, unique=True, null=True, blank=True)
    api_calls_today = models.IntegerField('Bugünkü API çağrısı', default=0)
    api_calls_reset_date = models.DateField('API sayaç sıfırlama tarihi', null=True, blank=True)
    
    # KVKK ve izinler
    kvkk_consent = models.BooleanField('KVKK onayı', default=False)
    kvkk_consent_date = models.DateTimeField('KVKK onay tarihi', null=True, blank=True)
    marketing_consent = models.BooleanField('Pazarlama izni', default=False)
    
    # Meta
    created_at = models.DateTimeField('Oluşturulma tarihi', auto_now_add=True)
    updated_at = models.DateTimeField('Güncellenme tarihi', auto_now=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    
    objects = CustomUserManager()
    
    class Meta:
        verbose_name = 'Kullanıcı'
        verbose_name_plural = 'Kullanıcılar'
        ordering = ['-created_at']
    
    def __str__(self):
        return self.email
    
    def get_full_name(self):
        return f"{self.first_name} {self.last_name}".strip() or self.email
    
    def has_active_subscription(self):
        """Aktif abonelik var mı?"""
        if self.subscription_plan == 'free':
            return True
        if not self.subscription_expires:
            return False
        return self.subscription_expires > timezone.now()
    
    def is_premium(self):
        """Premium kullanıcı mı?"""
        return self.subscription_plan in ['pro', 'enterprise'] and self.has_active_subscription()
    
    def get_api_limit(self):
        """Günlük API limiti"""
        limits = {
            'free': 10,
            'basic': 100,
            'pro': 1000,
            'enterprise': 10000,
        }
        return limits.get(self.subscription_plan, 10)
    
    def can_make_api_call(self):
        """API çağrısı yapabilir mi?"""
        # Günlük sayacı sıfırla
        today = timezone.now().date()
        if self.api_calls_reset_date != today:
            self.api_calls_today = 0
            self.api_calls_reset_date = today
            self.save(update_fields=['api_calls_today', 'api_calls_reset_date'])
        
        return self.api_calls_today < self.get_api_limit()
    
    def increment_api_calls(self):
        """API çağrısı sayacını artır"""
        self.api_calls_today += 1
        self.save(update_fields=['api_calls_today'])
    
    def generate_api_key(self):
        """Yeni API anahtarı oluştur"""
        import secrets
        self.api_key = secrets.token_hex(32)
        self.save(update_fields=['api_key'])
        return self.api_key


class UserActivity(models.Model):
    """
    Kullanıcı aktivite logu.
    """
    
    ACTION_TYPES = [
        ('login', 'Giriş'),
        ('logout', 'Çıkış'),
        ('api_call', 'API çağrısı'),
        ('prediction_view', 'Tahmin görüntüleme'),
        ('subscription_change', 'Abonelik değişikliği'),
        ('payment', 'Ödeme'),
    ]
    
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='activities')
    action = models.CharField('Aksiyon', max_length=30, choices=ACTION_TYPES)
    details = models.JSONField('Detaylar', default=dict, blank=True)
    ip_address = models.GenericIPAddressField('IP adresi', null=True, blank=True)
    user_agent = models.TextField('User agent', blank=True)
    created_at = models.DateTimeField('Tarih', auto_now_add=True)
    
    class Meta:
        verbose_name = 'Kullanıcı aktivitesi'
        verbose_name_plural = 'Kullanıcı aktiviteleri'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.email} - {self.action}"
