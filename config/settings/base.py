"""
Django Base Settings - PTF Tahmin SaaS Platformu
================================================
Tüm ortamlar için ortak ayarlar.
"""

import os
from pathlib import Path
from datetime import timedelta

# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ============================================================================
# UYGULAMA TANIMLARI (INSTALLED_APPS) - DÜZELTİLDİ
# ============================================================================
INSTALLED_APPS = [
    # Django Apps
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'whitenoise.runserver_nostatic', # Whitenoise (Geliştirme ortamı için)
    'django.contrib.staticfiles',
    'django.contrib.sites',  # Allauth için gerekli

    # Third Party Apps
    'rest_framework',
    'rest_framework_simplejwt',
    'rest_framework_simplejwt.token_blacklist',
    'drf_spectacular',
    'corsheaders',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.google', # Google girişi için
    'crispy_forms',
    'crispy_tailwind',
    'django_celery_beat',

    # Local Apps (Senin Uygulamaların)
    'apps.core',
    'apps.users',
    'apps.predictions',
    'apps.subscriptions',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'allauth.account.middleware.AccountMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'apps.subscriptions.context_processors.subscription_context',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'tr'
TIME_ZONE = 'Europe/Istanbul'
USE_I18N = True
USE_TZ = True

# Email Settings
EMAIL_BACKEND = os.environ.get('EMAIL_BACKEND', 'django.core.mail.backends.console.EmailBackend')
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USE_TLS = os.environ.get('EMAIL_USE_TLS', 'True') == 'True'
EMAIL_USE_SSL = os.environ.get('EMAIL_USE_SSL', 'False') == 'True'
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD', '')
DEFAULT_FROM_EMAIL = os.environ.get('DEFAULT_FROM_EMAIL', 'PTF Tahmin <noreply@ptftahmin.com>')

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Custom User Model
AUTH_USER_MODEL = 'users.CustomUser'

# Site ID for allauth
SITE_ID = 1

# ============================================================================
# DJANGO-ALLAUTH AYARLARI
# ============================================================================
AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
]

ACCOUNT_LOGIN_METHODS = {'email'}
ACCOUNT_EMAIL_VERIFICATION = 'mandatory'
ACCOUNT_SIGNUP_FIELDS = ['email', 'password'] # Yıldızları kaldırdım, hata yapabiliyor
ACCOUNT_UNIQUE_EMAIL = True
ACCOUNT_LOGIN_ON_EMAIL_CONFIRMATION = True
ACCOUNT_SESSION_REMEMBER = True
ACCOUNT_LOGOUT_ON_GET = False

# Rate limiting
ACCOUNT_RATE_LIMITS = {
    "login": "5/m",
    "login_failed": "3/m",
    "signup": "5/m",
    "password_reset": "3/m",
}

LOGIN_REDIRECT_URL = '/dashboard/'
LOGOUT_REDIRECT_URL = '/'
LOGIN_URL = '/accounts/login/'

# ============================================================================
# DJANGO REST FRAMEWORK
# ============================================================================
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',
        'user': '1000/hour',
    },
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 100,
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
}
SPECTACULAR_SETTINGS = {
    'TITLE': 'PTF Tahmin API',
    'DESCRIPTION': 'Elektrik fiyat tahmin API\'si',
    'VERSION': '1.0.0',
}

# JWT Settings
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=30),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'ALGORITHM': 'HS256',
    'AUTH_HEADER_TYPES': ('Bearer',),
}

# ============================================================================
# CELERY AYARLARI
# ============================================================================
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 dakika
CELERY_BEAT_SCHEDULER = 'django_celery_beat.schedulers:DatabaseScheduler'

# ============================================================================
# CRISPY FORMS
# ============================================================================
CRISPY_ALLOWED_TEMPLATE_PACKS = "tailwind"
CRISPY_TEMPLATE_PACK = "tailwind"

# ============================================================================
# PTF TAHMİN AYARLARI
# ============================================================================
PTF_MODEL_PATH = BASE_DIR / 'ml_models' / 'ptf_model.pkl'
PTF_PREDICTION_HORIZON = 72  # 72 saat
PTF_UPDATE_HOUR = 6  # Her gün saat 06:00'da güncelle

# ============================================================================
# ABONELİK PLANLARI
# ============================================================================
SUBSCRIPTION_PLANS = {
    'free': {
        'name': 'Ücretsiz',
        'price': 0,
        'api_calls_per_day': 10,
        'history_days': 7,
        'features': ['Günlük PTF özeti', '7 günlük geçmiş']
    },
    'basic': {
        'name': 'Başlangıç',
        'price': 299,  # TL/ay
        'api_calls_per_day': 100,
        'history_days': 30,
        'features': ['Saatlik tahminler', '30 günlük geçmiş', 'Email bildirimleri']
    },
    'pro': {
        'name': 'Profesyonel',
        'price': 799,  # TL/ay
        'api_calls_per_day': 1000,
        'history_days': 365,
        'features': ['72 saatlik tahmin', 'API erişimi', 'Sınırsız geçmiş', 'Öncelikli destek']
    },
    'enterprise': {
        'name': 'Kurumsal',
        'price': 2499,  # TL/ay
        'api_calls_per_day': 10000,
        'history_days': -1,  # Sınırsız
        'features': ['Tüm özellikler', 'Özel API limiti', 'Dedicated destek', 'SLA garantisi']
    },
}

# Site URL (email linkleri için)
SITE_URL = os.environ.get('SITE_URL', 'http://127.0.0.1:8000')
