"""
Django Development Settings
"""

from .base import *

DEBUG = False

SECRET_KEY = 'django-insecure-dev-key-change-in-production-12345'

ALLOWED_HOSTS = ['*']

SITE_URL = 'http://127.0.0.1:8000'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Cache (development için local memory)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
    }
}

# Celery (development için eager mode)
CELERY_TASK_ALWAYS_EAGER = True
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

# Email (console backend for development)
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# CORS (development için tüm origins)
CORS_ALLOW_ALL_ORIGINS = True

# Allauth - development için email doğrulamayı opsiyonel yap
ACCOUNT_EMAIL_VERIFICATION = 'optional'

# Debug toolbar (opsiyonel)
try:
    import debug_toolbar
    INSTALLED_APPS += ['debug_toolbar']
    MIDDLEWARE.insert(0, 'debug_toolbar.middleware.DebugToolbarMiddleware')
    INTERNAL_IPS = ['127.0.0.1']
except ImportError:
    pass

# iyzico Sandbox
IYZICO_API_KEY = os.environ.get('IYZICO_API_KEY', 'sandbox-xxx')
IYZICO_SECRET_KEY = os.environ.get('IYZICO_SECRET_KEY', 'sandbox-xxx')
IYZICO_BASE_URL = 'https://sandbox-api.iyzipay.com'

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'apps': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}
