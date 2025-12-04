"""
PTF Tahmin SaaS Platform - URL Configuration
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # Authentication (django-allauth)
    path('accounts/', include('allauth.urls')),
    
    # Dashboard (ana uygulama)
    path('', include('apps.dashboard.urls')),
    
    # Predictions
    path('predictions/', include('apps.predictions.urls')),
    
    # Subscriptions
    path('subscriptions/', include('apps.subscriptions.urls')),
    
    # API
    path('api/', include('apps.api.urls')),
]

# Development için static ve media dosyaları
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    
    # Debug toolbar
    try:
        import debug_toolbar
        urlpatterns = [
            path('__debug__/', include(debug_toolbar.urls)),
        ] + urlpatterns
    except ImportError:
        pass

# Custom error handlers
handler404 = 'apps.dashboard.views.error_404'
handler500 = 'apps.dashboard.views.error_500'
