"""
API App - URLs
==============
REST API endpoints.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)
from . import views

# Router for viewsets
router = DefaultRouter()
router.register(r'predictions', views.PTFPredictionViewSet, basename='predictions')

urlpatterns = [
    # JWT Authentication
    path('v1/auth/token/', TokenObtainPairView.as_view(), name='token_obtain'),
    path('v1/auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('v1/auth/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    
    # API Status
    path('v1/status/', views.api_status, name='api_status'),
    path('v1/usage/', views.api_usage, name='api_usage'),
    
    # Predictions
    path('v1/predictions/72h/', views.predictions_72h, name='predictions_72h'),
    path('v1/summary/<str:date>/', views.daily_summary, name='daily_summary'),
    
    # Historical Data (Premium)
    path('v1/historical/', views.historical_data, name='historical_data'),
    
    # Model Performance (Premium)
    path('v1/performance/', views.model_performance_api, name='model_performance'),
    
    # Router URLs
    path('v1/', include(router.urls)),
]
