"""
Dashboard App - URLs
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    path('api-settings/', views.api_settings, name='api_settings'),
]
