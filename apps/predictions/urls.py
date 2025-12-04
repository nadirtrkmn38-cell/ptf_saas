"""
Predictions App - URLs
"""

from django.urls import path
from . import views

urlpatterns = [
    # Web views
    path('', views.predictions_list, name='predictions'),
    path('<str:date>/', views.prediction_detail, name='prediction_detail'),
    path('chart/<str:date>/', views.chart_data, name='chart_data'),
    path('performance/', views.model_performance, name='model_performance'),
]
