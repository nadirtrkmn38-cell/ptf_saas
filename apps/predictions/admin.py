"""
Predictions App - Admin Configuration
"""

from django.contrib import admin
from .models import PTFPrediction, HistoricalPTF, ModelPerformance


@admin.register(PTFPrediction)
class PTFPredictionAdmin(admin.ModelAdmin):
    list_display = ['date', 'hour', 'predicted_price', 'actual_price', 'confidence', 'model_version']
    list_filter = ['date', 'model_version']
    search_fields = ['date']
    ordering = ['-date', 'hour']
    date_hierarchy = 'date'
    
    readonly_fields = ['created_at', 'updated_at']


@admin.register(HistoricalPTF)
class HistoricalPTFAdmin(admin.ModelAdmin):
    list_display = ['date', 'hour', 'ptf', 'smf', 'load']
    list_filter = ['date']
    search_fields = ['date']
    ordering = ['-date', 'hour']
    date_hierarchy = 'date'


@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ['date', 'mape', 'rmse', 'mae', 'accuracy_rate', 'model_version']
    list_filter = ['model_version', 'date']
    ordering = ['-date']
    date_hierarchy = 'date'
    readonly_fields = ['created_at']
