"""
API App - Serializers
=====================
DRF Serializers for PTF models.
"""

from rest_framework import serializers
from apps.predictions.models import PTFPrediction, HistoricalPTF, ModelPerformance


class PTFPredictionSerializer(serializers.ModelSerializer):
    """PTF Tahmin serializer"""
    
    datetime = serializers.SerializerMethodField()
    error = serializers.SerializerMethodField()
    error_percent = serializers.SerializerMethodField()
    
    class Meta:
        model = PTFPrediction
        fields = [
            'date', 'hour', 'datetime',
            'predicted_price', 'lower_bound', 'upper_bound',
            'confidence', 'actual_price', 'error', 'error_percent',
            'model_version', 'created_at'
        ]
    
    def get_datetime(self, obj):
        return obj.datetime.isoformat() if obj.datetime else None
    
    def get_error(self, obj):
        return obj.error
    
    def get_error_percent(self, obj):
        return obj.error_percent


class HistoricalPTFSerializer(serializers.ModelSerializer):
    """Geçmiş PTF serializer"""
    
    class Meta:
        model = HistoricalPTF
        fields = ['date', 'hour', 'ptf', 'smf', 'load', 'wind_generation', 'solar_generation']


class DailySummarySerializer(serializers.Serializer):
    """Günlük özet serializer"""
    
    date = serializers.DateField()
    min_price = serializers.FloatField()
    max_price = serializers.FloatField()
    avg_price = serializers.FloatField()
    min_hour = serializers.IntegerField()
    max_hour = serializers.IntegerField()
    predictions = serializers.ListField()


class ModelPerformanceSerializer(serializers.ModelSerializer):
    """Model performans serializer"""
    
    accuracy_rate = serializers.SerializerMethodField()
    
    class Meta:
        model = ModelPerformance
        fields = [
            'date', 'mape', 'rmse', 'mae', 'r2_score',
            'total_predictions', 'accurate_predictions',
            'accuracy_rate', 'model_version', 'created_at'
        ]
    
    def get_accuracy_rate(self, obj):
        return obj.accuracy_rate
