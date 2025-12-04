"""
API App - Views
===============
Django REST Framework ile PTF tahmin API'si.
"""

from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.throttling import UserRateThrottle
from django.utils import timezone
from datetime import timedelta

from apps.predictions.services import PTFPredictionService
from apps.predictions.models import PTFPrediction, HistoricalPTF, ModelPerformance
from .serializers import (
    PTFPredictionSerializer,
    HistoricalPTFSerializer,
    DailySummarySerializer,
    ModelPerformanceSerializer
)
from .permissions import IsPremiumUser, HasAPIAccess
from .permissions import SubscriptionBasedThrottle


class PTFPredictionViewSet(viewsets.ReadOnlyModelViewSet):
    """
    PTF tahminleri API.
    
    list: Tüm tahminleri listele
    retrieve: Belirli bir tahmini getir
    """
    serializer_class = PTFPredictionSerializer
    permission_classes = [IsAuthenticated, HasAPIAccess]
    throttle_classes = [SubscriptionBasedThrottle]
    
    def get_queryset(self):
        queryset = PTFPrediction.objects.all()
        
        # Tarih filtresi
        date = self.request.query_params.get('date')
        if date:
            queryset = queryset.filter(date=date)
        
        # Free kullanıcılar sadece yarını görebilir
        if not self.request.user.is_premium():
            tomorrow = timezone.now().date() + timedelta(days=1)
            queryset = queryset.filter(date=tomorrow)
        
        return queryset.order_by('date', 'hour')


@api_view(['GET'])
@permission_classes([IsAuthenticated, HasAPIAccess])
@throttle_classes([SubscriptionBasedThrottle])
def predictions_72h(request):
    """
    Önümüzdeki 72 saatlik tahminler.
    
    Premium: 72 saat
    Free: 24 saat
    """
    # API çağrısı sayacını artır
    if not request.user.can_make_api_call():
        return Response({
            'error': 'Günlük API limitinize ulaştınız',
            'limit': request.user.get_api_limit(),
            'used': request.user.api_calls_today,
        }, status=status.HTTP_429_TOO_MANY_REQUESTS)
    
    request.user.increment_api_calls()
    
    service = PTFPredictionService()
    predictions = service.predict_next_72h()
    
    # Plan bazlı kısıtlama
    if not request.user.is_premium():
        predictions = predictions[:24]
    
    return Response({
        'status': 'success',
        'user': request.user.email,
        'plan': request.user.subscription_plan,
        'count': len(predictions),
        'predictions': predictions,
        'api_calls_remaining': request.user.get_api_limit() - request.user.api_calls_today,
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated, HasAPIAccess])
def daily_summary(request, date):
    """
    Belirli bir günün özeti.
    
    URL: /api/v1/summary/{date}/
    """
    if not request.user.can_make_api_call():
        return Response({
            'error': 'Günlük API limitinize ulaştınız'
        }, status=status.HTTP_429_TOO_MANY_REQUESTS)
    
    request.user.increment_api_calls()
    
    try:
        from datetime import datetime
        target_date = datetime.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        return Response({
            'error': 'Geçersiz tarih formatı. YYYY-MM-DD kullanın.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    service = PTFPredictionService()
    summary = service.get_daily_summary(target_date)
    
    if not summary:
        return Response({
            'error': 'Bu tarih için tahmin bulunamadı'
        }, status=status.HTTP_404_NOT_FOUND)
    
    # Free kullanıcılar için kısıtla
    if not request.user.is_premium():
        summary['predictions'] = summary['predictions'][:6]
    
    return Response({
        'status': 'success',
        **summary
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated, IsPremiumUser])
def historical_data(request):
    """
    Geçmiş PTF verileri.
    Premium only.
    
    Query params:
        start_date: Başlangıç tarihi (YYYY-MM-DD)
        end_date: Bitiş tarihi (YYYY-MM-DD)
    """
    if not request.user.can_make_api_call():
        return Response({
            'error': 'Günlük API limitinize ulaştınız'
        }, status=status.HTTP_429_TOO_MANY_REQUESTS)
    
    request.user.increment_api_calls()
    
    # Tarih parametreleri
    from datetime import datetime
    
    end_date = request.query_params.get('end_date', timezone.now().date().isoformat())
    start_date = request.query_params.get('start_date')
    
    try:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            # Plan bazlı varsayılan süre
            history_days = request.user.get_history_days()
            start_dt = end_dt - timedelta(days=history_days)
        
    except ValueError:
        return Response({
            'error': 'Geçersiz tarih formatı'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Veriyi çek
    data = HistoricalPTF.objects.filter(
        date__gte=start_dt,
        date__lte=end_dt
    ).order_by('date', 'hour')
    
    serializer = HistoricalPTFSerializer(data, many=True)
    
    return Response({
        'status': 'success',
        'start_date': str(start_dt),
        'end_date': str(end_dt),
        'count': data.count(),
        'data': serializer.data
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated, IsPremiumUser])
def model_performance_api(request):
    """
    Model performans metrikleri.
    Premium only.
    """
    days = int(request.query_params.get('days', 30))
    days = min(days, 365)  # Max 1 yıl
    
    start_date = timezone.now().date() - timedelta(days=days)
    
    performances = ModelPerformance.objects.filter(
        date__gte=start_date
    ).order_by('-date')
    
    serializer = ModelPerformanceSerializer(performances, many=True)
    
    # Özet hesapla
    if performances.exists():
        avg_mape = sum(p.mape for p in performances if p.mape) / performances.count()
        avg_rmse = sum(p.rmse for p in performances if p.rmse) / performances.count()
    else:
        avg_mape = avg_rmse = None
    
    return Response({
        'status': 'success',
        'period_days': days,
        'avg_mape': avg_mape,
        'avg_rmse': avg_rmse,
        'performances': serializer.data
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def api_status(request):
    """
    API durum bilgisi.
    Public endpoint.
    """
    return Response({
        'status': 'online',
        'version': 'v1',
        'timestamp': timezone.now().isoformat(),
        'endpoints': {
            'predictions': '/api/v1/predictions/',
            'predictions_72h': '/api/v1/predictions/72h/',
            'daily_summary': '/api/v1/summary/{date}/',
            'historical': '/api/v1/historical/',
            'performance': '/api/v1/performance/',
        }
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_usage(request):
    """
    Kullanıcının API kullanım bilgileri.
    """
    return Response({
        'email': request.user.email,
        'plan': request.user.subscription_plan,
        'api_limit': request.user.get_api_limit(),
        'api_calls_today': request.user.api_calls_today,
        'api_calls_remaining': request.user.get_api_limit() - request.user.api_calls_today,
        'subscription_expires': request.user.subscription_expires,
        'is_premium': request.user.is_premium(),
    })
