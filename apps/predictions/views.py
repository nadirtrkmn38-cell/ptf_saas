"""
Predictions App - Views
=======================
PTF tahmin görüntüleme sayfaları.
Premium içerik koruması ile.
"""

from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.utils import timezone
from datetime import timedelta
from functools import wraps

from .models import PTFPrediction, HistoricalPTF, ModelPerformance
from .services import PTFPredictionService


def premium_required(view_func):
    """Premium üyelik gerektiren view decorator"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({
                'error': 'Giriş yapmalısınız',
                'redirect': '/accounts/login/'
            }, status=401)
        
        if not request.user.is_premium():
            return JsonResponse({
                'error': 'Bu özellik premium üyelik gerektirir',
                'upgrade_url': '/subscriptions/pricing/'
            }, status=403)
        
        return view_func(request, *args, **kwargs)
    return wrapper


@login_required
def predictions_list(request):
    """
    Tahminler listesi.
    Free: Sadece bugünkü özet
    Premium: 72 saatlik detaylı tahmin
    """
    service = PTFPredictionService()
    
    context = {
        'today': timezone.now().date(),
        'is_premium': request.user.is_premium(),
    }
    
    if request.user.is_premium():
        # Premium: Tam 72 saatlik tahmin
        predictions = service.predict_next_72h()
        context['predictions'] = predictions
        context['next_3_days'] = [
            service.get_daily_summary(timezone.now().date() + timedelta(days=i))
            for i in range(1, 4)
        ]
    else:
        # Free: Sadece yarının özeti
        tomorrow_summary = service.get_daily_summary()
        context['tomorrow_summary'] = tomorrow_summary
        
        # Free kullanıcılar için kısıtlı veri
        if tomorrow_summary:
            context['limited_predictions'] = tomorrow_summary['predictions'][:6]  # İlk 6 saat
    
    return render(request, 'predictions/list.html', context)


@login_required
def prediction_detail(request, date):
    """
    Belirli bir günün detaylı tahmini.
    Premium only.
    """
    if not request.user.is_premium():
        return render(request, 'predictions/premium_required.html')
    
    predictions = PTFPrediction.objects.filter(date=date).order_by('hour')
    
    # Gerçek değerler varsa karşılaştır
    historical = HistoricalPTF.objects.filter(date=date).order_by('hour')
    
    return render(request, 'predictions/detail.html', {
        'date': date,
        'predictions': predictions,
        'historical': historical,
    })


@login_required
def chart_data(request, date):
    """
    Grafik için JSON veri.
    """
    predictions = PTFPrediction.objects.filter(date=date).order_by('hour')
    
    # Premium değilse sadece 6 saat göster
    if not request.user.is_premium():
        predictions = predictions[:6]
    
    data = {
        'labels': [f'{p.hour:02d}:00' for p in predictions],
        'datasets': [{
            'label': 'Tahmin (TL/MWh)',
            'data': [float(p.predicted_price) for p in predictions],
            'borderColor': '#3b82f6',
            'backgroundColor': 'rgba(59, 130, 246, 0.1)',
            'fill': True,
            'tension': 0.4
        }]
    }
    
    # Gerçek değerler varsa ekle
    historical = HistoricalPTF.objects.filter(date=date).order_by('hour')
    if historical.exists():
        if not request.user.is_premium():
            historical = historical[:6]
        
        data['datasets'].append({
            'label': 'Gerçek (TL/MWh)',
            'data': [float(h.ptf) for h in historical],
            'borderColor': '#10b981',
            'backgroundColor': 'rgba(16, 185, 129, 0.1)',
            'fill': False,
            'tension': 0.4
        })
    
    return JsonResponse(data)


@login_required
def model_performance(request):
    """
    Model performans metrikleri.
    Premium only.
    """
    if not request.user.is_premium():
        return render(request, 'predictions/premium_required.html')
    
    # Son 30 günün performansı
    performances = ModelPerformance.objects.order_by('-date')[:30]
    
    # Özet istatistikler
    if performances:
        avg_mape = sum(p.mape for p in performances if p.mape) / len(performances)
        avg_rmse = sum(p.rmse for p in performances if p.rmse) / len(performances)
    else:
        avg_mape = avg_rmse = None
    
    return render(request, 'predictions/performance.html', {
        'performances': performances,
        'avg_mape': avg_mape,
        'avg_rmse': avg_rmse,
    })


# API Views (JSON)

@login_required
def api_predictions(request):
    """
    API: Tahmin verileri
    """
    # Rate limiting
    if not request.user.can_make_api_call():
        return JsonResponse({
            'error': 'Günlük API limitinize ulaştınız',
            'limit': request.user.get_api_limit(),
            'upgrade_url': '/subscriptions/pricing/'
        }, status=429)
    
    request.user.increment_api_calls()
    
    service = PTFPredictionService()
    
    # Plan bazlı veri
    if request.user.is_premium():
        predictions = service.predict_next_72h()
    else:
        predictions = service.predict_next_72h()[:24]  # Free: sadece 24 saat
    
    return JsonResponse({
        'status': 'success',
        'count': len(predictions),
        'predictions': predictions,
        'api_calls_remaining': request.user.get_api_limit() - request.user.api_calls_today,
    })


@login_required
def api_daily_summary(request, date):
    """
    API: Günlük özet
    """
    if not request.user.can_make_api_call():
        return JsonResponse({
            'error': 'Günlük API limitinize ulaştınız'
        }, status=429)
    
    request.user.increment_api_calls()
    
    from datetime import datetime
    try:
        target_date = datetime.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        return JsonResponse({'error': 'Geçersiz tarih formatı'}, status=400)
    
    service = PTFPredictionService()
    summary = service.get_daily_summary(target_date)
    
    if not summary:
        return JsonResponse({'error': 'Bu tarih için tahmin bulunamadı'}, status=404)
    
    # Free kullanıcılar için detayları kısıtla
    if not request.user.is_premium():
        summary['predictions'] = summary['predictions'][:6]
    
    return JsonResponse({
        'status': 'success',
        **summary
    })
