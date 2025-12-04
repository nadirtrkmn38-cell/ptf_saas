"""
Dashboard App - Views
=====================
Ana sayfa, dashboard ve profil sayfaları.
"""

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from datetime import timedelta

from apps.predictions.services import PTFPredictionService
from apps.predictions.models import PTFPrediction, ModelPerformance
from apps.subscriptions.models import Plan


def home(request):
    """
    Ana sayfa.
    Giriş yapmamış kullanıcılar için landing page.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    # Güncel tahmin özeti (demo)
    service = PTFPredictionService()
    tomorrow_summary = service.get_daily_summary()
    
    plans = Plan.objects.filter(is_active=True).order_by('sort_order')
    
    return render(request, 'dashboard/home.html', {
        'tomorrow_summary': tomorrow_summary,
        'plans': plans,
    })


@login_required
def dashboard(request):
    """
    Kullanıcı dashboard'u.
    """
    service = PTFPredictionService()
    
    # Bugünkü ve yarınki tahminler
    today = timezone.now().date()
    tomorrow = today + timedelta(days=1)
    
    context = {
        'user': request.user,
        'today': today,
        'tomorrow': tomorrow,
    }
    
    # Premium kullanıcılar için detaylı veri
    if request.user.is_premium():
        predictions = service.predict_next_72h()
        context['predictions'] = predictions
        
        # 3 günlük özet
        context['daily_summaries'] = [
            service.get_daily_summary(today + timedelta(days=i))
            for i in range(1, 4)
        ]
        
        # Son performans
        context['recent_performance'] = ModelPerformance.objects.order_by('-date').first()
    else:
        # Free kullanıcılar için kısıtlı veri
        tomorrow_summary = service.get_daily_summary(tomorrow)
        context['tomorrow_summary'] = tomorrow_summary
        
        if tomorrow_summary:
            context['limited_predictions'] = tomorrow_summary['predictions'][:6]
    
    return render(request, 'dashboard/dashboard.html', context)


@login_required
def profile(request):
    """
    Kullanıcı profil sayfası.
    """
    if request.method == 'POST':
        user = request.user
        
        user.first_name = request.POST.get('first_name', '')
        user.last_name = request.POST.get('last_name', '')
        user.phone = request.POST.get('phone', '')
        user.company_name = request.POST.get('company_name', '')
        user.job_title = request.POST.get('job_title', '')
        
        # Pazarlama izni
        user.marketing_consent = request.POST.get('marketing_consent') == 'on'
        
        user.save()
        messages.success(request, 'Profiliniz güncellendi.')
        return redirect('profile')
    
    return render(request, 'dashboard/profile.html')


@login_required
def api_settings(request):
    """
    API ayarları sayfası.
    """
    if request.method == 'POST':
        if 'regenerate_key' in request.POST:
            new_key = request.user.generate_api_key()
            messages.success(request, f'Yeni API anahtarınız oluşturuldu.')
        
        return redirect('api_settings')
    
    return render(request, 'dashboard/api_settings.html')


def error_404(request, exception):
    """404 hata sayfası"""
    return render(request, 'errors/404.html', status=404)


def error_500(request):
    """500 hata sayfası"""
    return render(request, 'errors/500.html', status=500)
