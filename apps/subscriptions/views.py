"""
Subscriptions App - Views
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils import timezone
from datetime import timedelta
import logging

from .models import Plan, Subscription, Payment, SubscriptionEvent
from .services import IyzicoService

logger = logging.getLogger(__name__)


def pricing(request):
    """Fiyatlandırma sayfası"""
    plans = Plan.objects.filter(is_active=True).order_by('sort_order')
    
    current_plan = None
    if request.user.is_authenticated:
        current_plan = request.user.subscription_plan
    
    return render(request, 'subscriptions/pricing.html', {
        'plans': plans,
        'current_plan': current_plan,
    })


@login_required
def subscription_detail(request):
    """Kullanıcının abonelik detayları"""
    subscription = Subscription.objects.filter(
        user=request.user,
        status='active'
    ).first()
    
    payments = Payment.objects.filter(
        subscription__user=request.user
    ).order_by('-created_at')[:10]
    
    return render(request, 'subscriptions/detail.html', {
        'subscription': subscription,
        'payments': payments,
    })


@login_required
def checkout(request, plan_slug):
    """Ödeme sayfası"""
    plan = get_object_or_404(Plan, slug=plan_slug, is_active=True)
    
    # Zaten bu plana sahipse
    if request.user.subscription_plan == plan.slug:
        messages.warning(request, 'Zaten bu plana sahipsiniz.')
        return redirect('subscription_detail')
    
    # Free plan ise direkt aktifle
    plan_price = float(plan.price)
    if plan_price == 0:
        request.user.subscription_plan = str(plan.slug)
        request.user.save()
        messages.success(request, str(plan.name) + ' plani aktiflestirildi!')
        return redirect('dashboard')
    
    if request.method == 'POST':
        try:
            service = IyzicoService()
            callback_url = request.build_absolute_uri('/subscriptions/callback/')
            
            result = service.create_checkout_form(
                plan=plan,
                user=request.user,
                callback_url=str(callback_url)
            )
            
            result_status = str(result.get('status', ''))
            
            if result_status == 'success':
                return render(request, 'subscriptions/checkout_form.html', {
                    'checkout_form_content': result.get('checkoutFormContent'),
                    'plan': plan,
                    'token': str(result.get('token', '')),
                })
            else:
                error_msg = str(result.get('errorMessage', 'Bilinmeyen hata'))
                messages.error(request, 'Odeme baslatilamadi: ' + error_msg)
                
        except Exception as e:
            logger.error('Checkout error: ' + str(e))
            messages.error(request, 'Odeme islemi baslatilirken bir hata olustu.')
    
    return render(request, 'subscriptions/checkout.html', {
        'plan': plan,
    })


@csrf_exempt
def iyzico_callback(request):
    """iyzico ödeme callback"""
    token = request.POST.get('token', '') or request.GET.get('token', '')
    plan_id = request.POST.get('plan_id', '')
    
    if not token:
        messages.error(request, 'Gecersiz odeme islemi.')
        return redirect('pricing')
    
    try:
        service = IyzicoService()
        result = service.retrieve_checkout_result(str(token))
        
        payment_status = str(result.get('paymentStatus', ''))
        status = str(result.get('status', ''))
        
        if status == 'success' and payment_status == 'SUCCESS':
            # Plan bul
            plan = None
            if plan_id:
                try:
                    plan = Plan.objects.get(id=int(plan_id))
                except Exception:
                    pass
            
            if not plan:
                plan = Plan.objects.filter(slug='pro').first()
            
            if request.user.is_authenticated and plan:
                user = request.user
                
                # Kullanıcı planını güncelle
                user.subscription_plan = str(plan.slug)
                user.subscription_expires = timezone.now() + timedelta(days=30)
                user.save()
                
                # Subscription kaydı oluştur
                subscription = Subscription.objects.create(
                    user=user,
                    plan=plan,
                    status='active',
                    current_period_start=timezone.now(),
                    current_period_end=timezone.now() + timedelta(days=30),
                )
                
                # Payment kaydı
                Payment.objects.create(
                    subscription=subscription,
                    amount=float(plan.price),
                    status='completed',
                    iyzico_payment_id=str(result.get('paymentId', '')),
                    payment_date=timezone.now(),
                )
                
                plan_name = str(plan.name)
                messages.success(request, 'Odeme basarili! ' + plan_name + ' plani aktiflestirildi.')
                return redirect('dashboard')
        
        error_msg = str(result.get('errorMessage', 'Odeme islemi basarisiz oldu.'))
        messages.error(request, error_msg)
        return redirect('pricing')
        
    except Exception as e:
        logger.error('Callback error: ' + str(e))
        messages.error(request, 'Odeme islenirken bir hata olustu.')
        return redirect('pricing')


@csrf_exempt
def iyzico_webhook(request):
    """iyzico webhook handler"""
    return HttpResponse(status=200)


@login_required
@require_POST
def cancel_subscription(request):
    """Abonelik iptali"""
    subscription = get_object_or_404(
        Subscription,
        user=request.user,
        status='active'
    )
    
    reason = request.POST.get('reason', '')
    immediate = request.POST.get('immediate', 'false') == 'true'
    
    try:
        if subscription.iyzico_subscription_ref:
            service = IyzicoService()
            service.cancel_subscription(str(subscription.iyzico_subscription_ref))
        
        subscription.cancel(reason=reason, immediate=immediate)
        
        if immediate:
            request.user.subscription_plan = 'free'
            request.user.subscription_expires = None
            request.user.save()
            messages.success(request, 'Aboneliginiz iptal edildi.')
        else:
            if subscription.current_period_end:
                end_date = subscription.current_period_end.strftime("%d.%m.%Y")
                messages.success(request, 'Aboneliginiz ' + end_date + ' tarihinde sona erecek.')
            else:
                messages.success(request, 'Aboneliginiz iptal edildi.')
        
    except Exception as e:
        logger.error('Cancel subscription error: ' + str(e))
        messages.error(request, 'Abonelik iptal edilirken bir hata olustu.')
    
    return redirect('subscription_detail')