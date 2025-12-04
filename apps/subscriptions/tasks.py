"""
Subscriptions App - Celery Tasks
================================
Abonelik kontrolü ve bildirim görevleri.
"""

from celery import shared_task
from django.utils import timezone
from django.core.mail import send_mail
from django.conf import settings
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


@shared_task
def check_subscription_expirations():
    """
    Süresi dolan abonelikleri kontrol et ve durumlarını güncelle.
    Her gün 00:05'te çalışır.
    """
    from .models import Subscription
    from apps.users.models import CustomUser
    
    now = timezone.now()
    
    # Süresi dolan aktif abonelikler
    expired_subscriptions = Subscription.objects.filter(
        status='active',
        current_period_end__lt=now
    )
    
    expired_count = 0
    for subscription in expired_subscriptions:
        # Dönem sonunda iptal edilecekse
        if subscription.cancel_at_period_end:
            subscription.status = 'cancelled'
            subscription.save()
            
            # Kullanıcıyı free plana düşür
            subscription.user.subscription_plan = 'free'
            subscription.user.subscription_expires = None
            subscription.user.save()
            
            logger.info(f"Subscription cancelled: {subscription.user.email}")
        else:
            # Otomatik yenilenecek - iyzico tarafından handle edilir
            # Burada sadece unpaid olarak işaretle
            subscription.status = 'unpaid'
            subscription.save()
            
            logger.warning(f"Subscription unpaid: {subscription.user.email}")
        
        expired_count += 1
    
    logger.info(f"Checked subscriptions: {expired_count} expired")
    return {'expired_count': expired_count}


@shared_task
def send_expiration_reminders():
    """
    Aboneliği bitmek üzere olan kullanıcılara hatırlatma emaili gönder.
    Her gün 10:00'da çalışır.
    """
    from .models import Subscription
    
    now = timezone.now()
    
    # 3 gün içinde bitecek abonelikler
    reminder_date = now + timedelta(days=3)
    
    expiring_soon = Subscription.objects.filter(
        status='active',
        cancel_at_period_end=False,  # İptal edilmemiş
        current_period_end__lte=reminder_date,
        current_period_end__gt=now
    )
    
    sent_count = 0
    for subscription in expiring_soon:
        try:
            send_mail(
                subject='PTF Tahmin - Aboneliğiniz yenileniyor',
                message=f"""
Merhaba {subscription.user.first_name or subscription.user.email},

{subscription.plan.name} aboneliğiniz {subscription.current_period_end.strftime('%d.%m.%Y')} tarihinde otomatik olarak yenilenecektir.

Plan: {subscription.plan.name}
Tutar: {subscription.plan.price} TL

Aboneliğinizi iptal etmek isterseniz hesabınızdan işlem yapabilirsiniz.

İyi günler,
PTF Tahmin Ekibi
                """,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[subscription.user.email],
                fail_silently=True,
            )
            sent_count += 1
        except Exception as e:
            logger.error(f"Email send error: {e}")
    
    logger.info(f"Sent {sent_count} reminder emails")
    return {'sent_count': sent_count}


@shared_task
def send_payment_failed_notification(subscription_id):
    """
    Ödeme başarısız olduğunda kullanıcıya bildirim gönder.
    """
    from .models import Subscription
    
    try:
        subscription = Subscription.objects.get(id=subscription_id)
        
        send_mail(
            subject='PTF Tahmin - Ödeme başarısız',
            message=f"""
Merhaba {subscription.user.first_name or subscription.user.email},

{subscription.plan.name} aboneliğiniz için ödeme alınamadı.

Lütfen ödeme bilgilerinizi güncelleyin veya farklı bir kart deneyin.

Aboneliğiniz: {subscription.plan.name}
Tutar: {subscription.plan.price} TL

Ödeme bilgilerinizi güncellemek için:
{settings.SITE_URL}/subscriptions/detail/

7 gün içinde ödeme alınamazsa aboneliğiniz iptal edilecektir.

İyi günler,
PTF Tahmin Ekibi
            """,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[subscription.user.email],
            fail_silently=False,
        )
        
        logger.info(f"Payment failed notification sent: {subscription.user.email}")
        
    except Subscription.DoesNotExist:
        logger.error(f"Subscription not found: {subscription_id}")
    except Exception as e:
        logger.error(f"Notification error: {e}")


@shared_task
def cleanup_pending_subscriptions():
    """
    24 saatten uzun süredir pending olan abonelikleri temizle.
    """
    from .models import Subscription
    
    cutoff = timezone.now() - timedelta(hours=24)
    
    old_pending = Subscription.objects.filter(
        status='pending',
        created_at__lt=cutoff
    )
    
    deleted_count = old_pending.count()
    old_pending.delete()
    
    logger.info(f"Cleaned up {deleted_count} pending subscriptions")
    return {'deleted_count': deleted_count}
