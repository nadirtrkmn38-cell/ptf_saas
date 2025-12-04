"""
Users App - Signals
"""

from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth import get_user_model
from .emails import send_welcome_email

User = get_user_model()

@receiver(post_save, sender=User)
def user_created(sender, instance, created, **kwargs):
    if created:
        send_welcome_email(instance)

@receiver(post_save, sender=CustomUser)
def create_api_key(sender, instance, created, **kwargs):
    """Yeni kullanıcı için API key oluştur"""
    if created and not instance.api_key:
        instance.generate_api_key()


@receiver(post_save, sender=CustomUser)
def log_subscription_change(sender, instance, **kwargs):
    """Abonelik değişikliğini logla"""
    if kwargs.get('update_fields') and 'subscription_plan' in kwargs['update_fields']:
        from .models import UserActivity
        UserActivity.objects.create(
            user=instance,
            action='subscription_change',
            details={'new_plan': instance.subscription_plan}
        )
