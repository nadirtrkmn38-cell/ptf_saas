"""
Email utility fonksiyonları
"""

from django.core.mail import send_mail, EmailMultiAlternatives
from django.template.loader import render_to_string
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


def send_welcome_email(user):
    """Hoşgeldin emaili"""
    subject = 'PTF Tahmin\'e Hoşgeldiniz!'
    
    html_content = """
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center;">
            <h1 style="color: white; margin: 0;">PTF Tahmin</h1>
        </div>
        <div style="padding: 30px; background: #f8f9fa;">
            <h2>Merhaba {first_name}!</h2>
            <p>PTF Tahmin platformuna kayıt olduğunuz için teşekkür ederiz.</p>
            <p>Artık günlük elektrik fiyat tahminlerine erişebilirsiniz.</p>
            <div style="text-align: center; margin: 30px 0;">
                <a href="{dashboard_url}" style="background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px;">Dashboard'a Git</a>
            </div>
            <p>Sorularınız için bize ulaşabilirsiniz.</p>
            <p>Saygılarımızla,<br>PTF Tahmin Ekibi</p>
        </div>
    </body>
    </html>
    """.format(
        first_name=user.first_name or 'Kullanıcı',
        dashboard_url=settings.SITE_URL + '/dashboard/'
    )
    
    try:
        msg = EmailMultiAlternatives(
            subject=subject,
            body='PTF Tahmin\'e hoşgeldiniz!',
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[user.email]
        )
        msg.attach_alternative(html_content, "text/html")
        msg.send()
        logger.info("Welcome email sent to: " + str(user.email))
    except Exception as e:
        logger.error("Email error: " + str(e))


def send_subscription_confirmation(user, plan):
    """Abonelik onay emaili"""
    subject = 'Aboneliğiniz Aktifleştirildi!'
    
    html_content = """
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center;">
            <h1 style="color: white; margin: 0;">PTF Tahmin</h1>
        </div>
        <div style="padding: 30px; background: #f8f9fa;">
            <h2>Abonelik Onayı</h2>
            <p>Merhaba {first_name},</p>
            <p><strong>{plan_name}</strong> planınız başarıyla aktifleştirildi!</p>
            <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <p><strong>Plan:</strong> {plan_name}</p>
                <p><strong>Tutar:</strong> {plan_price} TL/ay</p>
                <p><strong>Özellikler:</strong></p>
                <ul>
                    <li>72 saatlik tahmin</li>
                    <li>API erişimi</li>
                    <li>Öncelikli destek</li>
                </ul>
            </div>
            <div style="text-align: center;">
                <a href="{dashboard_url}" style="background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px;">Dashboard'a Git</a>
            </div>
        </div>
    </body>
    </html>
    """.format(
        first_name=user.first_name or 'Kullanıcı',
        plan_name=plan.name,
        plan_price=plan.price,
        dashboard_url=settings.SITE_URL + '/dashboard/'
    )
    
    try:
        msg = EmailMultiAlternatives(
            subject=subject,
            body='Aboneliğiniz aktifleştirildi!',
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[user.email]
        )
        msg.attach_alternative(html_content, "text/html")
        msg.send()
    except Exception as e:
        logger.error("Email error: " + str(e))


def send_expiration_reminder(user, days_left):
    """Abonelik bitiş hatırlatması"""
    subject = 'Aboneliğiniz ' + str(days_left) + ' Gün İçinde Sona Erecek'
    
    html_content = """
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: #f59e0b; padding: 30px; text-align: center;">
            <h1 style="color: white; margin: 0;">⚠️ Hatırlatma</h1>
        </div>
        <div style="padding: 30px; background: #f8f9fa;">
            <p>Merhaba {first_name},</p>
            <p>Aboneliğinizin sona ermesine <strong>{days_left} gün</strong> kaldı.</p>
            <p>Kesintisiz hizmet almaya devam etmek için aboneliğinizi yenileyin.</p>
            <div style="text-align: center; margin: 30px 0;">
                <a href="{pricing_url}" style="background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px;">Aboneliği Yenile</a>
            </div>
        </div>
    </body>
    </html>
    """.format(
        first_name=user.first_name or 'Kullanıcı',
        days_left=days_left,
        pricing_url=settings.SITE_URL + '/subscriptions/pricing/'
    )
    
    try:
        msg = EmailMultiAlternatives(
            subject=subject,
            body='Aboneliğiniz yakında sona erecek.',
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[user.email]
        )
        msg.attach_alternative(html_content, "text/html")
        msg.send()
    except Exception as e:
        logger.error("Email error: " + str(e))