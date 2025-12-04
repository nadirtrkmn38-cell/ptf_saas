"""
Subscriptions App - iyzico Service
Demo mod
"""

from django.conf import settings
import uuid
import logging

logger = logging.getLogger(__name__)


class IyzicoService:
    
    def __init__(self):
        pass
    
    def create_checkout_form(self, plan, user, callback_url):
        try:
            token = "demo_" + uuid.uuid4().hex[:8]
            
            plan_id_str = str(plan.id)
            plan_name_str = str(plan.name)
            plan_price_str = str(plan.price)
            callback_str = str(callback_url)
            
            html_parts = []
            html_parts.append('<div style="text-align:center;padding:40px;background:#f8f9fa;border-radius:12px;margin:20px 0;">')
            html_parts.append('<div style="background:white;padding:30px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.1);">')
            html_parts.append('<h3 style="color:#333;margin-bottom:20px;">Demo Odeme</h3>')
            html_parts.append('<p style="color:#666;margin-bottom:10px;"><strong>Plan:</strong> ')
            html_parts.append(plan_name_str)
            html_parts.append('</p>')
            html_parts.append('<p style="color:#666;margin-bottom:20px;"><strong>Tutar:</strong> ')
            html_parts.append(plan_price_str)
            html_parts.append(' TL</p>')
            html_parts.append('<hr style="margin:20px 0;border:none;border-top:1px solid #eee;">')
            html_parts.append('<p style="color:#888;font-size:14px;margin-bottom:20px;">Demo moddur.</p>')
            html_parts.append('<form action="')
            html_parts.append(callback_str)
            html_parts.append('" method="POST">')
            html_parts.append('<input type="hidden" name="token" value="')
            html_parts.append(token)
            html_parts.append('">')
            html_parts.append('<input type="hidden" name="plan_id" value="')
            html_parts.append(plan_id_str)
            html_parts.append('">')
            html_parts.append('<button type="submit" style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:15px 50px;border:none;border-radius:8px;cursor:pointer;font-size:16px;font-weight:bold;">Odemeyi Tamamla</button>')
            html_parts.append('</form>')
            html_parts.append('</div>')
            html_parts.append('</div>')
            
            form_content = "".join(html_parts)
            
            return {
                "status": "success",
                "checkoutFormContent": form_content,
                "token": token
            }
            
        except Exception as e:
            logger.error("Checkout error")
            logger.error(str(e))
            return {
                "status": "failure",
                "errorMessage": str(e)
            }
    
    def retrieve_checkout_result(self, token):
        token_str = str(token)
        if token_str.startswith("demo_"):
            return {
                "status": "success",
                "paymentStatus": "SUCCESS",
                "basketId": "demo",
                "paidPrice": "799",
                "paymentId": "demo_" + uuid.uuid4().hex[:8]
            }
        
        return {
            "status": "failure",
            "errorMessage": "Gecersiz token"
        }
    
    def cancel_subscription(self, subscription_ref):
        return {"status": "success"}