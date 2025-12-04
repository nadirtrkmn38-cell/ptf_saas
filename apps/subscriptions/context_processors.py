"""
Subscriptions App - Context Processors
======================================
Template'lerde abonelik bilgilerini global olarak erişilebilir yapar.
"""


def subscription_context(request):
    """
    Abonelik bilgilerini tüm template'lere ekle.
    """
    context = {
        'is_premium': False,
        'subscription_plan': 'free',
        'subscription_expires': None,
        'api_calls_remaining': 0,
    }
    
    if request.user.is_authenticated:
        context.update({
            'is_premium': request.user.is_premium(),
            'subscription_plan': request.user.subscription_plan,
            'subscription_expires': request.user.subscription_expires,
            'api_calls_remaining': request.user.get_api_limit() - request.user.api_calls_today,
        })
    
    return context
