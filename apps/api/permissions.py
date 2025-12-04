"""
API App - Permissions & Throttling
==================================
Custom permissions ve rate limiting.
"""

from rest_framework.permissions import BasePermission
from rest_framework.throttling import UserRateThrottle


class IsPremiumUser(BasePermission):
    """
    Premium üyelik gerektiren endpoint'ler için.
    """
    message = "Bu endpoint premium üyelik gerektirir."
    
    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False
        return request.user.is_premium()


class HasAPIAccess(BasePermission):
    """
    API erişimi olan kullanıcılar için.
    Tüm planlar API'ye erişebilir, sadece limitler farklı.
    """
    message = "API erişimi için hesabınızın aktif olması gerekir."
    
    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False
        return request.user.has_active_subscription()


class IsEnterprise(BasePermission):
    """
    Enterprise plan gerektiren endpoint'ler için.
    """
    message = "Bu endpoint Enterprise plan gerektirir."
    
    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False
        return request.user.subscription_plan == 'enterprise'


class SubscriptionBasedThrottle(UserRateThrottle):
    """
    Plan bazlı rate limiting.
    
    Free: 100/gün
    Basic: 500/gün
    Pro: 5000/gün
    Enterprise: 50000/gün
    """
    
    def get_cache_key(self, request, view):
        if request.user.is_authenticated:
            ident = request.user.pk
        else:
            ident = self.get_ident(request)
        
        return self.cache_format % {
            'scope': self.scope,
            'ident': ident
        }
    
    def get_rate(self):
        """Plan bazlı rate döndür"""
        if hasattr(self, 'request') and self.request.user.is_authenticated:
            plan = self.request.user.subscription_plan
            rates = {
                'free': '100/day',
                'basic': '500/day',
                'pro': '5000/day',
                'enterprise': '50000/day',
            }
            return rates.get(plan, '100/day')
        return '100/day'
    
    def allow_request(self, request, view):
        # Request'i sakla (get_rate'de kullanmak için)
        self.request = request
        return super().allow_request(request, view)


class BurstThrottle(UserRateThrottle):
    """
    Anlık istek limiti.
    Saniyede max 10 istek.
    """
    scope = 'burst'
    rate = '10/second'
