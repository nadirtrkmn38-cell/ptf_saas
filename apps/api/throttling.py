"""
API App - Throttling
"""

from rest_framework.throttling import UserRateThrottle


class SubscriptionBasedThrottle(UserRateThrottle):
    """
    Plan bazlý rate limiting.
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
    
    def allow_request(self, request, view):
        self.request = request
        return super().allow_request(request, view)