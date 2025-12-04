"""
Subscriptions App - Admin Configuration
"""

from django.contrib import admin
from .models import Plan, Subscription, Payment, SubscriptionEvent


@admin.register(Plan)
class PlanAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug', 'price', 'interval', 'api_calls_per_day', 'is_active', 'is_featured']
    list_filter = ['is_active', 'is_featured', 'interval']
    search_fields = ['name', 'slug']
    prepopulated_fields = {'slug': ('name',)}
    ordering = ['sort_order', 'price']


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ['user', 'plan', 'status', 'current_period_end', 'created_at']
    list_filter = ['status', 'plan', 'created_at']
    search_fields = ['user__email', 'iyzico_subscription_ref']
    readonly_fields = ['id', 'created_at', 'updated_at']
    raw_id_fields = ['user']
    date_hierarchy = 'created_at'


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ['subscription', 'amount', 'status', 'payment_date', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['subscription__user__email', 'iyzico_payment_id']
    readonly_fields = ['id', 'created_at']
    date_hierarchy = 'created_at'


@admin.register(SubscriptionEvent)
class SubscriptionEventAdmin(admin.ModelAdmin):
    list_display = ['subscription', 'event_type', 'created_at']
    list_filter = ['event_type', 'created_at']
    search_fields = ['subscription__user__email']
    readonly_fields = ['subscription', 'event_type', 'data', 'created_at']
    date_hierarchy = 'created_at'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False
