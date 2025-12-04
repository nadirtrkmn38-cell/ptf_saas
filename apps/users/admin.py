"""
Users App - Admin Configuration
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, UserActivity


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    list_display = ['email', 'first_name', 'last_name', 'subscription_plan', 'subscription_expires', 'is_active']
    list_filter = ['subscription_plan', 'is_active', 'is_staff']
    search_fields = ['email', 'first_name', 'last_name']
    ordering = ['-date_joined']
    
    fieldsets = UserAdmin.fieldsets + (
        ('Abonelik Bilgileri', {
            'fields': ('subscription_plan', 'subscription_expires', 'api_key', 'phone', 'company')
        }),
    )
    
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Ek Bilgiler', {
            'fields': ('first_name', 'last_name', 'subscription_plan')
        }),
    )
    
    readonly_fields = ['created_at', 'updated_at', 'api_calls_today', 'api_calls_reset_date']

    admin.site.site_header = 'PTF Tahmin Yönetim Paneli'
    admin.site.site_title = 'PTF Tahmin Admin'
    admin.site.index_title = 'Hoşgeldiniz'


@admin.register(UserActivity)
class UserActivityAdmin(admin.ModelAdmin):
    list_display = ['user', 'action', 'ip_address', 'created_at']
    list_filter = ['action', 'created_at']
    search_fields = ['user__email', 'ip_address']
    readonly_fields = ['user', 'action', 'details', 'ip_address', 'user_agent', 'created_at']
    ordering = ['-created_at']
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


