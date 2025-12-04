"""
Users App - Admin Configuration
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, UserActivity


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    model = CustomUser
    
    list_display = ['email', 'first_name', 'last_name', 'subscription_plan', 
                    'subscription_expires', 'is_active', 'created_at']
    list_filter = ['subscription_plan', 'is_active', 'is_staff', 'kvkk_consent']
    search_fields = ['email', 'first_name', 'last_name', 'company_name']
    ordering = ['-created_at']
    
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Kişisel Bilgiler', {'fields': ('first_name', 'last_name', 'phone', 'company_name', 'job_title')}),
        ('Abonelik', {'fields': ('subscription_plan', 'subscription_expires')}),
        ('API', {'fields': ('api_key', 'api_calls_today', 'api_calls_reset_date')}),
        ('İzinler', {'fields': ('kvkk_consent', 'kvkk_consent_date', 'marketing_consent')}),
        ('Yetkiler', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Tarihler', {'fields': ('last_login', 'date_joined')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'first_name', 'last_name'),
        }),
    )
    
    readonly_fields = ['created_at', 'updated_at', 'api_calls_today', 'api_calls_reset_date']


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
