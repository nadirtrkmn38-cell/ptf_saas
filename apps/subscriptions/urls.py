from django.urls import path
from . import views

urlpatterns = [
    path('pricing/', views.pricing, name='pricing'),
    path('detail/', views.subscription_detail, name='subscription_detail'),
    path('checkout/<slug:plan_slug>/', views.checkout, name='checkout'),
    path('callback/', views.iyzico_callback, name='iyzico_callback'),
    path('webhook/', views.iyzico_webhook, name='iyzico_webhook'),
    path('cancel/', views.cancel_subscription, name='cancel_subscription'),
]