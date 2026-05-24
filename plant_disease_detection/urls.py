from django.contrib import admin
from django.urls import path
from detection import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("home/", views.home_view, name="home"),  # Redirect to Home after login
    path("performance/", views.performance_view, name="performance"),
    path("charts/", views.charts_view, name="charts"),
     path('accounts/login/', views.login_view, name='login'),  # Custom login view
    path('accounts/logout/', views.logout_view, name='logout'),  # Custom logout view

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
