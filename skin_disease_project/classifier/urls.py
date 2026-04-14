from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing_page'),

    # Admin
    path('admin_login/', views.admin_login, name='admin_login'),
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('upload_dataset/', views.upload_dataset, name='upload_dataset'),
    path('preprocess/', views.preprocess_dataset, name='preprocess_dataset'),
    path('train_mobilenet/', views.train_mobilenet, name='train_mobilenet'),
    path('train_inception/', views.train_inception, name='train_inception'),
    path('train_vgg/', views.train_vgg, name='train_vgg'),
    path('train_resnet/', views.train_resnet, name='train_resnet'),
    path('comparison/', views.comparison, name='comparison'),
    path('users_details/', views.users_details, name='users_details'),
    path('admin_logout/', views.admin_logout, name='admin_logout'),

    # User
    path('register/', views.register_user, name='register_user'),
    path('login/', views.login_user, name='login_user'),
    path('user_dashboard/', views.user_dashboard, name='user_dashboard'),
    path('predict/', views.predict_disease, name='predict'),
    path('user_logout/', views.user_logout, name='user_logout'),
]
