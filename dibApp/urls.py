from django.urls import path
from .import views

urlpatterns=[
    path('',views.homepage,name='home'),
    path('heart/',views.heart_disease,name='heart'),
    path('diabetes/',views.diabetes_disease,name='diabetes'),
    path('parkinson/',views.Parkinson_disease,name='parkinson')
]