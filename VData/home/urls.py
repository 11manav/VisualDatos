from django.urls import path,include
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.home, name='home'),
    path('preprocessing/',views.preprocessing,name='preprocessing'),
    path('preprocessing/nullvalues',views.dropingnull,name='dropnull_preprocessing'),
    path('preprocessing/minmax',views.minmaxScaler,name='minmaxScaler'),
   
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)