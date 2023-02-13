from django.urls import path,include
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.home, name='home'),
    path('preprocessing/',views.preprocessing,name='preprocessing'),
    path('preprocessing/nullvalues',views.dropingnull,name='dropnull_preprocessing'),
    path('preprocessing/minmax_scaler',views.minmaxScaler,name='minmaxScaler'),
    path('preprocessing/standard_scaler',views.standard_Scaler,name='standardScaler'),
    path('preprocessing/fillna_mean',views.fillingNullMean,name='fillingNullMean'),
    path('preprocessing/fillna_median',views.fillingNullMedian,name='fillingNullMedian'),
    path('preprocessing/fillna_mode',views.fillingNullMode,name='fillingNullMode'),
    path('preprocessing/fillna_modenumeric',views.fillingNullModeNumeric,name='fillingNullModeNumeric'),
    path('preprocessing/delete',views.deleteColumns,name='deleteColumn'),
    path('mlalgorithms/',views.mlalgorithms,name='mlalgorithms'),
    path('mlalgorithms/logistic_reg',views.logistic_reg,name='logistic_reg'),
    path('mlalgorithms/linear_reg',views.linear_reg,name='linear_reg')
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)