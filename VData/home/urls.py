from django.urls import path,include
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.home, name='home'),
    path('downloadDataset/', views.downloadDataset, name='downloadDataset'),
    path('showFulldataset/',views.showFulldataset,name='showFulldataset'),
    path('preprocessing/',views.preprocessing,name='preprocessing'),
    path('preprocessing/nullvalues',views.dropingnull,name='dropnull_preprocessing'),
    path('preprocessing/minmax_scaler',views.minmaxScaler,name='minmaxScaler'),
    path('preprocessing/standard_scaler',views.standard_Scaler,name='standardScaler'),
    path('preprocessing/fillna_mean',views.fillingNullMean,name='fillingNullMean'),
    path('preprocessing/fillna_median',views.fillingNullMedian,name='fillingNullMedian'),
    path('preprocessing/fillna_mode',views.fillingNullMode,name='fillingNullMode'),
    path('preprocessing/fillna_modenumeric',views.fillingNullModeNumeric,name='fillingNullModeNumeric'),
    path('preprocessing/delete',views.deleteColumns,name='deleteColumn'),
    path('preprocessing/categorical_data',views.cat_data,name='cat_data_form'),
    path('mlalgorithms/',views.mlalgorithms,name='mlalgorithms'),
    path('mlalgorithms/logistic_reg',views.logistic_reg,name='logistic_reg'),
    path('mlalgorithms/linear_reg',views.linear_reg,name='linear_reg'),
    path('mlalgorithms/knn',views.knn,name='knn'),
    path('mlalgorithms/kmeans',views.kmeans,name='kmeans'),
    path('visualisation/',views.visualisation,name='visualisation'),
    path('visualisation/pie_chart',views.pie_chart,name='pie_chart'),
    path('visualisation/histogram',views.histogram,name='histogram'),
    path('visualisation/box_plot',views.box_plot,name='box_plot'),
    path('visualisation/line_plot',views.line_plot,name='line_plot'),
    path('visualisation/elbow_plot',views.elbow_plot,name='elbow_plot')
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)