from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.templatetags.static import static
import os
import pandas as pd

# IMPORTANT!!! pip install scikit-learn
from sklearn.preprocessing import StandardScaler


code = [] 
data=None

def home(request):

    global data
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        data = pd.read_csv('./media/{}'.format(myfile.name))
        file_name=myfile.name
        # data = data.head(10)
        data_html = data.to_html()
        data_shape = data.shape
        context = {'loaded_data': data_html,
                    'shape_of_data': data_shape}
        return render(request, './main.html',context)
    return render(request, './index.html')




def preprocessing(request):
     global data
    #  data = data.head(10)
     data_html = data.to_html()
     data_shape = data.shape
     context = {'loaded_data': data_html,
                    'shape_of_data': data_shape}
     return render(request,'./preprocessing.html',context)


def dropingnull(request):
     global data
     data =data.dropna()
    #  data = data.head(10)
     data_html = data.to_html()
     data_shape = data.shape
     context = {'loaded_data': data_html,
                    'shape_of_data': data_shape}
     return render(request,'./preprocessing.html',context)

def minmaxScaler(request):
    global data
    scaler=StandardScaler()
    model=scaler.fit(data)
    data=model.transform(data)
    data_html = data.to_html()
    data_shape = data.shape
    context = {'loaded_data': data_html,
                    'shape_of_data': data_shape}
    return render(request,'./preprocessing.html',context)