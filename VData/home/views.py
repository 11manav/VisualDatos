from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.templatetags.static import static
import os
import pandas as pd
# Create your views here.



def home(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        data = pd.read_csv('./media/{}'.format(myfile.name))
        data_html = data.to_html()
        context = {'loaded_data': data_html}
        return render(request, './testTable.html',context)
    return render(request, './index.html')

