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
    global code
    code = ["import pandas as pd"]
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        code.append("df = pd.read_csv('{}')".format(filename))
        data = pd.read_csv('./media/{}'.format(myfile.name))
        os.remove(os.path.join(settings.MEDIA_ROOT, myfile.name))

        file_name=myfile.name
        # data = data.head(10)
        data_html = data.to_html()
        data_shape, nullValues = getStatistics(data)
        context = {'loaded_data': data_html,
                    'shape_of_data': data_shape,
                    'null_count': nullValues,
                    'backgroundCode': code}
        
        # print(code)
        return render(request, './main.html',context)
    return render(request, './index.html')




def preprocessing(request):
     global data
    #  data = data.head(10)
     data_html = data.to_html()
     data_shape, nullValues = getStatistics(data)
     columns=list(data.columns)
     context = {'loaded_data': data_html,
                    'shape_of_data': data_shape,
                    'null_count': nullValues,'columns':columns}
     return render(request,'./preprocessing.html',context)


def dropingnull(request):
    global data
    data =data.dropna()
#  data = data.head(10)
    data_html = data.to_html()
    data_shape, nullValues = getStatistics(data)
    context = {'loaded_data': data_html,
                'shape_of_data': data_shape,
                'null_count': nullValues}
    return render(request,'./preprocessing.html',context)

def minmaxScaler(request):
    global data
    scaler=StandardScaler()
    model=scaler.fit(data)
    data=model.transform(data)
    data_html = data.to_html()
    data_shape, nullValues = getStatistics(data)
    context = {'loaded_data': data_html,
                    'shape_of_data': data_shape,
                    'null_count': nullValues}
    return render(request,'./preprocessing.html',context)


def getStatistics(data):
    return data.shape,data.isna().sum().sum()




def fillingNullMean(request):
    global data
    mean_of_columns = data.mean()

    # replace the data with the mean calculated
    for col in data.columns:
        try:
            data[col].fillna(mean_of_columns[col], inplace=True)
        except:
            print(col)
            continue
    print("MEAN")
    print(data)
    data_html = data.to_html()
    data_shape, nullValues = getStatistics(data)
    context = {'loaded_data': data_html,
                'shape_of_data': data_shape,
                'null_count': nullValues}
    return render(request,'./preprocessing.html',context)





def fillingNullMedian(request):
    global data
    median_of_columns = data.median()

    # replace the data with the mean calculated
    for col in data.columns:
        try:
            data[col].fillna(median_of_columns[col], inplace=True)
        except:
            print(col)
            continue
    print(data)
    data_html = data.to_html()
    data_shape, nullValues = getStatistics(data)
    context = {'loaded_data': data_html,
                'shape_of_data': data_shape,
                'null_count': nullValues}
    return render(request,'./preprocessing.html',context)




def fillingNullMode(request):
    global data
    for col in data.columns:
        try:
            data[col].fillna(data.mode()[col][0], inplace=True)
        except:
            print(col)
            continue
    print(data)
    data_html = data.to_html()
    data_shape, nullValues = getStatistics(data)
    context = {'loaded_data': data_html,
                'shape_of_data': data_shape,
                'null_count': nullValues}
    return render(request,'./preprocessing.html',context)



def deleteColumns(request):
     global data
    #  data = data.head(10)
     name = request.GET.get('name')
     print(name)
     if name is not None:
        data=data.drop([name], axis=1)
        data_html = data.to_html()
        columns=list(data.columns)
        data_shape, nullValues = getStatistics(data)
        context = {'loaded_data': data_html,
                    'shape_of_data': data_shape,
                    'null_count': nullValues,'columns':columns}
        return render(request,'./preprocessing.html',context)