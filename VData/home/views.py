from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.templatetags.static import static
import os
import pandas as pd
import uuid

# IMPORTANT!!! pip install scikit-learn
from sklearn.preprocessing import  MinMaxScaler,StandardScaler

# Global Variables
code = [] 
# data=None


# --------Common data required for all pages--------------

def getContext(data_html,data_shape,nullValues,code,columns):
    context = {'loaded_data': data_html,
                    'shape_of_data': data_shape,
                    'null_count': nullValues,
                    'backgroundCode': code,
                    'columns':columns}
    return context

def getStatistics(data):
    return data.shape,data.isna().sum().sum(),list(data.columns)

# --------------------------------------------------------


def home(request):

    request.session['session_key'] = str(uuid.uuid4())
    session_key = request.session.get('session_key', None)
    code = ["import pandas as pd"]
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        base, ext = os.path.splitext(myfile.name)
        newFileName = base + session_key + ext
        filename = fs.save(newFileName, myfile)

        request.session['filename'] = filename

        code.append("df = pd.read_csv('{}')".format(myfile.name))
        data = pd.read_csv('./media/{}'.format(newFileName))
        # os.remove(os.path.join(settings.MEDIA_ROOT, myfile.name))

        
        # data = data.head(10)
        data_html = data.to_html()
        data_shape, nullValues, columns = getStatistics(data)
        
        context = getContext(data_html,data_shape,nullValues,code,columns)
        
        # print(code)
        return render(request, './main.html',context)
    return render(request, './index.html')




def preprocessing(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    data_html = data.to_html()
    data_shape, nullValues, columns = getStatistics(data)

    context = getContext(data_html,data_shape,nullValues,code,columns)

    return render(request,'./preprocessing.html',context)


def dropingnull(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    data = data.dropna()
    data.to_csv('./media/{}'.format(filename),index=False)
    data_html = data.to_html()
    data_shape, nullValues, columns = getStatistics(data)

    context = getContext(data_html,data_shape,nullValues,code,columns)

    return render(request,'./preprocessing.html',context)

def minmaxScaler(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    columns=[]
    for col in data.columns:
        datatypes = data.dtypes[col]
        if datatypes=='float64' or datatypes=='int64':
            columns.append(col)
    # scaler=StandardScaler()
    # model=scaler.fit(data)
    # data=model.transform(data)
    min_max_scaler =MinMaxScaler(feature_range =(0, 1))
    data[columns]= min_max_scaler.fit_transform(data[columns])
    data.to_csv('./media/{}'.format(filename),index=False)
    data_html = data.to_html()
    data_shape, nullValues, columns = getStatistics(data)
    
    context = getContext(data_html,data_shape,nullValues,code,columns)
    return render(request,'./preprocessing.html',context)


def standard_Scaler(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    columns=[]
    for col in data.columns:
        datatypes = data.dtypes[col]
        if datatypes=='float64' or datatypes=='int64':
            columns.append(col)
    scaler=StandardScaler()
    model=scaler.fit(data[columns])
    data[columns]=model.transform(data[columns])
    data.to_csv('./media/{}'.format(filename),index=False)
    data_html = data.to_html()
    data_shape, nullValues, columns = getStatistics(data)
    
    context = getContext(data_html,data_shape,nullValues,code,columns)
    return render(request,'./preprocessing.html',context)


def fillingNullMean(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    mean_of_columns = data.mean()

    # replace the data with the mean calculated
    for col in data.columns:
        try:
            data[col].fillna(mean_of_columns[col], inplace=True)
        except:
            print(col)
            continue
    print("MEAN")
    data.to_csv('./media/{}'.format(filename),index=False)
    data_html = data.to_html()
    data_shape, nullValues, columns = getStatistics(data)
    context = getContext(data_html,data_shape,nullValues,code,columns)
    return render(request,'./preprocessing.html',context)





def fillingNullMedian(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    median_of_columns = data.median()

    # replace the data with the mean calculated
    for col in data.columns:
        try:
            data[col].fillna(median_of_columns[col], inplace=True)
        except:
            print(col)
            continue
    data.to_csv('./media/{}'.format(filename),index=False)
    data_html = data.to_html()
    data_shape, nullValues, columns = getStatistics(data)
    context = getContext(data_html,data_shape,nullValues,code,columns)
    return render(request,'./preprocessing.html',context)




def fillingNullMode(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    for col in data.columns:
        try:
            data[col].fillna(data.mode()[col][0], inplace=True)
        except:
            print(col)
            continue
    data.to_csv('./media/{}'.format(filename),index=False)
    data_html = data.to_html()
    data_shape, nullValues, columns = getStatistics(data)
    context = getContext(data_html,data_shape,nullValues,code,columns)
    return render(request,'./preprocessing.html',context)

def fillingNullModeNumeric(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    for col in data.columns:
        if data[col].dtypes == object:
            pass
        else:
            try:
                data[col].fillna(data.mode()[col][0], inplace=True)
            except:
                print(col)
                continue
    data.to_csv('./media/{}'.format(filename),index=False)
    data_html = data.to_html()
    data_shape, nullValues, columns = getStatistics(data)
    context = getContext(data_html,data_shape,nullValues,code,columns)
    return render(request,'./preprocessing.html',context)




def deleteColumns(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    name = request.GET.get('name')
    print(name)
    if name is not None:
        data=data.drop([name], axis=1)
        data.to_csv('./media/{}'.format(filename),index=False)
        data_html = data.to_html()
        data_shape, nullValues, columns = getStatistics(data)
        context = getContext(data_html,data_shape,nullValues,code,columns)
        return render(request,'./preprocessing.html',context)