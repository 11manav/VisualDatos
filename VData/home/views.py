from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import pandas as pd
import numpy as np
import time
from django.http import HttpResponse


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
# IMPORTANT!!! pip install scikit-learn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
import seaborn as sns

# Global Variables
code = []
# data=None


# --------Common data required for all pages--------------

def getContext(data_html, data_shape, nullValues, code, columns):
    if (len(columns) == 0):
        data_html = ""

    context = {'loaded_data': data_html,
               'shape_of_data': data_shape,
               'null_count': nullValues,
               'backgroundCode': code,
               'columns': columns}
    return context


def getStatistics(data):
    return data.shape, data.isna().sum().sum(), list(data.columns)

# --------------------------------------------------------

# ----------- Session File Handling --------------------


def delete_old_datasets():
    media_storage = FileSystemStorage(location='media')

    files = media_storage.listdir('')[1]

    for file in files:
        filename = os.path.join(settings.MEDIA_ROOT, file)
        age_of_file = time.time() - os.path.getmtime(filename)

        if (age_of_file > 120):
            media_storage.delete(file)
# --------------------------------------------------------

# ------------- Download Dataset --------------------


def downloadDataset(request):
    fileName = request.session.get('filename', None)
    file_path = './media/{}'.format(fileName)
    session_key = request.session.get('session_key', None)

    # Open the file for reading
    with open(file_path, 'rb') as f:
        # Create the HttpResponse object with the file as content
        response = HttpResponse(f.read())
        # Set the content type header
        content_type = 'application/octet-stream'
        response['Content-Type'] = content_type
        # Set the Content-Disposition header to force file download
        filename = os.path.basename(file_path)
        filename = filename.replace(session_key, '')
        print(filename)
        response['Content-Disposition'] = 'attachment; filename="%s"' % filename
        return response


# ---------------------------------------------------


def home(request):

    delete_old_datasets()

    request.session['session_key'] = request.session._get_session_key()
    session_key = request.session.get('session_key', None)
    # print(dir(request.session))
    # print(request.session._get_session_key())
    # code.append(["import pandas as pd"])
    if request.method == 'POST' and request.FILES['myfile']:

        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        base, ext = os.path.splitext(myfile.name)
        newFileName = base + session_key + ext

        if (fs.exists(newFileName)):
            fs.delete(newFileName)
        fs.save(newFileName, myfile)

        request.session['filename'] = newFileName

        code.append("data = pd.read_csv('{}')".format(myfile.name))

        data = pd.read_csv('./media/{}'.format(newFileName))
        
        plt.switch_backend('Agg')
        sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
        fig_location = './media/correlational_matrix{}.png'.format(session_key)
        plt.savefig(fig_location)
        image_url = '../media/correlational_matrix{}.png'.format(session_key)
        # data = data.head(10)
        
        data_html = data.head(10).to_html()
        data_shape, nullValues, columns = getStatistics(data)

        context = getContext(data_html, data_shape, nullValues, code, columns)
        context.update({'image_url':image_url})

        # print(code)
        return render(request, './main.html', context)
    return render(request, './index.html')


def showFulldataset(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    data_html = data.to_html()
    data_shape, nullValues, columns = getStatistics(data)

    context = getContext(data_html, data_shape, nullValues, code, columns)

    return render(request, './showFulldataset.html', context)


def preprocessing(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)

    context = getContext(data_html, data_shape, nullValues, code, columns)

    return render(request, './preprocessing.html', context)


def dropingnull(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    data = data.dropna()
    code.append('data.dropna()')
    print('dropingnull')
    data.to_csv('./media/{}'.format(filename), index=False)
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)

    context = getContext(data_html, data_shape, nullValues, code, columns)

    return render(request, './preprocessing.html', context)


def minmaxScaler(request):
    # if request
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        X1 = request.POST.getlist('value-x')
        min_range = request.POST.get('start_of_range')
        max_range = request.POST.get('end_of_range')
        print(min_range, max_range)
        columns = []
        for col in X1:
            columns.append(col)
        min_max_scaler = MinMaxScaler(
            feature_range=(int(min_range), int(max_range)))
        data[columns] = min_max_scaler.fit_transform(data[columns])
        code.append("minmax_scaler()")
        data.to_csv('./media/{}'.format(filename), index=False)
        data_html = data.head(10).to_html()
        data_shape, nullValues, columns = getStatistics(data)
        context = getContext(data_html, data_shape, nullValues, code, columns)
        return render(request, './preprocessing.html', context)
    # only integer and float type columns will be send
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    columns_send = []
    for col in data.columns:
        datatypes = data.dtypes[col]
        if datatypes == 'float64' or datatypes == 'int64':
            columns_send.append(col)
    context = getContext(data_html, data_shape, nullValues, code, columns_send)
    return render(request, './minmaxScale.html', context)


def standard_Scaler(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        X1 = request.POST.getlist('value-x')
        columns = []
        for col in X1:
            columns.append(col)
        scaler = StandardScaler()
        model = scaler.fit(data[columns])
        data[columns] = model.transform(data[columns])
        code.append("standard_scaler()")
        print("standard_scaler")
        data.to_csv('./media/{}'.format(filename), index=False)
        data_html = data.head(10).to_html()
        data_shape, nullValues, columns = getStatistics(data)
        context = getContext(data_html, data_shape, nullValues, code, columns)
        return render(request, './preprocessing.html', context)

    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    # only integer and float type columns will be send
    columns_send = []
    for col in data.columns:
        datatypes = data.dtypes[col]
        if datatypes == 'float64' or datatypes == 'int64':
            columns_send.append(col)
    context = getContext(data_html, data_shape, nullValues, code, columns_send)
    return render(request, './standardScale.html', context)


def fillingNullMean(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        mean_of_columns = data.mean()
        columns = request.POST.getlist('value-x')
        print(len(columns))
        for col in range(len(columns)):
            code.append('data.fillna({})'.format(columns[col]))
            try:
                data[columns[col]].fillna(
                    mean_of_columns[columns[col]], inplace=True)
                print(columns[col], "mean")
            except:
                print(columns[col])
                continue
        print(data.isnull().sum())
        data.to_csv('./media/{}'.format(filename), index=False)
        data_html = data.head(10).to_html()
        data_shape, nullValues, columns = getStatistics(data)
        context = getContext(data_html, data_shape, nullValues, code, columns)
        return render(request, './preprocessing.html', context)
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    columns_send = []
    for col in data.columns:
        datatypes = data.dtypes[col]
        if datatypes == 'float64' or datatypes == 'int64':
            columns_send.append(col)
    context = getContext(data_html, data_shape, nullValues, code, columns_send)
    return render(request, './meanForm.html', context)


def fillingNullMedian(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        median_of_columns = data.median()
        columns = request.POST.getlist('value-x')
        print(len(columns))
        for col in range(len(columns)):
            code.append('data.fillna({})'.format(columns[col]))
            try:
                data[columns[col]].fillna(
                    median_of_columns[columns[col]], inplace=True)
                print(columns[col], "median")
            except:
                print(columns[col])
                continue

        print(data.isnull().sum())
        data.to_csv('./media/{}'.format(filename), index=False)
        data_html = data.head(10).to_html()
        data_shape, nullValues, columns = getStatistics(data)
        context = getContext(data_html, data_shape, nullValues, code, columns)
        return render(request, './preprocessing.html', context)
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    columns_send = []
    for col in data.columns:
        datatypes = data.dtypes[col]
        if datatypes == 'float64' or datatypes == 'int64':
            columns_send.append(col)
    context = getContext(data_html, data_shape, nullValues, code, columns_send)
    return render(request, './medianForm.html', context)


def fillingNullMode(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        columns = request.POST.getlist('value-x')
        print(len(columns))
        for col in range(len(columns)):
            code.append('data.fillna({})'.format(columns[col]))
            try:
                data[columns[col]].fillna(
                    data.mode()[columns[col]][0], inplace=True)
                print(columns[col], "mode")
            except:
                print(columns[col])
                continue
        print(data.isnull().sum())
        data.to_csv('./media/{}'.format(filename), index=False)
        data_html = data.head(10).to_html()
        data_shape, nullValues, columns = getStatistics(data)
        context = getContext(data_html, data_shape, nullValues, code, columns)
        return render(request, './preprocessing.html', context)
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    columns_send = []
    for col in data.columns:
        datatypes = data.dtypes[col]
        if datatypes == 'float64' or datatypes == 'int64':
            columns_send.append(col)
    context = getContext(data_html, data_shape, nullValues, code, columns_send)
    return render(request, './modeForm.html', context)


def fillingNullModeNumeric(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    # handle this in post request refer to minmax scaler for this
    # for col in data.columns:
    #     if data[col].dtypes == object:
    #         pass
    #     else:
    #         try:
    #             data[col].fillna(data.mode()[col][0], inplace=True)
    #         except:
    #             print(col)
    #             continue
    # code.append('data.fillna(data.modenumeric())')
    # print("mode_numeric")
    # data.to_csv('./media/{}'.format(filename),index=False)
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    context = getContext(data_html, data_shape, nullValues, code, columns)
    return render(request, './modeForm.html', context)


def deleteColumns(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    name = request.GET.get('name')
    print(name)
    if name is not None:
        data = data.drop([name], axis=1)
        code.append('data.drop{}'.format([name]))
        print("deletecol")
        data.to_csv('./media/{}'.format(filename), index=False)
        data_html = data.head(10).to_html()
        data_shape, nullValues, columns = getStatistics(data)
        context = getContext(data_html, data_shape, nullValues, code, columns)
        return render(request, './preprocessing.html', context)


def mlalgorithms(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    context = getContext(data_html, data_shape, nullValues, code, columns)
    return render(request, './ml.html', context)


def logistic_reg(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        X1 = request.POST.getlist('value-x')
        y1 = request.POST['value-y']
        test_size1 = request.POST['test_size']
        if len(X1) == 1:
            X = data[X1].values.reshape(-1, 1)
        else:
            X = data[X1]
        y = data[y1]
        test1 = int(test_size1)/100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=int(test_size1)/100, random_state=10)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        confusion = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        # print(accuracy)
        variance_score = model.score(X_test, y_test)
        # code.append(" X-{},y-{},X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={}, random_state=10) , model = LogisticRegression(),model.fit(X_train, y_train)y_pred = model.predict(X_test) ,confusion = confusion_matrix(y_test, y_pred),accuracy = accuracy_score(y_test, y_pred)".format(X1,y1,test1))
        code1 = "X-{}".format(X1)
        code2 = "y-{}".format(y1)
        code3 = "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={}, random_state=10)".format(
            test1)
        code.append(code1)
        code.append(code2)
        code.append(code3)
        code4 = [" model = LogisticRegression()", "model.fit(X_train, y_train)", "y_pred = model.predict(X_test)",
                 "confusion = confusion_matrix(y_test, y_pred)", "accuracy = accuracy_score(y_test, y_pred)"]
        code.extend(code4)

        plt.switch_backend('Agg')
        # plt.plot(X_test, y_test, c="green")
        # plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        # plt.axhline(.5, color="red", label="cutoff")

        sns.regplot(x=X_test, y=y_test, data=data, logistic=True, ci=None,scatter_kws = {'color': 'black'}, line_kws = {'color': 'red'})
        
        session_key = request.session.get('session_key', None)

        fig_location = './media/logisticReg{}.png'.format(session_key)
        plt.savefig(fig_location)

        image_url = '../media/logisticReg{}.png'.format(session_key)

        context = {'accuracy': accuracy,
                   'variance_score': variance_score, 'y_predict': y_pred, 'image_url': image_url}

        return render(request, './results.html', context)
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    context = getContext(data_html, data_shape, nullValues, code, columns)
    return render(request, './logistic.html', context)


def linear_reg(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        X1 = request.POST.getlist('value-x')
        y1 = request.POST['value-y']
        test_size1 = request.POST['test_size']
        if len(X1) == 1:

            X = data[X1].values.reshape(-1, 1)
        else:
            X = data[X1]

        y = data[y1]
        test1 = int(test_size1)/100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=int(test_size1)/100, random_state=10)
        model = LinearRegression()
        model.fit(X_train, y_train)
        variance_score = model.score(X_test, y_test)
        # print('Variance score: {}'.format(model.score(X_test, y_test)))
        # ---linear-regression doesnt have confusion matrix nOTE
        y_pred = model.predict(X_test)
        # confusion = confusion_matrix(y_test, y_pred)
        # accuracy = accuracy_score(y_test, y_pred)
        # accuracy="NA" #not avialable so kept zero
        score = r2_score(y_test, y_pred)

        plt.switch_backend('Agg')
        plt.scatter(X_test[:, 0], y_test, color="black")
        plt.plot(X_test, y_pred, color="blue", linewidth=3)

        session_key = request.session.get('session_key', None)

        fig_location = './media/linearReg{}.png'.format(session_key)
        plt.savefig(fig_location)

        image_url = '../media/linearReg{}.png'.format(session_key)

        # code.append(" X-{},y-{},X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={}, random_state=10) , model = LinearRegression(),model.fit(X_train, y_train)y_pred = model.predict(X_test) ,variance_score=model.score(X_test,y_test),accuracy = accuracy_score(y_test, y_pred) , y_pred= ".format(X1,y1,test1))
        code1 = "X-{}".format(X1)
        code2 = "y-{}".format(y1)
        code3 = "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={}, random_state=10)".format(
            test1)
        code.append(code1)
        code.append(code2)
        code.append(code3)
        code4 = ["model = LinearRegression()", "model.fit(X_train, y_train)", "y_pred = model.predict(X_test)",
                 "variance_score=model.score(X_test,y_test)", "accuracy = accuracy_score(y_test, y_pred)"]
        code.extend(code4)

        context = {'r2_score': score,
                   'image_url': image_url, 'backgroundCode': code}

        return render(request, './results.html', context)
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    context = getContext(data_html, data_shape, nullValues, code, columns)
    return render(request, './linear.html', context)


def knn(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        no_of_neighbours = request.POST['no_of_neighbors']
        X1 = request.POST.getlist('value-x')
        y1 = request.POST['value-y']
        test_size1 = request.POST['test_size']
        X = data[X1]
        y = data[y1]

        test1 = int(test_size1)/100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=int(test_size1)/100, random_state=10)
        knn = KNeighborsClassifier(int(no_of_neighbours))
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("accuracy: ", accuracy)
        variance_score = knn.score(X_test, y_test)

        # print("Successfylyyy",y_pred)
        # code.append(" X-{},y-{},X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={}, random_state=10) ,  knn=KNeighborsClassifier(int({})),knn.fit(X_train,y_train),y_pred=knn.predict(X_test),accuracy = accuracy_score(y_test, y_pred),\nvariance_score=knn.score(X_test,y_test)".format(X1,y1,test1,no_of_neighbours))
        code1 = "X-{}".format(X1)
        code2 = "y-{}".format(y1)
        code3 = "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={}, random_state=10)".format(
            test1)
        code4 = " knn=KNeighborsClassifier(int({})),knn.fit(X_train,y_train)".format(
            no_of_neighbours)

        code.append(code1)
        code.append(code2)
        code.append(code3)
        code.append(code4)
        code5 = ["y_pred = knn.predict(X_test)", "accuracy = accuracy_score(y_test, y_pred)",
                 "variance_score=knn.score(X_test,y_test)"]
        code.extend(code5)

        plt.switch_backend('Agg')
        plot_decision_regions(X_test.values, y_test.values, knn)

        session_key = request.session.get('session_key', None)

        fig_location = './media/knn{}.png'.format(session_key)
        plt.savefig(fig_location)
        image_url = '../media/knn{}.png'.format(session_key)

        context = {'accuracy': accuracy, 'variance_score': variance_score,
                   'y_predict': y_pred, 'image_url': image_url}
        return render(request, './results.html', context)
    data_html = data.to_html()
    data_shape, nullValues, columns = getStatistics(data)
    context = getContext(data_html, data_shape, nullValues, code, columns)
    return render(request, './knn.html', context)


def kmeans(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))

 # -------->We will provide Elbow method to user so that he could figure out number of clusters<-------------------
 # elbow method is plotting of scatter plot so 2 options we have either in visualization section or in Kmeans section will decide<----
 # After that we will process K means algo ...
    if request.method == 'POST':
        no_of_clusters = request.POST['no_of_clusters']
        X1 = request.POST.getlist('value-x')
        # no test size required in this algo
        X = data[X1]
        kmeans = KMeans(n_clusters=int(no_of_clusters),
                        init='k-means++', random_state=42)
        y_pred = kmeans.fit_predict(X)
        accuracy = "NA"  # not avialable so kept zero
        variance_score = "NA"  # not avialable so kept zero

        code1 = "x={}".format(X1)
        code2 = "kmeans = KMeans(n_clusters=int({}) init='k-means++',random_state= 42)".format(
            no_of_clusters)
        code3 = ["y_pred=kmeans.fit_predict(X)"]
        code.append(code1)
        code.append(code2)
        code.extend(code3)
        # code.append("x={},kmeans = KMeans(n_clusters=int({}), init='k-means++', random_state= 42),y_pred=kmeans.fit_predict(X),accuracy= ,variance_score= ".format(X1,no_of_clusters))
        # Need to pass on plots of cluster as output...

        # Visualization
        plt.switch_backend('Agg')
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, s=50, cmap='viridis')
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

        session_key = request.session.get('session_key', None)

        fig_location = './media/kmeans{}.png'.format(session_key)
        plt.savefig(fig_location)
        image_url = '../media/kmeans{}.png'.format(session_key)

        context = {'accuracy': accuracy, 'variance_score': variance_score,
                   'y_predict': y_pred, 'image_url': image_url}

        # different template will come need to change kept it temprory
        return render(request, './results.html', context)

    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    context = getContext(data_html, data_shape, nullValues, code, columns)
    return render(request, './kmeans.html', context)


def cat_data(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        columns = request.POST.getlist('value-x')
        for col in columns:
            label_encoder = LabelEncoder()
            data[col] = label_encoder.fit_transform(data[col])
            data[col].unique()
            # we can also pass value of each category for user info in statistics
            # print(label_encoder.fit_transform(data[col]))
        data.to_csv('./media/{}'.format(filename), index=False)
        data_html = data.head(10).to_html()
        data_shape, nullValues, columns = getStatistics(data)
        context = getContext(data_html, data_shape, nullValues, code, columns)
        return render(request, './preprocessing.html', context)
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    columns_send = []
    for col in data.columns:
        datatypes = data.dtypes[col]
        if datatypes != 'float64' and datatypes != 'int64':
            columns_send.append(col)
    context = getContext(data_html, data_shape, nullValues, code, columns_send)
    return render(request, './categoricalData_form.html', context)


def visualisation(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    data_html = data.head(10).to_html()
    plt.switch_backend('Agg')
    sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
    session_key = request.session.get('session_key', None)
    fig_location = './media/correlational_matrix{}.png'.format(session_key)
    plt.savefig(fig_location)
    image_url = '../media/correlational_matrix{}.png'.format(session_key)
    data_shape, nullValues, columns = getStatistics(data)
    context = getContext(data_html, data_shape, nullValues, code, columns)
    context.update({'image_url':image_url})
    return render(request, './visualisation.html', context)


def pie_chart(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        column= request.POST.getlist('value-x')
        # print(data[column].value_counts().count())
        grouped=data.groupby(column).groups
        categories=[]     #all categories names
        for group in grouped:
            categories.append(group)
        single_column=data[column].value_counts()  #particular column total count
        cnts=[]  # for storing cnt of individual categories
        for cnt in categories:
            cnts.append(single_column[cnt])
        plt.switch_backend('Agg')
        plt.pie(cnts,labels=categories,autopct='%.0f%%')
        session_key = request.session.get('session_key', None)
        fig_location = './media/pieplot{}.png'.format(session_key)
        plt.savefig(fig_location)
        image_url = '../media/pieplot{}.png'.format(session_key)
        context={"image_url":image_url}
        return render(request, './visualization_output.html', context)
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    columns_send = []
    for col in data.columns:
        datatypes = data.dtypes[col].name
        if datatypes != 'float64' and datatypes != 'int64':
            columns_send.append(col)
    context = getContext(data_html, data_shape, nullValues, code, columns_send)
    return render(request, './pie_chart.html', context)


def histogram(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        column= request.POST.getlist('value-x')
        plt.switch_backend('Agg')
        sns.histplot(data[column])
        session_key = request.session.get('session_key', None)
        fig_location = './media/histoplot{}.png'.format(session_key)
        plt.savefig(fig_location)

        image_url = '../media/histoplot{}.png'.format(session_key)
        context={"image_url":image_url}
        return render(request, './visualization_output.html', context)
       
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    columns_send = []
    for col in data.columns:
        datatypes = data.dtypes[col]
        if datatypes == 'float64' or datatypes == 'int64':
            columns_send.append(col)
    context = getContext(data_html, data_shape, nullValues, code, columns_send)
    return render(request, './histogram.html', context)


def box_plot(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        columns= request.POST.getlist('value-x')
        plt.switch_backend('Agg')
        sns.boxplot(data[columns])
        session_key = request.session.get('session_key', None)
        fig_location = './media/boxplot{}.png'.format(session_key)
        plt.savefig(fig_location)

        image_url = '../media/boxplot{}.png'.format(session_key)
        context={"image_url":image_url}
        return render(request, './visualization_output.html', context)
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    columns_send = []
    for col in data.columns:
        datatypes = data.dtypes[col]
        if datatypes == 'float64' or datatypes == 'int64':
            columns_send.append(col)
    context = getContext(data_html, data_shape, nullValues, code, columns_send)
    return render(request, './box_plot.html', context)


def line_plot(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        columns= request.POST.getlist('value-x')
        plt.switch_backend('Agg')
        sns.lineplot(data[columns])
        session_key = request.session.get('session_key', None)
        fig_location = './media/lineplot{}.png'.format(session_key)
        plt.savefig(fig_location)

        image_url = '../media/lineplot{}.png'.format(session_key)
        context={"image_url":image_url}
        return render(request, './visualization_output.html', context)
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    columns_send = []
    for col in data.columns:
        datatypes = data.dtypes[col]
        if datatypes == 'float64' or datatypes == 'int64':
            columns_send.append(col)
    context = getContext(data_html, data_shape, nullValues, code, columns_send)
    return render(request, './line_plot.html', context)

def elbow_plot(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    if request.method == 'POST':
        columns= request.POST.getlist('value-x')
        #loop_cnt cant be more than the record size otherwise it willgive the error # ERROR HANDLING
        loop_cnt=request.POST.get('test_for_no_of_clusters')
        print(columns,loop_cnt)
        wcss_list= []      
        for i in range(1, int(loop_cnt)+1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)  
            kmeans.fit(data[columns])  
            wcss_list.append(kmeans.inertia_)
        plt.switch_backend('Agg')  
        plt.plot(range(1, int(loop_cnt)+1), wcss_list)  
        plt.title('The Elobw Method Graph')  
        plt.xlabel('Number of clusters(k)')  
        plt.ylabel('wcss_list')  
        session_key = request.session.get('session_key', None)
        fig_location = './media/elbowplot{}.png'.format(session_key)
        plt.savefig(fig_location)

        image_url = '../media/elbowplot{}.png'.format(session_key)
        context={"image_url":image_url}
        return render(request, './visualization_output.html', context)
    



    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    columns_send = []
    for col in data.columns:
        datatypes = data.dtypes[col].name
        if datatypes == 'float64' or datatypes == 'int64':
            columns_send.append(col)
    context = getContext(data_html, data_shape, nullValues, code, columns_send)
    return render(request, './elbow_plot.html', context)

def landing(request):
    filename = request.session.get('filename', None)
    data = pd.read_csv('./media/{}'.format(filename))
    data_html = data.head(10).to_html()
    data_shape, nullValues, columns = getStatistics(data)
    columns_send = []
    for col in data.columns:
        datatypes = data.dtypes[col].name
        if datatypes != 'float64' and datatypes != 'int64':
            columns_send.append(col)
    context = getContext(data_html, data_shape, nullValues, code, columns_send)
    return render(request, './landing.html', context)
