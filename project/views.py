from django.shortcuts import render;
from django.http import HttpResponse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle


def home(request):
    return render(request,"home.html")

def predict(request):
    return render(request,"predict.html")

def result(request):

    data=pd.read_csv(r'C:\Users\USER\Machine Learning Libraries\ML\P\airline22.csv')
    data=data.drop(['duration'],axis=1)

    data["Journey_day"] = pd.to_datetime(data.date, format="%d/%m/%Y").dt.day
    data["Journey_month"] = pd.to_datetime(data["date"], format = "%d/%m/%Y").dt.month
    data["Journey_year"] = pd.to_datetime(data["date"], format = "%d/%m/%Y").dt.year

    data=data.drop(["date"], axis = 1,)

    airline=LabelEncoder()
    source=LabelEncoder()
    dest=LabelEncoder()
    clas=LabelEncoder()

    data['airline']=airline.fit_transform(data['airline'])
    data['source_city']=source.fit_transform(data['source_city'])
    data['destination_city']=dest.fit_transform(data['destination_city'])
    data['class']=clas.fit_transform(data['class'])

    x=data.drop(['price'],axis=1)
    y=data['price']

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=2)

    scale=MinMaxScaler()
    x_train=scale.fit_transform(x_train)
    x_test=scale.transform(x_test)

    model=DecisionTreeRegressor(max_depth=3)
    model.fit(x_train,y_train)

    var1 = (request.GET['n1'])
    var2 = (request.GET['n2'])
    var3 = (request.GET['n3'])
    var4=  (request.GET["n4"])
    var5 = (request.GET['n5'])
    var6 = (request.GET['n6'])
    var7 = (request.GET['n7'])
    var8 = (request.GET['n8'])

   
    Airlines=var1
    Airlines=int(airline.transform([Airlines]))

    Sourcecity=var2
    Sourcecity=int(source.transform([Sourcecity]))

    Destinationcity=var3
    Destinationcity=int(dest.transform([Destinationcity]))  

    Stops=var4

    Cabinclass=var5
    Cabinclass=int(clas.transform([Cabinclass]))

    Day=var6
    Month=var7
    Year=var8

    scaled=scale.transform([[Airlines,Sourcecity,Destinationcity,Stops,Cabinclass,Day,Month,Year]])
    result=model.predict(scaled)
    

    price = "The Predicted Flight Price is Rs "+str(result)
    return render(request,"predict.html",{"result2":price})  