#Import Libraries and Read the data
import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
import xgboost
from sklearn.model_selection import train_test_split

data  =  pd.read_csv("D://Blogs//Iris Multiple Algo//Iris.csv")
#Create Dependent and Independent Datasets based on our Dependent #and Independent features
X  = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm']]
y= data['Species']

#Split the Data into Training and Testing sets with test size as #30%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, shuffle=True)



