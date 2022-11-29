#--------------------------------------------------about problem 
#Breast cancer classification divides breast cancer into categories according to different schemes criteria and serving a different purpose. 
# The major categories are the histopathological type, the grade of the tumor, the stage of the tumor, and the expression of proteins and genes.
#  As knowledge of cancer cell biology develops these classifications are updated.

#----------------------------------------------------work flow
#dataset download
#import labrary
#data anlysis and visulazation
#data separtion into x, y
#tarin and test data separtion
#model selection and fit model
#train and test data prediction 

#----------------------------------------------------import useful labrary
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#-----------------------------------------------------dataset anlysis
data = pd.read_csv("C:/Users/kunde/all vs code/ml prject/data.csv")
print(data.shape)
print(data.columns)
print(data.info())
print(data.describe())
print(data.isnull().sum())
print(data.head(5))
print(data.tail(5))
data = data.drop(columns=["Unnamed: 32", "id"], axis=1)
print(data["diagnosis"].value_counts())
data.replace({"diagnosis": { "M" : 0, "B" : 1}}, inplace=True)
print(data.head(5))
#------------------------------------------------------dataset separtion
y = data["diagnosis"]
x = data.drop(columns=["diagnosis"], axis=1)
print(x.head(5))

#-----------------------------------------------------train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
print(x.shape, x_train.shape, x_test.shape)
print(y.shape, y_train.shape, y_test.shape)

#-----------------------------------------------------model anlysis
model = LogisticRegression()
model.fit(x_train, y_train)

#-------------------------------------------------------train data prediction
y_tr = model.predict(x_train)
accur = accuracy_score(y_tr, y_train)
print(accur, 'this is accurancy of train data')

#-----------------------------------------------------test data prediction
y_te = model.predict(x_test)
accur =  accuracy_score(y_te, y_test)
print(accur, "this is test data accurancy")

#------------------------------------------------single data prediction
a = [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]
print(len(a))
arr = np.asarray(a)
arra = arr.reshape(1, -1)
y_pre = model.predict(arra)
print(y_pre, "this is single data prediction and true value is M(0)")


















