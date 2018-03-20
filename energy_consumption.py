# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 00:07:02 2018

@author: Abhilash Srivastava
"""

# import data 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
data_file = 'household_power_consumption.txt'
data = pd.read_csv(data_file,delimiter=';')
data = data.drop(['Date','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'],axis=1)
X = data
X.replace('?', 0)
print(X.columns)
X = np.array(X)
import math
def create_data(x, look_back=1):
    dataX, datay = [], []
    for i in range(4000):
        a = X[i:i + look_back,1]
        dataX.append(a)
        datay.append(X[i + look_back,1])
    return np.array(dataX), np.array(datay)
    
X, Y = create_data(X, look_back=100 )

X = np.reshape(X, (X.shape[0], X.shape[1]))
#Xout = np.reshape(X, ( 1, X.shape[0]))
print(X.shape)
from sklearn import svm,cross_validation

#x_train,y_train,x_test,y_test=cross_validation.train_test_split(X,Y,test_size=0.2)

#clf=svm.SVC(gamma=0.001,C=100)
clf=svm.SVR()

clf.fit(X,Y)

test=clf.predict(X)

fig=plt.figure()
fig.subplots_adjust(hspace=0.5)
ax1=fig.add_subplot(1,2,1)
ax1.plot(test)
ax1.set_xlabel('time')
ax1.set_ylabel('Global_active_power')
ax1.set_title('predicted')
ax2=fig.add_subplot(2,2,2)
ax2.plot(test)
ax2.set_xlabel('time')
ax2.set_ylabel('Global_active_power')
ax2.set_title('recored')

'''
plt.subplot(2,1,1)

p1=plt.plot(test)
plt.setp(p1, color='r')
plt.xlabel('time')
plt.ylabel('Global_active_power')
plt.title('predicted')
plt.subplot(2,1,2)
p2=plt.plot(Y)
plt.setp(p2, color='b')

plt.xlabel('time')
plt.ylabel('Global_active_power')
plt.title('recorded')
plt.show()
'''