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
#import math
def create_data(x, look_back=1):
    dataX, datay = [], []
    for i in range(4000):
        a = X[i:i + look_back,1]
        dataX.append(a)
        datay.append(X[i + look_back,1])
    return np.array(dataX), np.array(datay)
    
X, Y = create_data(X, look_back=100 )
#print(X, Y)
""" plt.plot(X)
plt.plot(Y)
plt.show()"""

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
#Xout = np.reshape(X, ( 1, X.shape))

print(X.shape)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
look_back=100
model = Sequential()
model.add(LSTM(8, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=5, batch_size=1, verbose=2)

trainPredict = model.predict(X)
print(trainPredict)
plt.plot(trainPredict)
plt.plot(Y)
plt.show()