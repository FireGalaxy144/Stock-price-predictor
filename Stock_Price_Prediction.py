import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#%matplotlib inline 



dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
dataset_train.head()

training_set = dataset_train.iloc[:,1:2].values
print(training_set)
print(training_set.shape)

from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range = (0,1))
scaled_training_set = scaler.fit_transform(training_set)

scaled_training_set

X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(scaled_training_set[i-60:i, 0])
    y_train.append(scaled_training_set[i,0])
X_train = np.array(X_train)
y_train = np.array(y_train)

print(X_train.shape)
print(y_train.shape)

X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1 ))

X_train.shape

from keras.models import Sequential 
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout 

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences= True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences= True))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss= 'mean_squared_error')
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

regressor.save('my_model.keras')

