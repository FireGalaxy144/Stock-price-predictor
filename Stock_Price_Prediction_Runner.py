import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")

training_set = dataset_train.iloc[:,1:2].values
scaler = MinMaxScaler(feature_range = (0,1))
scaled_training_set = scaler.fit_transform(training_set)


actual_stock_price = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis= 0)
inputs = dataset_total[len(dataset_total)- len(dataset_test)-60:].values

inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
new_model = tf.keras.models.load_model('my_model.keras')
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))

preditcted_stock_price =  new_model.predict(X_test)
preditcted_stock_price = scaler.inverse_transform(preditcted_stock_price)

plt.plot(actual_stock_price, color = 'red', label = 'Actual Google Stock Price')
plt.plot(preditcted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()