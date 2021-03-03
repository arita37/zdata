import os, copy
import pandas as pd, numpy as np

import importlib
import math
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List
from pathlib import Path


import mdn

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.regularizers import l2
from keras.optimizers import Adam






###################################################################################
###########Funtions ROLLING FORCAST AND SUPERVISED CLASSIFICATION OF TIMESERIES####
###################################################################################

# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/



# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size=1, nb_epoch=5, lstm_neurons=1,timesteps=1,dense_neurons=1,mdn_output=1,mdn_Nmixes=1):
  X, y = train[:, 0:-1], train[:, -1]
  #print(X.shape)
  X = X.reshape(X.shape[0], timesteps, X.shape[1])
  model = Sequential()
  model.add(LSTM(lstm_neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
  model.add(Dense(dense_neurons))
  model.add(mdn.MDN(mdn_output,mdn_Nmixes))

  adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08,   decay=0.0)
  model.compile(loss=mdn.get_mixture_loss_func(mdn_output,mdn_Nmixes),
             optimizer=adam )
  print(model.summary())
  for i in range(nb_epoch):
    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
    model.reset_states()
  return model

# make a one-step forecast
def forecast_lstm(model, batch_size,timesteps, X):
	X = X.reshape(1, timesteps, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

  
 


####################################################

def test_supervised() :
    pass
      ######################################
    # Test supervised timeseries classification with armdn
 

#################################################################
########EXAMPLE CODE TO RUN ARMDN#######################
#############################################################################


timesteps=1
interval=1
test_values=30
batch_size, nb_epoch, neurons  = 1,10,1
data_folder = "kaggle_data/synthetic_data"
def myfun(t):
       return 20*np.sin(2*np.pi *2* t/100)
df=create_timeseries1d(fun= myfun)
df.to_csv(data_folder+'/cos_timeseries.csv')
series = pd.read_csv(data_folder+'/cos_timeseries.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values,interval)
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, timesteps)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-test_values], supervised_values[-test_values:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
lstm_model = fit_lstm(train_scaled, batch_size, nb_epoch,neurons,timesteps)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), timesteps,1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
  # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, batch_size,timesteps, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+interval-i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i +1]
    print('Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))


# report performance
rmse = math.sqrt(mean_squared_error(raw_values[-test_values:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(raw_values[-test_values:])
plt.plot(predictions)
plt.show()

'''

if __name__ == '__main__':
   VERBOSE = False
   import numpy as np
   def myfun(t):
       return 20*np.sin(2*np.pi *2* index/100)
       
   df=create_timeseries1d(fun= myfun)
   print(df.head(5))


    #test()
