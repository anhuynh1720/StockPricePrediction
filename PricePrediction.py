import altair as alt
import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, LSTM

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping

start = "2010-01-01"
end = date.today()

st.set_page_config(
    page_title="Real-Time Stock Price Prediction Dashboard",
    page_icon="⏳",
    layout="wide",
)
t1, t2 = st.columns((0.07,1)) 

t1.image('./logo.png', width = 80)
t2.title("Real-Time Stock Price Prediction Dashboard")


###
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)
df.index = pd.to_datetime(df.index)

#Describing Data
# st.subheader('Data from 2010 - Present')
# st.write(df.describe())

#Visualization
st.subheader('Closing Price vs Time chart')
st.line_chart(df.Close, use_container_width=True)

#Splitting Data into Training and Testing
df1 = df[['Close']]
scaler = MinMaxScaler(feature_range=(0,1))
df1_new = scaler.fit_transform(df1)
train_size = round(len(df1_new) * 75/100) 
train, test = df1_new[0:train_size, :], df1_new[train_size:, :]

def create_dataset(dataset, look_back=1): 
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
	#t
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
	#t+1
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# Build module
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model1 = Sequential()
model1.add(LSTM(units = 128, return_sequences = True, input_shape = (1, look_back)))
model1.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model1.add(LSTM(units = 256, return_sequences = True))
model1.add(Dropout(0.3))

# Adding a third LSTM layer and some Dropout regularisation
model1.add(LSTM(units = 256, return_sequences = True))
model1.add(Dropout(0.3))

# Adding a fourth LSTM layer and some Dropout regularisation
model1.add(LSTM(units = 256))
model1.add(Dropout(0.2))

# Adding the output layer
model1.add(Dense(units = 1))
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.summary()

early_stopping = EarlyStopping(min_delta = 0.0001, patience = 50, restore_best_weights = True)
history = model1.fit(trainX, trainY, validation_data=(testX, testY), 
                    epochs=500, 
                    batch_size=128,
                    verbose=1,
                    callbacks = [early_stopping])

# make predictions
trainPredict = model1.predict(trainX)
testPredict = model1.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# EVALUATION
# calculate root mean squared error
train_rmse = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train RMSE: %.2f RMSE' % (train_rmse))
test_rmse = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE: %.2f RMSE' % (test_rmse))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(df1_new)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1_new)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(df1_new)-1, :] = testPredict

# plot baseline and predictions
st.subheader("Prediction Stock Ticker Price - " + user_input)
fig_LSTM = plt.figure()
plt.plot(scaler.inverse_transform(df1_new), label='Real Data')
plt.plot(trainPredictPlot, label='Train Predict')
plt.plot(testPredictPlot, label='Test Predict')
plt.legend(title="Notes")
plt.ylabel("Price (USD)")
plt.title("LTSM")
plt.grid()
plt.xticks([])

st.plotly_chart(fig_LSTM, use_container_width=True)



