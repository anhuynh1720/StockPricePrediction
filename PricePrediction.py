from tabnanny import check
import altair as alt
import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from datetime import date, timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split



st.set_page_config(
    page_title="Real-Time Stock Price Prediction Dashboard",
    page_icon="⏳",
    layout="wide",
)

st.title("Real-Time Stock Price Prediction Dashboard")
###
start = "2010-01-01"
end = date.today()
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)
df.index = pd.to_datetime(df.index)

l1, l2 = st.columns([2,5])
with l1:
	n = st.text_input('Enter the number of days you want to see the history', '3')
	st.write(df[['High', 'Low', 'Open', 'Close']][-int(n):])
with l2:
	st.subheader('Closing Price vs Time chart')
	real_time_close = df.Close
	st.line_chart(real_time_close, height = 450, use_container_width=True)


df_test = df.tail(100)
df = df.iloc[:-100, :]
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

model1.save("LSTM_one_to_one_LSTM.h5")

import pickle
pickle.dump(scaler, open('scaler.sav', 'wb'))

# Prediction in the future
df_test_new = df_test[['Close']]
real_data = df_test_new['Close'].values

df_test_new['Predict_Close'] = [scaler.inverse_transform(model1.predict(numpy.reshape(scaler.transform([[real_data[0]]]), (1, 1, 1))))[0][0]]+[scaler.inverse_transform(model1.predict(numpy.reshape(scaler.transform([[i]]), (1, 1, 1))))[0][0] for i in real_data[1:]]
df_test_new['Error'] = df_test_new.Close - df_test_new.Predict_Close
df1_close = df_test_new[['Predict_Close']]
df1_close = df1_close.set_index(df_test.index)

def get_tomorrow(df, numberDays):
	tomorrow = pd.to_datetime(pd.Series(df[-1:].index.format())) + timedelta(days=numberDays)
	check_date = tomorrow.dt.weekday.values
	print(check_date)
	if (check_date < 5):
		return tomorrow
	if (check_date == 5):
		return tomorrow + timedelta(days=2)
	if (check_date == 6):
		return tomorrow + timedelta(days=1)

def get_prediction(df_prediction, numberDays):
	if (numberDays == 0):
		return df_prediction
	today_close = df_prediction['Predict_Close'][-1:].values
	tomorrow = get_tomorrow(df_prediction, 1)
	
	tomorrow_prediction = scaler.inverse_transform(model1.predict(numpy.reshape(scaler.transform([today_close]), (1, 1, 1))))[0][0] # tomorrow close value
	data_predict = pd.DataFrame(tomorrow_prediction, columns = ['Predict_Close'], index = tomorrow)
	df_prediction = df_prediction.append(data_predict)

	return get_prediction(df_prediction, numberDays - 1)

def get_tomorrow_predition(df):
	df_prediction = df
	today_close = df_test_new['Close'][-1:].values	
	tomorrow = get_tomorrow(df, 1)
	
	tomorrow_prediction = scaler.inverse_transform(model1.predict(numpy.reshape(scaler.transform([today_close]), (1, 1, 1))))[0][0] # tomorrow close value
	data_predict = pd.DataFrame(tomorrow_prediction, columns = ['Predict_Close'], index = tomorrow)
	df_prediction = df_prediction.append(data_predict)

	return df_prediction

df1_close = get_tomorrow_predition(df1_close) 
df1_close = get_prediction(df1_close, 30)


st.subheader("Prediction Stock Ticker Price - " + user_input)
fig_LSTM = plt.figure()
plt.plot(real_time_close, label='Real Data')
plt.plot(df1_close, label='Prediction')
plt.legend(title="Notes")
plt.ylabel("Price (USD)", fontsize=15)
plt.xlabel("Datetime", fontsize=15)

plt.title("LSTM", fontsize=24)
plt.grid()

st.plotly_chart(fig_LSTM, use_container_width=True)


## XGBOOST
X = df[['Open', 'High', 'Low', 'Volume', 'Adj Close']]
y = df[['Close']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)
ms_X = MinMaxScaler()
X_train = ms_X.fit_transform(X_train)

ms_y = MinMaxScaler()
y_train = ms_y.fit_transform(y_train)

X_test = ms_X.transform(X_test)
y_test = ms_y.transform(y_test)

def evaluate(model, X_train, X_test, y_train, y_test):
    # print("Variance score: ", model.score(X, y))
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    print(
        "\n***TRAINIG RESULTS***: \n=================================================="
    )
    print("R-squared train dataset:", model.score(X_train, y_train))
    print("MSE train dataset:", mean_squared_error(y_train, yhat_train))
    print("RMSE train dataset:", mean_squared_error(y_train, yhat_train, squared=False))
    print("MAE train dataset:", mean_absolute_error(y_train, yhat_train))

    print("\n***TEST RESULTS***: \n===================================================")
    print("R-squared test dataset:", model.score(X_test, y_test))
    print("MSE test dataset:", mean_squared_error(y_test, yhat_test))
    print("RMSE test dataset:", mean_squared_error(y_test, yhat_test, squared=False))
    print("MAE test dataset:", mean_absolute_error(y_test, yhat_test))

import xgboost as xgb
model_xgboost = xgb.XGBRegressor().fit(X_train, y_train)
evaluate(model_xgboost, X_train, X_test, y_train, y_test)
#predictions
df_test1 = df_test[["Close"]]
df_test1["Predict_Close"] = ms_y.inverse_transform(
    [
        model_xgboost.predict(
            ms_X.transform(numpy.reshape(df_test.drop(columns=["Close"]).values, (-1, 5)))
        )
    ]
)[0]

df_test1['Error'] = df_test1["Close"] - df_test1["Predict_Close"]

df_test_xgboost = get_tomorrow_predition(df_test1) 
df_test_xgboost = get_prediction(df_test_xgboost, 30)
xgboost = plt.figure()
plt.plot(df_test_xgboost["Close"], label='Real Data')
plt.plot(df_test_xgboost["Predict_Close"], label='Prediction')

plt.ylabel("Price (USD)", fontsize=15)
plt.xlabel("Datetime", fontsize=15)
plt.axis("equal")

plt.legend(title="Notes")
plt.title("XGBOOST", fontsize=24)
plt.grid()

st.plotly_chart(xgboost, use_container_width=True)

st.subheader("Summary")
tab1, tab2, tab3 = st.tabs(["LSTM Error", "XGBOOST Error", "Summary"])
with tab1:
	st.write("The error between actual price and predicted price - LSTM", df_test_new)

with tab2:
	st.write("The error between actual price and predicted price - XGBOOST", df_test1)

with tab3:
	summary = pd.DataFrame(list(zip(df1_close['Predict_Close'], df_test_xgboost['Predict_Close'])), columns=['LSTM Prediction', 'XGBOOST Prediction'], index=df_test_xgboost.index)
	st.write("Price prediction for the next 30 days of both models:", summary)
