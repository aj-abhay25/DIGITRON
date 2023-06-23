import streamlit as st
from datetime import date
import numpy as np
import yfinance as yf
from plotly import graph_objs as go
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Visualizations
st.subheader('Closing Price vs Time Chart')
fig = go.Figure(layout=go.Layout(width=600, height=300))
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
st.plotly_chart(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = data['Close'].rolling(100).mean()

fig = go.Figure(layout=go.Layout(width=600, height=300))

fig.add_trace(go.Scatter(x=data.index, y=ma100, name='100MA'))
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))

st.plotly_chart(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = data['Close'].rolling(100).mean()
ma200 = data['Close'].rolling(200).mean()
fig = go.Figure(layout=go.Layout(width=600, height=300))
fig.add_trace(go.Scatter(x=data.index, y=ma100, name='100MA'))
fig.add_trace(go.Scatter(x=data.index, y=ma200, name='200MA'))
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
st.plotly_chart(fig)


# Splitting data into training and testing

data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70): int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Load my model
model = load_model('C:\\Users\\Admin\\TradePreNew-main\\Stock_Predictor\\code_jupyter\\keras_model.h5')

#Testing part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#Plotting predicted vs actual prices
plt.plot(y_test, color = 'red', label = 'Real Price')
plt.plot(y_predicted, color = 'blue', label = 'Predicted Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

#Calculating the mean squared error (MSE)
mse = np.mean((y_predicted - y_test)**2)
st.write("Mean Squared Error (MSE) = ", mse)
# Forecasting for next n years
def predict_prices(model, n_years, input_data, scale_factor):
    last_100_days = input_data[-100:]
    input_data = np.reshape(last_100_days, (last_100_days.shape[0], last_100_days.shape[1]))
    input_data = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1]))

    y_pred = model.predict(input_data)
    y_pred = y_pred * scale_factor
    y_pred = y_pred[0][0]

    predictions = []
    for i in range(n_years * 365):
        last_100_days = np.append(last_100_days[1:], [[y_pred]], axis=0)
        input_data = np.reshape(last_100_days, (last_100_days.shape[0], last_100_days.shape[1]))
        input_data = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1]))
        y_pred = model.predict(input_data)
        y_pred = y_pred * scale_factor
        y_pred = y_pred[0][0]
        predictions.append(y_pred)

    return predictions

predictions = predict_prices(model, n_years, input_data, scale_factor)

# Plotting predicted prices for next n_years
plt.plot(predictions, color='blue', label='Predicted Price')
plt.title('Price Prediction for next ' + str(n_years) + ' years')
plt.xlabel('Time (in days)')
plt.ylabel('Price')
plt.legend()

st.write("Predicted stock prices for next", n_years, "years:")
st.pyplot(plt)