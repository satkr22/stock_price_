import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
# import os
# os.environ['TF_CPP_MIN_LOGLEVEL'] = '2'


st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)


google_data = yf.download(stock, start, end)

model = load_model("stock_price_model.keras")

st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])
# st.write(x_test)

# print(x_test.shape) 

def plt_graph(figSize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figSize)
    plt.plot(values, 'orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig


st.subheader("Original Close price and MA for 250 days")
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plt_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))


st.subheader("Original Close price and MA for 200 days")
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plt_graph((15, 6), google_data['MA_for_200_days'], google_data, 0))


st.subheader("Original Close price and MA for 100 days")
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plt_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))

st.subheader("Original Close price and MA for 100 days and MA for 250 days")
st.pyplot(plt_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))





from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(x_test)

x_data =  []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100: i])
    y_data.append(scaled_data[i])


x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)


plotting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions' : inv_predictions.reshape(-1)
    }, 
    index = google_data.index[splitting_len+100:]
)
st.subheader("Original Values vs Predicted Values")
st.write(plotting_data)

st.subheader("Original CLose Price vs Predicted Close Price")

fig = plt.figure(figsize=(15, 6))

plt.plot(pd.concat([google_data.Close[:splitting_len+100], plotting_data], axis=0))

plt.legend(["Data not used", "Original Test Data", "Predicted Test Data"])

st.pyplot(fig)