from tabnanny import verbose
from time import time
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder


def LTSM_model_build(units, x_train, y_train, dense):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=units, return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
    lstm_model.add(LSTM(units=units))
    lstm_model.add(Dense(dense))

    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    return lstm_model


def RNN_model_build(units, dropout, x_train, y_train, dense):
    rnn_model = Sequential()
    rnn_model.add(LSTM(units=units, return_sequences=True,
                  input_shape=(x_train.shape[1], 1)))
    rnn_model.add(Dropout(dropout))
    for i in [True, True, False]:
        rnn_model.add(LSTM(units=units, return_sequences=i))
        rnn_model.add(Dropout(dropout))

    rnn_model.add(Dense(units=dense))
    rnn_model.compile(optimizer='adam', loss='mean_squared_error')
    rnn_model.fit(x_train, y_train, epochs=10, batch_size=32)
    return rnn_model


def XGBoost_model_build(X_train, y_train, eval_set):
    start_time = time()
    model = XGBRegressor()
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    print('Fit time : ', time() - start_time)
    return model


def forecastingPrice(df, n_lookback, n_forecast, model, feature, dt_freq):
    length = len(df)
    data = df

    new_dataset = pd.DataFrame(index=range(
        0, length), columns=['Date', feature])

    for i in range(0, len(data)):
        new_dataset["Date"][i] = data['Date'][i]
        new_dataset[feature][i] = data[feature][i]

    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis=1, inplace=True)
    final_dataset = new_dataset.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []
    for i in range(n_lookback, len(final_dataset) - n_forecast + 1):
        x_train_data.append(scaled_data[i-n_lookback:i, 0])
        y_train_data.append(scaled_data[i:i+n_forecast, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1]))

    X_test = scaled_data[-n_lookback:]
    X_test = np.array(X_test).reshape(1, n_lookback)

    match model:
        case "LTSM":
            _model = LTSM_model_build(
                50, x_train_data, y_train_data, n_forecast)
        case "RNN":
            _model = RNN_model_build(
                45, 0.2, x_train_data, y_train_data, n_forecast)
        case "XGBoost":
            test_size = 0.1
            test_ind = int(len(final_dataset) * (1-test_size))

            x_train, y_train = [], []
            for i in range(n_lookback, test_ind - n_forecast + 1):
                x_train.append(scaled_data[i-n_lookback:i, 0])
                y_train.append(scaled_data[i:i+n_forecast, 0])

            x_train, y_train = np.array(
                x_train), np.array(y_train)

            x_train = np.reshape(
                x_train, (x_train.shape[0], x_train.shape[1]))

            x_valid, y_valid = [], []
            for i in range(test_ind-n_forecast+1, len(final_dataset) - n_forecast + 1):
                x_valid.append(scaled_data[i-n_lookback:i, 0])
                y_valid.append(scaled_data[i:i+n_forecast, 0])

            x_valid, y_valid = np.array(x_valid), np.array(y_valid)
            x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1])

            _model = XGBoost_model_build(x_train, y_train, [
                                         (x_train, y_train), (x_valid, y_valid)])

    predicted_closing_price = _model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

    df_past = new_dataset[:]
    t = pd.date_range(
        start=data['Date'][len(data)-1], periods=n_forecast, freq=dt_freq)
    df_future = pd.DataFrame(columns=["Date", "Predictions"])
    df_future["Date"] = t
    df_future.index = df_future["Date"]
    df_future.drop("Date", axis=1, inplace=True)
    df_future["Predictions"] = predicted_closing_price.flatten()

    return df_past, df_future
