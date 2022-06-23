from time import time
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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


def XGBoost_model_build(X_train, y_train):
    start_time = time()
    model = XGBRegressor(max_depth=7)
    model.fit(X_train, y_train)
    print('Fit time : ', time() - start_time)
    return model


def forecastingPrice(df, n_lookback, n_forecast, model, feature, dt_freq):
    length = len(df)

    # data = df.sort_index(ascending=True, axis=0)
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
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    match model:
        case "LTSM":
            _model = LTSM_model_build(
                50, x_train_data, y_train_data, n_forecast)
        case "RNN":
            _model = RNN_model_build(
                45, 0.2, x_train_data, y_train_data, n_forecast)
        case "XGBoost":
            _df = df.drop("Date", 1)
            cols = _df.columns.values
            target = feature
            predictors = cols[cols != target]
            X = _df[predictors].values
            y = _df[target].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=n_forecast)

            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            _model = XGBoost_model_build(X_train, y_train)
            predicted_closing_price = _model.predict(X_test)

            train_data = new_dataset[:length-n_forecast]
            valid_data = new_dataset[length-n_forecast:]
            valid_data['Predictions'] = predicted_closing_price
            return train_data, valid_data

    X_test = []
    X_test = scaled_data[-n_lookback:]
    X_test = np.array(X_test).reshape(1, n_lookback, 1)

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
