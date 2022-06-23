from time import time
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def LTSM_model_build(units, x_train, y_train):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=units, return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
    lstm_model.add(LSTM(units=units))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
    return lstm_model


def RNN_model_build(units, dropout, x_train, y_train):
    rnn_model = Sequential()
    rnn_model.add(LSTM(units=units, return_sequences=True,
                  input_shape=(x_train.shape[1], 1)))
    rnn_model.add(Dropout(dropout))
    for i in [True, True, False]:
        rnn_model.add(LSTM(units=units, return_sequences=i))
        rnn_model.add(Dropout(dropout))

    rnn_model.add(Dense(units=1))
    rnn_model.compile(optimizer='adam', loss='mean_squared_error')
    rnn_model.fit(x_train, y_train, epochs=10, batch_size=32)
    return rnn_model


def XGBoost_model_build(X_train, y_train):
    start_time = time()
    model = XGBRegressor(max_depth=7)
    model.fit(X_train, y_train)
    print('Fit time : ', time() - start_time)
    return model


def forecastingPrice(df, start_ind, offset, model, feature):
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

    train_data = final_dataset[0:length-offset, :]
    valid_data = final_dataset[length-offset:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []

    for i in range(start_ind, len(train_data)):
        x_train_data.append(scaled_data[i-start_ind:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    inputs_data = new_dataset[len(new_dataset) -
                              len(valid_data)-start_ind:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    match model:
        case "LTSM":
            _model = LTSM_model_build(50, x_train_data, y_train_data)
        case "RNN":
            _model = RNN_model_build(45, 0.2, x_train_data, y_train_data)
        case "XGBoost":
            _df = df.drop("Date", 1)
            cols = _df.columns.values
            target = feature
            predictors = cols[cols != target]
            X = _df[predictors].values
            y = _df[target].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=offset)

            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            _model = XGBoost_model_build(X_train, y_train)
            predicted_closing_price = _model.predict(X_test)

            train_data = new_dataset[:length-offset]
            valid_data = new_dataset[length-offset:]
            valid_data['Predictions'] = predicted_closing_price
            return train_data, valid_data

    X_test = []
    for i in range(start_ind, inputs_data.shape[0]):
        X_test.append(inputs_data[i-start_ind:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_closing_price = _model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

    train_data = new_dataset[:length-offset]
    valid_data = new_dataset[length-offset:]
    valid_data['Predictions'] = predicted_closing_price
    return train_data, valid_data
