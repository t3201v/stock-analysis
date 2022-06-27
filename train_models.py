from time import time
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from time_embedding import TATE_train_model, seq_len
import pickle


def LSTM_build(units, x_train, y_train, dense, stock, feature):
    f = f'models/{stock}_{feature}_LSTM.h5'
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=units, return_sequences=True,
                        input_shape=(x_train.shape[1], 1)))
    lstm_model.add(LSTM(units=units))
    lstm_model.add(Dense(dense))

    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    lstm_model.save(f)

    return lstm_model


def RNN_build(units, dropout, x_train, y_train, dense, stock, feature):
    f = f'models/{stock}_{feature}_RNN.h5'
    rnn_model = Sequential()
    rnn_model.add(LSTM(units=units, return_sequences=True,
                       input_shape=(x_train.shape[1], 1)))
    rnn_model.add(Dropout(dropout))
    for i in [True, True, False]:
        rnn_model.add(LSTM(units=units, return_sequences=i))
        rnn_model.add(Dropout(dropout))

    rnn_model.add(Dense(units=dense))
    rnn_model.compile(optimizer='adam', loss='mean_squared_error')
    rnn_model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    rnn_model.save(f)

    return rnn_model


def XGBoost_build(X_train, y_train, eval_set, stock, feature):
    f = f'models/{stock}_{feature}_XGBoost.pkl'
    start_time = time()
    model = XGBRegressor(max_depth=7)
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    pickle.dump(model, open(f, "wb"))
    print('Fit time : ', time() - start_time)
    return model


def LSTM_train_forecast_prices(df, n_lookback, n_forecast, feature, dt_freq, stock):
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

    _model = LSTM_build(
        50, x_train_data, y_train_data, n_forecast, stock, feature)

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


def RNN_train_forecast_prices(df, n_lookback, n_forecast, feature, dt_freq, stock):
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

    _model = RNN_build(
        45, 0.2, x_train_data, y_train_data, n_forecast, stock, feature)

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


def XGBoost_train_forecast_prices(df, n_lookback, n_forecast, feature, dt_freq, stock):
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

    X_test = scaled_data[-n_lookback:]
    X_test = np.array(X_test).reshape(1, n_lookback)

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

    _model = XGBoost_build(x_train, y_train, [
        (x_train, y_train), (x_valid, y_valid)], stock, feature)

    predicted_closing_price = _model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)
    print(predicted_closing_price)

    df_past = new_dataset[:]
    t = pd.date_range(
        start=data['Date'][len(data)-1], periods=n_forecast, freq=dt_freq)
    df_future = pd.DataFrame(columns=["Date", "Predictions"])
    df_future["Date"] = t
    df_future.index = df_future["Date"]
    df_future.drop("Date", axis=1, inplace=True)
    df_future["Predictions"] = predicted_closing_price.flatten()

    return df_past, df_future


def TATE_train_forecast_prices(df, n_forecast, dt_freq, feature, stock):
    __df = df
    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[cols].copy()

    # '''Calculate percentage change'''

    df['Open'] = df['Open'].pct_change()  # Create arithmetic returns column
    df['High'] = df['High'].pct_change()  # Create arithmetic returns column
    df['Low'] = df['Low'].pct_change()  # Create arithmetic returns column
    df['Close'] = df['Close'].pct_change()  # Create arithmetic returns column
    df['Volume'] = df['Volume'].pct_change()

    df.dropna(how='any', axis=0, inplace=True)  # Drop all rows with NaN values

    ###############################################################################
    # '''Create indexes to split dataset'''

    times = sorted(df.index.values)
    # Last 20% of series
    last_20pct = sorted(df.index.values)[-int(0.2*len(times))]

    ###############################################################################
    # '''Normalize price columns'''

    min_return = min(df[(df.index < last_20pct)]
                     [['Open', 'High', 'Low', 'Close']].min(axis=0))
    max_return = max(df[(df.index < last_20pct)]
                     [['Open', 'High', 'Low', 'Close']].max(axis=0))

    # Min-max normalize price columns (0-1 range)
    df['Open'] = (df['Open'] - min_return) / (max_return - min_return)
    df['High'] = (df['High'] - min_return) / (max_return - min_return)
    df['Low'] = (df['Low'] - min_return) / (max_return - min_return)
    df['Close'] = (df['Close'] - min_return) / (max_return - min_return)

    ###############################################################################
    # '''Normalize volume column'''

    min_volume = df[(df.index < last_20pct)]['Volume'].min(axis=0)
    max_volume = df[(df.index < last_20pct)]['Volume'].max(axis=0)

    # Min-max normalize volume columns (0-1 range)
    df['Volume'] = (df['Volume'] - min_volume) / (max_volume - min_volume)

    ###############################################################################
    # '''Create training, validation and test split'''

    # Training data are 80% of total data
    df_train = df[(df.index < last_20pct - n_forecast)].copy()
    df_val = df[(df.index >= last_20pct - n_forecast)].copy()

    # Remove date column
    df_train.drop(columns=['Date'], axis=1, inplace=True)
    df_val.drop(columns=['Date'], axis=1, inplace=True)

    # scaled data
    scaled_data = df['Close'].values

    # Convert pandas columns into arrays
    train_data = df_train.values
    val_data = df_val.values

    # Training data
    X_train, y_train = [], []
    for i in range(seq_len, len(train_data)):
        # Chunks of training data with a length of 128 df-rows
        X_train.append(train_data[i-seq_len:i])
        # Value of the feature that we work on
        y_train.append(scaled_data[i:i+n_forecast])
    X_train, y_train = np.array(
        X_train, dtype=list), np.array(y_train, dtype=list)
    X_train = X_train.reshape(X_train.shape[0], seq_len, X_train.shape[2])
    y_train = y_train.reshape(y_train.shape[0], n_forecast)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    ###############################################################################

    # Validation data
    X_val, y_val = [], []
    for i in range(seq_len, len(val_data) - n_forecast + 1):
        X_val.append(val_data[i-seq_len:i])
        y_val.append(scaled_data[i:i+n_forecast])
    X_val, y_val = np.array(X_val, dtype=list), np.array(y_val, dtype=list)
    X_val = X_val.reshape(X_val.shape[0], seq_len, X_val.shape[2])
    y_val = y_val.reshape(y_val.shape[0], n_forecast)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)

    ###############################################################################

    # Test data
    X_test = val_data[-seq_len:]
    X_test = np.array(X_test, dtype=list)
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
    X_test = X_test.astype(np.float32)
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)

    _model = TATE_train_model(X_train, y_train, X_val,
                              y_val, n_forecast, stock)

    test_pred = _model.predict(X_test)
    # print(test_pred)
    pred = test_pred.flatten()*(max_return - min_return) + min_return
    # print(pred)
    pred = pd.DataFrame(pred, columns=['Predictions'])
    pred = pred['Predictions'].add(
        1, fill_value=0).cumprod()*__df['Close'][len(__df)-1]
    # print(pred.values)

    if feature == 'PoC':
        pred['Predictions'] = pred['Predictions'].pct_change()
        pred['Predictions'][0] = 0

    df_past = __df[:]
    t = pd.date_range(
        start=__df['Date'][len(__df)-1], periods=n_forecast, freq=dt_freq)
    df_future = pd.DataFrame(columns=["Date", "Predictions"])
    df_future["Date"] = t
    df_future.index = df_future["Date"]
    df_future.drop("Date", axis=1, inplace=True)
    df_future["Predictions"] = pred.values
    # print(df_future)

    return df_past, df_future
