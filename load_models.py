from os import path
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from time_embedding import TATE_load_model, seq_len
import pickle
from keras.models import load_model


def LSTM_load(stock, feature):
    f = f'models/{stock}_{feature}_RNN.h5'
    if path.exists(f):
        lstm_model = load_model(f)
    else:
        raise Exception("Not found file path")
    return lstm_model


def RNN_load(stock, feature):
    f = f'models/{stock}_{feature}_RNN.h5'
    if path.exists(f):
        rnn_model = load_model(f)
    else:
        raise Exception("Not found file path")
    return rnn_model


def XGBoost_load(stock, feature):
    f = f'models/{stock}_{feature}_XGBoost.pkl'
    if path.exists(f):
        model = pickle.load(open(f, "rb"))
    else:
        raise Exception("Not found file path")
    return model


def LSTM_load_forecast_prices(df, n_lookback, n_forecast, feature, dt_freq, stock):
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

    _model = LSTM_load(stock, feature)

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


def RNN_load_forecast_prices(df, n_lookback, n_forecast, feature, dt_freq, stock):
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

    _model = RNN_load(stock, feature)

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


def XGBoost_load_forecast_prices(df, n_lookback, n_forecast, feature, dt_freq, stock):
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

    _model = XGBoost_load(stock, feature)

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


def TATE_load_forecast_prices(df, n_forecast, dt_freq, feature, stock):
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
    df_val = df[(df.index >= last_20pct - n_forecast)].copy()

    # Remove date column
    df_val.drop(columns=['Date'], axis=1, inplace=True)

    # Convert pandas columns into arrays
    val_data = df_val.values

    ###############################################################################

    # Test data
    X_test = val_data[-seq_len:]
    X_test = np.array(X_test, dtype=list)
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
    X_test = X_test.astype(np.float32)

    _model = TATE_load_model(stock)

    test_pred = _model.predict(X_test)
    # print(test_pred)
    pred = test_pred.flatten()*(max_return - min_return) + min_return
    # print(pred)
    pred = pd.DataFrame(pred, columns=['Predictions'])
    pred = pred['Predictions'].add(
        1, fill_value=0).cumprod()*__df['Close'][len(__df)-1]
    # print(pred.values)

    if feature == 'PoC':
        pred = pred.pct_change()
        pred[0] = 0

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
