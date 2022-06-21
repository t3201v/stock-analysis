from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from xgboost import XGBClassifier


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


def XGBoost_model_build(x_train, y_train):
    model = XGBClassifier()
    model.fit(x_train, y_train)
    return model
