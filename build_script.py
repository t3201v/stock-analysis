from datetime import datetime
from context_provider import ContextProvider
from train_models import LSTM_train_forecast_prices, RNN_train_forecast_prices, TATE_train_forecast_prices, XGBoost_train_forecast_prices


# stocks = ['AAPL', 'BTCUSDT', 'META', 'GOOG', 'NFLX', 'TSLA']
stocks = ['AAPL']
# features = ["Close", "PoC", "MA"]
features = ["Close"]
# models = ["XGBoost", "RNN", "LSTM", "TATE"]
models = ["XGBoost"]

ctx = ContextProvider()

for s in stocks:
    df, freq = ctx.get_data(s, datetime(
        2010, 1, 1).isoformat(), datetime.now().isoformat())
    for f in features:
        for m in models:
            if m == "XGBoost":
                XGBoost_train_forecast_prices(df, 60, 15, f, freq, s)
            if m == "RNN":
                RNN_train_forecast_prices(df, 60, 15, f, freq, s)
            if m == "LSTM":
                LSTM_train_forecast_prices(df, 60, 15, f, freq, s)
            # if m == "TATE":
            #     TATE_train_forecast_prices(df, 15, freq, f, s)
