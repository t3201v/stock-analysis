# Stock/Crypto Analysis Dashboard

This project's meant for ML, tensorflow/keras research purposes. We are not financial experts to ensure what's going on in the stock/crypto market.

---

- This app supports in visualizing multiple kinds of stock like `AAPL`, `META`, `GOOG`, `NFLX`, `TSLA` and `BTCUSDT`

- Users can play around with a couple of models and see how it's going on the graphs. Currently, the app supported a few models in prediction's `LSTM`, `RNN`, `XGBoost` and `Transformer and Time Embedding` _(TATE)_ (only worked with **_Moving Average (MA)_**)

- Some features for prediction parameter are `Close Price`, `Price of Change` _(PoC)_ and `Moving Average` _(MA)_

---

### Project structure

```
root
│ README.md
└── models
│ │ ...
│
│ build_script.py
│ stock_app.py
| ...
```

- `build_script.py`: Build models we use for stock price prediction that will be stored in `./models`
- `stock_app.py`: Our main app content. Start the app with

```
py stock_app.py
```

---

### Websocket

Specifically for `BTCUSDT`, we have integrated a websocket to fetch a new candle around every second from the binance socket stream `wss://stream.binance.com:9443`. The graphs will update every `30s` and will add a new candle into the present dataset if its mounted time higher than the latter of the dataset by `60s`.

Then it will re-predict based on the new dataset and visualize it on our app.

---

### Dev Dependencies

- Made with python 3.10, keras 2.9.0, tensorflow 2.9.1
- Trained `LSTM`, `RNN`, `TATE` models on colab
