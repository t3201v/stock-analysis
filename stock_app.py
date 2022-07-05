import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from datetime import datetime as date
from binance_socket import StreamKline
from context_provider import ContextProvider
from load_models import LSTM_load_forecast_prices, RNN_load_forecast_prices, TATE_load_forecast_prices, XGBoost_load_forecast_prices


app = dash.Dash()
server = app.server
stream = StreamKline()
ctx = ContextProvider()
stream.bind_cb_message(ctx.handle_ws_message)
n_lookback = 60
n_forecast = 15

app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Div([
            dcc.Dropdown(id="stock-dropdown",
                         options=[{'label': 'Apple', 'value': 'AAPL'},
                                  {'label': 'BTCUSDT', 'value': 'BTCUSDT'},
                                  {'label': 'Metaverse/Facebook', 'value': 'META'},
                                  {'label': 'Google', 'value': 'GOOG'},
                                  {'label': 'Netflix', 'value': 'NFLX'},
                                  {'label': 'Tesla', 'value': 'TSLA'}],
                         style={"width": "10rem"},
                         value="NFLX", multi=False, clearable=False),

            dcc.Dropdown(id="graph-type-dropdown",
                         options=[{'label': 'Close Price', 'value': 'Close'},
                                  {'label': 'Candlestick', 'value': 'Candle'},
                                  {'label': 'Price of Change', 'value': 'PoC'},
                                  {'label': 'Volume', 'value': 'Volume'}],
                         style={"width": "10rem",
                                "marginLeft": "4px"},
                         value="Close", multi=False, clearable=False),

            html.Div([
                dcc.DatePickerRange(
                    id='date-picker-range',
                    minimum_nights=50,
                    min_date_allowed=date(2015, 1, 1),
                    max_date_allowed=date.today(),
                    start_date=date(2019, 1, 1),
                    end_date=date.today(),
                    number_of_months_shown=2,
                )], style={"marginLeft": "8px"}),
        ], style={"display": "flex", "justifyContent": "center"}),

        html.Div([
            html.Div([
                dcc.Dropdown(["XGBoost", "RNN", "LSTM", "TATE"],
                             id="model-dropdown",
                             style={"width": "100%", "margin": "0 auto"},
                             value="XGBoost", multi=True, clearable=True),
            ], style={"width": "50%"})
        ], style={
            "display": "flex",
            "width": "100%",
            "justifyContent": "center",
            "margin": "1rem auto",
        }),

        html.Div([
            html.Div([
                dcc.Dropdown(options=[
                    {'label': 'Close Price', 'value': 'Close'},
                    {'label': 'Price of Change', 'value': 'PoC'},
                    {'label': 'Moving Average', 'value': 'MA'}
                ],
                    id="feature-dropdown",
                    style={"width": "100%", "margin": "0 auto"},
                    value="Close", multi=True, clearable=True),
            ], style={"width": "50%"})
        ], style={
            "display": "flex",
            "width": "100%",
            "justifyContent": "center",
            "margin": "1rem auto",
        }),

        html.H2("Actual closing price", style={"textAlign": "center"}),
        dcc.Graph(
            id="actual-data-graph",
        ),

        html.H2("Stock Forecasting",
                id="model-predict-label", style={"textAlign": "center"}),
        dcc.Graph(
            id="predicted-data-graph",
        ),

        html.Div(id="ws"),
        dcc.Interval(
            id='interval-comp',
            interval=30*1000,  # in milliseconds
            n_intervals=0
        ),

    ])

])


# app callback visualizes predicted close price


@app.callback(
    Output("actual-data-graph", "figure"),
    [
        Input("graph-type-dropdown", "value"),
        Input("stock-dropdown", "value"),
        Input("date-picker-range", "start_date"),
        Input("date-picker-range", "end_date"),
        Input("interval-comp", "n_intervals"),
    ]
)
def update_graph(graph_type, stock, start_date, end_date, n):
    df, _ = ctx.get_data(stock, start_date, end_date)

    match graph_type:
        case "Close":
            realistic_data_go = go.Scatter(
                x=df["Date"], y=df['Close'], name="actual")
            title = f"{stock} closing values"
        case "Candle":
            realistic_data_go = go.Candlestick(x=df["Date"],
                                               open=df['Open'],
                                               high=df['High'],
                                               low=df['Low'],
                                               close=df['Close'],
                                               name="actual")
            title = f"{stock} Candlestick chart"
        case "Volume":
            realistic_data_go = go.Scatter(
                x=df["Date"], y=df['Volume'], name="actual")
            title = f"{stock} volume values"
        case "PoC":
            realistic_data_go = go.Scatter(
                x=df["Date"], y=df['PoC'], name="actual")
            title = f"{stock} Price of Change values"

    figure = {"data": [realistic_data_go],
              "layout": {"title": title}}

    return figure


@app.callback(
    Output("predicted-data-graph", "figure"),
    Output("model-predict-label", "children"),
    [
        Input("graph-type-dropdown", "value"),
        Input("stock-dropdown", "value"),
        Input("model-dropdown", "value"),
        Input("feature-dropdown", "value"),
        Input("date-picker-range", "start_date"),
        Input("date-picker-range", "end_date"),
        Input("interval-comp", "n_intervals"),
    ]
)
def update_graph(graph_type, stock, models, features, start_date, end_date, n):
    df, freq = ctx.get_data(stock, start_date, end_date)

    if isinstance(models, str):
        models = [models]

    if isinstance(features, str):
        features = [features]

    label = "Stock Forecasting"

    match graph_type:
        case "Close":
            realistic_data_go = go.Scatter(
                x=df["Date"], y=df['Close'], name="actual")
            title = f"{stock} closing values"
        case "Candle":
            realistic_data_go = go.Candlestick(x=df["Date"],
                                               open=df['Open'],
                                               high=df['High'],
                                               low=df['Low'],
                                               close=df['Close'],
                                               name="actual")
            title = f"{stock} Candlestick chart"
        case "Volume":
            realistic_data_go = go.Scatter(
                x=df["Date"], y=df['Volume'], name="actual")
            title = f"{stock} volume values"
        case "PoC":
            realistic_data_go = go.Scatter(
                x=df["Date"], y=df['PoC'], name="actual")
            title = f"{stock} Price of Change values"

    data_plot = [realistic_data_go]

    for feature in features:
        for m in models:
            match m:
                case "LSTM":
                    train_data, valid_data = LSTM_load_forecast_prices(
                        df=df, n_lookback=n_lookback, n_forecast=n_forecast, feature=feature, dt_freq=freq, stock=stock)
                    train_data_go = go.Scatter(
                        x=train_data.index, y=train_data[feature], name=f"{m}, {feature} train")
                    predicted_data_go = go.Scatter(
                        x=valid_data.index, y=valid_data["Predictions"], name=f"{m}, {feature} predicted")

                    data_plot.append(train_data_go)
                    data_plot.append(predicted_data_go)

                case "RNN":
                    train_data, valid_data = RNN_load_forecast_prices(
                        df=df, n_lookback=n_lookback, n_forecast=n_forecast, feature=feature, dt_freq=freq, stock=stock)
                    train_data_go = go.Scatter(
                        x=train_data.index, y=train_data[feature], name=f"{m}, {feature} train")
                    predicted_data_go = go.Scatter(
                        x=valid_data.index, y=valid_data["Predictions"], name=f"{m}, {feature} predicted")

                    data_plot.append(train_data_go)
                    data_plot.append(predicted_data_go)

                case "XGBoost":
                    train_data, valid_data = XGBoost_load_forecast_prices(
                        df=df, n_lookback=n_lookback, n_forecast=n_forecast, feature=feature, dt_freq=freq, stock=stock)
                    train_data_go = go.Scatter(
                        x=train_data.index, y=train_data[feature], name=f"{m}, {feature} train")
                    predicted_data_go = go.Scatter(
                        x=valid_data.index, y=valid_data["Predictions"], name=f"{m}, {feature} predicted")

                    data_plot.append(train_data_go)
                    data_plot.append(predicted_data_go)

                case "TATE":
                    train_data, valid_data = TATE_load_forecast_prices(
                        df=df, n_forecast=n_forecast,  dt_freq=freq, feature=feature, stock=stock)
                    train_data_go = go.Scatter(
                        x=train_data.index, y=train_data[feature], name=f"{m}, {feature} train")
                    predicted_data_go = go.Scatter(
                        x=valid_data.index, y=valid_data["Predictions"], name=f"{m}, {feature} predicted")

                    data_plot.append(train_data_go)
                    data_plot.append(predicted_data_go)

    figure = {"data": data_plot,
              "layout": {"title": title}}

    return figure, label


# app callback websocket fetch new candle


@app.callback(
    Output("ws", "children"),
    [
        Input("stock-dropdown", "value"),
    ]
)
def update_graph(stock):
    if stock == "BTCUSDT":
        # setup socket update data every interval of time
        if (stream.isRunning()):
            stream.stop()
        stream.set_url(stock.lower(), "5m")
        stream.run()

    else:
        if (stream.isRunning()):
            stream.stop()
    return dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)
