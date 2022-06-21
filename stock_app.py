import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime as date
import yfinance as yf
import binance as bi


app = dash.Dash()
server = app.server

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
                         value="TSLA", multi=False, clearable=False),

            dcc.Dropdown(id="graph-type-dropdown",
                         options=[{'label': 'Close Price', 'value': 'Close'},
                                  {'label': 'Candlestick', 'value': 'Candle'},
                                  {'label': 'Volume', 'value': 'Volume'}],
                         style={"width": "10rem",
                                "marginLeft": "4px"},
                         value="Close", multi=False, clearable=False),

            dcc.Dropdown(["XGBoost", "RNN", "LTSM"],
                         id="model-dropdown",
                         style={"width": "10rem",
                                "marginLeft": "4px"},
                         value="LTSM", multi=False, clearable=False),

            dcc.Dropdown(["Close", "PoC"],
                         id="feature-dropdown",
                         style={"width": "10rem",
                                "marginLeft": "4px"},
                         value="Close", multi=False, clearable=False),

            dcc.Dropdown(["1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd"],
                         id="period-dropdown",
                         style={"width": "10rem",
                                "marginLeft": "4px"},
                         value="1d", multi=False, clearable=False),

            dcc.Dropdown(["1m", "2m", "5m", "15m", "30m", "1h", "1d", "5d", "1wk"],
                         id="interval-dropdown",
                         style={"width": "10rem",
                                "marginLeft": "4px"},
                         value="5m", multi=False, clearable=False),

            html.Div([
                dcc.DatePickerRange(
                    id='date-picker-range',
                    min_date_allowed=date(2015, 1, 1),
                    max_date_allowed=date.today(),
                    start_date=date(2020, 1, 1),
                    end_date=date.today(),
                    number_of_months_shown=2,
                )], style={"marginLeft": "8px"}),
        ], style={"display": "flex"}),

        html.H2("Actual closing price", style={"textAlign": "center"}),
        dcc.Graph(
            id="actual-data-graph",
        ),

        html.H2("LSTM Predicted closing price",
                id="model-label",
                style={"textAlign": "center"}),
        dcc.Graph(
            id="predicted-data-graph",
        )
    ])

])


def filter(item: list):
    return item[0:6]

# app callback visualizes closing values


@app.callback(
    Output("actual-data-graph", "figure"),
    [Input("graph-type-dropdown", "value"),
     Input("stock-dropdown", "value"),
     Input("period-dropdown", "value"),
     Input("interval-dropdown", "value"),
     Input("date-picker-range", "start_date"),
     Input("date-picker-range", "end_date")]
)
def update_graph(graph_type, stock, period, interval, start_date, end_date):
    if stock == "BTCUSDT":
        tmp_start = int(
            round(date.fromisoformat(start_date).timestamp())) * 1000
        tmp_end = int(round(date.fromisoformat(end_date).timestamp())) * 1000
        client = bi.Client()
        res = client.get_klines(
            symbol=stock,
            interval=client.KLINE_INTERVAL_5MINUTE,
            limit=1000,
            startTime=tmp_start,
            endTime=tmp_end)
        fil = list(map(filter, res))
        df = pd.DataFrame(
            fil, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Open'] = df['Open'].astype('float64')
        df['High'] = df['High'].astype('float64')
        df['Low'] = df['Low'].astype('float64')
        df['Close'] = df['Close'].astype('float64')
        df['Volume'] = df['Volume'].astype('float64')
        df["Date"] = pd.to_datetime(df["Date"], unit='ms')
        df.index = df["Date"]
    else:
        yf_ticker_data = yf.Ticker(stock)
        df = yf_ticker_data.history(
            period=period,
            start=date.fromisoformat(start_date).strftime("%Y-%m-%d"),
            end=date.fromisoformat(end_date).strftime("%Y-%m-%d"))

    match graph_type:
        case "Close":
            tmp = go.Scatter(x=df.index, y=df['Close'])
            title = f"{stock} closing values"
        case "Candle":
            tmp = go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'])
            title = f"{stock} Candlestick chart"
        case "Volume":
            tmp = go.Scatter(x=df.index, y=df['Volume'])
            title = f"{stock} volume values"

    figure = {"data": [tmp],
              "layout": {"title": title}}
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
