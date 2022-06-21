import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import date
import yfinance as yf


app = dash.Dash()
server = app.server

app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Div([
            dcc.Dropdown(id="stock-dropdown",
                         options=[{'label': 'Apple', 'value': 'AAPL'},
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

# app callback visualizes closing values


@app.callback(
    Output("actual-data-graph", "figure"),
    [Input("graph-type-dropdown", "value"),
     Input("stock-dropdown", "value"),
     Input("period-dropdown", "value"),
     Input("date-picker-range", "start_date"),
     Input("date-picker-range", "end_date")]
)
def update_graph(graph_type, stock, period, start_date, end_date):
    yf_ticker_data = yf.Ticker(stock)
    df = yf_ticker_data.history(period=period, start=start_date, end=end_date)

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
