import pandas as pd
from datetime import datetime as date
import yfinance as yf
import binance as bi
import json


def filter(item: list):
    return item[0:6]


class ContextProvider:
    def __init__(self) -> None:
        self.df = None
        self.stock = None
        self.start_date = None
        self.end_date = None
        self.freq = None
        self.interval = 1

    def get_data(self, stock, start_date, end_date):
        if (self.stock == stock and self.start_date == start_date and self.end_date == end_date):
            return self.df, self.freq
        return self.fetch_data(stock, start_date, end_date)

    def fetch_data(self, stock, start_date, end_date):
        if stock == "BTCUSDT":
            # tmp_end = int(date.fromisoformat(end_date).timestamp() * 1000)
            client = bi.Client()
            res = client.get_klines(
                symbol=stock,
                interval=client.KLINE_INTERVAL_5MINUTE,
                limit=1000,
                endTime=int(date.now().timestamp()*1000)
            )
            fil = list(map(filter, res))
            self.df = pd.DataFrame(
                fil, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            self.df['Open'] = self.df['Open'].astype('float64')
            self.df['High'] = self.df['High'].astype('float64')
            self.df['Low'] = self.df['Low'].astype('float64')
            self.df['Close'] = self.df['Close'].astype('float64')
            self.df['Volume'] = self.df['Volume'].astype('float64')
            self.df["Date"] = pd.to_datetime(self.df["Date"], unit='ms')

            self.freq = "5min"
        else:
            yf_ticker_data = yf.Ticker(stock)
            self.df = yf_ticker_data.history(
                period="1d",
                start=date.fromisoformat(start_date).strftime("%Y-%m-%d"),
                end=date.fromisoformat(end_date).strftime("%Y-%m-%d"))
            self.df = pd.DataFrame(self.df)
            self.df = self.df.reset_index()

            self.freq = "D"

        poc = [100 * (b - a) / a for a,
               b in zip(self.df["Close"][::1], self.df["Close"][1::1])]
        # the beginning is always set 0
        poc.insert(0, 0)
        self.df["PoC"] = poc

        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date

        return self.df, self.freq

    def handle_ws_message(self, message):
        # print(message)
        if isinstance(message, str):
            m = json.loads(message)
            # print(m)
            d = {}
            diff = 0
            at = None
            for key, value in m.items():
                if key == "E":
                    event_time = date.fromtimestamp(value/1000)
                    diff = (event_time -
                            self.df["Date"][len(self.df) - 1]).total_seconds()
                    # print(diff)
                    at = event_time
                if key == "k":
                    for key2, val2 in value.items():
                        if key2 == "o":
                            d["Open"] = float(val2)
                        if key2 == "c":
                            d["Close"] = float(val2)
                        if key2 == "h":
                            d["High"] = float(val2)
                        if key2 == "l":
                            d["Low"] = float(val2)
                        if key2 == "v":
                            d["Volume"] = float(val2)

            # print(new_row)
            print("adding new candlestick in: -" + str(60-diff) + "s")
            if diff > 60:
                d["Date"] = at
                last = self.df["Close"][len(self.df) - 1]
                d["PoC"] = 100*(d["Close"]-last)/last
                new_row = pd.DataFrame(d, index=[0])
                self.df = pd.concat([self.df, new_row], ignore_index=True)
