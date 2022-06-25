from datetime import datetime
import websocket


class StreamKline():
    def __init__(self) -> None:
        self.socket_url = ""
        self._ws_instance = None
        self.isBound=False

    def set_url(self, currency, interval):
        self.socket_url = f'wss://stream.binance.com:9443/ws/{currency}@kline_{interval}'

    def bind_cb_message(self, cb):
        self.cb_message = cb
        self.isBound=True

    def run(self):
        websocket.enableTrace(True)
        self._ws_instance = websocket.WebSocketApp(
            self.socket_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error_binance,
            on_close=self.on_close_binance
        )

        self._ws_instance.run_forever()

    def stop(self):
        self._ws_instance.close()

    def isRunning(self):
        return self._ws_instance is not None and self._ws_instance.keep_running

    def on_open(self, ws):
        print('WS binance opened !!')

    def on_message(self, ws, message):
        # print(message)
        if (self.isBound):
            self.cb_message(message)

    def on_error_binance(self, ws, error):
        print(error)

    def on_close_binance(self, ws, close_status_code, close_msg):
        print("### closed  ###")
