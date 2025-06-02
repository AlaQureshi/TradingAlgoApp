import websocket
import json
import logging
from datetime import datetime
from threading import Thread, Event

class PolygonWebSocket:
    def __init__(self, api_key, symbols, on_data_callback):
        self.api_key = api_key
        self.symbols = symbols
        self.ws = None
        self.on_data_callback = on_data_callback
        self.logger = logging.getLogger(__name__)
        self.connected = Event()
        self.authenticated = Event()
        
    def is_connected(self):
        return self.connected.is_set() and self.authenticated.is_set()

    def connect(self):
        """Connect to Polygon.io websocket"""
        self.connected.clear()
        self.authenticated.clear()
        
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            "wss://socket.polygon.io/stocks",
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        ws_thread = Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

    def _on_open(self, ws):
        """Handle websocket connection open"""
        self.connected.set()
        self.logger.info("Websocket connection established")
        auth_data = {"action": "auth", "params": self.api_key}
        self.ws.send(json.dumps(auth_data))

    def _on_message(self, ws, message):
        """Handle incoming messages"""
        try:
            data = json.loads(message)
            
            # Handle authentication status
            if isinstance(data, list) and data[0].get('ev') == 'status':
                if data[0].get('status') == 'auth_success':
                    self.authenticated.set()
                    self._subscribe()
                elif data[0].get('status') == 'auth_failed':
                    self.logger.error(f"Authentication failed: {data[0].get('message')}")
                return
            
            # Handle trade data
            if isinstance(data, list) and len(data) > 0 and data[0].get('ev') == 'T':
                trade = data[0]
                processed_data = {
                    'symbol': trade['sym'],
                    'price': float(trade['p']),
                    'size': int(trade['s']),
                    'timestamp': datetime.fromtimestamp(trade['t']/1000)
                }
                self.on_data_callback(processed_data)
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def _subscribe(self):
        """Subscribe to stock channels"""
        subscribe_data = {
            "action": "subscribe",
            "params": ",".join([f"T.{sym}" for sym in self.symbols])
        }
        self.ws.send(json.dumps(subscribe_data))
        self.logger.info(f"Subscribed to: {self.symbols}")

    def disconnect(self):
        """Close websocket connection"""
        if self.ws:
            self.ws.close()
        self.connected.clear()
        self.authenticated.clear()

    def _on_error(self, ws, error):
        self.logger.error(f"Websocket error: {error}")
        self.connected.clear()

    def _on_close(self, ws, close_status_code, close_msg):
        self.logger.info(f"Websocket connection closed: {close_msg}")
        self.connected.clear()
        self.authenticated.clear()
